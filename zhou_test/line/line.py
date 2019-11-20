#!/usr/bin/python
# coding=utf-8

import math
import numpy as np
from itertools import combinations

import cv2
from scipy.signal import argrelmax
from sklearn import linear_model
from sklearn.cluster import DBSCAN

from data.utils import checkAnyNone
from geometry.line import GLine
from geometry.object import GObject, groupData
from geometry.point import GPoint, boundingPoint, PointsToNdarry, getPtsMaxDis, getMinDisPts


# 线
class GLine(GObject):
    # f(x) = ax + by + c = 0
    minSampleSize = 2  # 最少构造点的数量

    def __init__(self, pt1=None, pt2=None):
        super(GLine, self).__init__()
        self.x1, self.y1, self.x2, self.y2 = 0, 0, 0, 0
        self.a, self.b, self.c, self.d, self.k = 0, 0, 0, 0, 0
        self.pt1, self.pt2 = None, None

        self.inPts, self.outPts = [], []  # 拟合的时候用
        if pt1 and pt2: self.initPts(pt1, pt2)

    # 使用坐标点初始化
    def initXY(self, x1, y1, x2, y2):
        self.x1 = int(x1)
        self.y1 = int(y1)
        self.x2 = int(x2)
        self.y2 = int(y2)
        # 直线的一般式：Ax + By + C = 0
        # 由两点求一般式 (y2-y1)x-(x2-x1)y-x1y2+x2y1=0
        # A = y2-y1,  B = x1 - x2
        # 法向量 n = (A,B) 与直线垂直
        # 法向量和直线的交点, p_vector = (-C*A / math.sqrt(A**2 + B**2), -C*B / math.sqrt(A**2 + B**2)))
        # 如果 C == 0, p_vector = (0, 0)
        # 方向向量 a = (B, -A)
        self.a = float(y2 - y1)
        self.b = float(x1 - x2)
        self.c = float(x2 * y1 - y2 * x1)
        # 点到直线的距离 d = abs(Ax_0 + By_0 + C) / math.sqrt(A**2 + B**2)
        self.d = math.sqrt(self.a ** 2 + self.b ** 2)
        self.pt1 = GPoint(x1, y1)
        self.pt2 = GPoint(x2, y2)
        self.k = (self.a / -self.b) if self.b else (self.a / 0.000001)
        return self

    # 转字符串函数
    def __str__(self):
        if checkAnyNone(self.pt1, self.pt2):
            return 'GLine a:%.3f b:%.3f c:%.3f' % (self.a, self.b, self.c)
        else:
            return 'GLine X1:%d  Y1:%d  X2:%d  Y2:%d' % (self.x1, self.y1, self.x2, self.y2)

    # 使用GPoint对象初始化
    def initPts(self, pt1, pt2):
        self.initXY(pt1.x, pt1.y, pt2.x, pt2.y)
        return self

    # 直线的内点距离容差
    def getInlierTolerance(self):
        s = self.pt1.distance(self.pt2)
        return max(math.sqrt(2), s * 0.005)

    # 获取点的极性, 极性相同表示在直线的同边上, 反之在两边, 0为在线上  返回值为-1  0  1
    def getPolarity(self, pt):
        val = self.a * pt.x + self.b * pt.y + self.c
        if val > 1e-8:
            return 1
        elif val < -1e-8:
            return -1
        else:
            return 0

    # 与另外直线求交叉点
    def crossPoint(self, line):
        d = self.a * line.b - line.a * self.b
        if abs(d) < 1e-8: return None  # 两条直线平行或共线
        x = (self.b * line.c - line.b * self.c) / d
        y = (self.c * line.a - line.c * self.a) / d
        return GPoint(x, y)

    # 根据y值, 计算x值
    def getX(self, y):
        return self.x1 if self.a == 0 else (self.b * y + self.c) * -1.0 / self.a

    # 根据x值, 计算y值
    def getY(self, x):
        return self.y1 if self.b == 0 else (self.a * x + self.c) * -1.0 / self.b

    # 线段长度
    def getLen(self):
        if checkAnyNone(self.pt1, self.pt2): return -1
        return self.pt1.distance(self.pt2)

    # 两条线的中线
    def midLine(self, line):
        if checkAnyNone(self.pt1, self.pt2, line.pt1, line.pt2): return None
        if self.pt1.distance(line.pt1) > self.pt1.distance(line.pt2):
            return GLine(self.pt1.midpoint(line.pt2), self.pt2.midpoint(line.pt1))
        else:
            return GLine(self.pt1.midpoint(line.pt1), self.pt2.midpoint(line.pt2))

    # 线段的中点
    def midPoint(self):
        if checkAnyNone(self.pt1, self.pt2): return None
        return GPoint((self.x1 + self.x2) / 2., (self.y1 + self.y2) / 2.)

    # 绘制线段
    def draw(self, img, color=255, thickness=1, **kwargs):
        if checkAnyNone(self.pt1, self.pt2): return
        cv2.line(img, self.pt1.getPoint(), self.pt2.getPoint(), color, thickness, **kwargs)

    # 计算点到直线的欧式距离
    # 参数: pt:参与计算的点   mode:计算模式 0:点到直线的距离    1:点到线段的距离
    def distance(self, pt, mode=0):
        if mode > 0 and checkAnyNone(self.pt1, self.pt2): mode = 0
        if mode == 0:  # 点到直线的垂直距离, 线上最近距离点可能在延长线上
            if self.d:
                return abs(self.a * pt.x + self.b * pt.y + self.c) / self.d
            else:
                return np.sqrt(np.power(self.pt1.x - pt.x, 2) + np.power(self.pt1.y - pt.y, 2))
        elif mode == 1:  # 点到线段的距离, 线上最近距离点在线段上
            x, y = pt.x, pt.y
            x1, y1, x2, y2 = self.x1, self.y1, self.x2, self.y2
            cross = (x2 - x1) * (x - x1) + (y2 - y1) * (y - y1)
            if cross <= 0: return math.sqrt((x - x1) * (x - x1) + (y - y1) * (y - y1))
            d2 = (x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1)
            if cross >= d2: return math.sqrt((x - x2) ** 2 + (y - y2) ** 2)
            r = cross / float(d2)
            px = x1 + (x2 - x1) * r
            py = y1 + (y2 - y1) * r
            return math.sqrt((x - px) ** 2 + (py - y) ** 2)
        else:
            print('line.distance模式不支持 %d' % (mode))

    # 计算线段与线段的距离的最小距离
    # 参数: line:参与计算的另外一条线段 类型为GLine,  mode:计算模式 0:两线段中间点距离  1:两线段之间的最小欧式距离
    def distanceWithLine(self, line):
        if checkAnyNone(self.pt1, self.pt2, line.pt1, line.pt2): return -1
        if isLineSegCross(self.pt1, self.pt2, line.pt1, line.pt2): return 0
        return min(self.distance(line.pt1, mode=1), self.distance(line.pt2, mode=1),
                   line.distance(self.pt1, mode=1), line.distance(self.pt2, mode=1))

    # 与另外直线的夹角(度数), 返回值范围为0到90度
    def angle(self, line):
        if self == line: return 0
        dot = abs(self.a * line.a + self.b * line.b)
        moda = math.sqrt(self.a ** 2 + self.b ** 2)
        modb = math.sqrt(line.a ** 2 + line.b ** 2)
        if moda * modb == 0:
            cosval = 10000
        else:
            cosval = dot / (moda * modb)
        if cosval > 1.0: cosval = 1.0
        return math.degrees(math.acos(cosval))

    # 获取直线与X轴的夹角, 返回值为0到90度
    def getXAngle(self):
        line = GLine()
        line.a, line.b, line.c = 0, 1, 0
        return self.angle(line)

    # 获取直线与Y轴的夹角, 返回值为0到90度
    def getYAngle(self):
        line = GLine()
        line.a, line.b, line.c = 1, 0, 0
        return self.angle(line)

    # 平移
    def move(self, x, y):
        if checkAnyNone(self.pt1, self.pt2): return self
        pt1 = self.pt1.move(x, y)
        pt2 = self.pt2.move(x, y)
        self.initPts(pt1, pt2)
        return self

    ##
    # 旋转(顺时针旋转)
    # @param rad： 旋转弧度
    # @param rotate_point： 旋转中心，默认线段的中心点
    # #
    def rotateUseRad(self, rad, rotate_point=None):
        if checkAnyNone(self.pt1, self.pt2): return self
        if not rotate_point: rotate_point = self.midPoint()
        pt1 = self.pt1.rotate(rad, rotate_point)
        pt2 = self.pt2.rotate(rad, rotate_point)
        self.initPts(pt1, pt2)
        return self

    def rotateUseMatrix(self, m):
        """
        旋转直线
        :param m: 旋转矩阵
        :return: 旋转后的直线
        """
        from utility.utils import translate_point
        pt1 = translate_point(m, np.asarray(self.pt1.getPoint()))
        pt2 = translate_point(m, np.asarray(self.pt2.getPoint()))
        self.initXY(pt1[0, 0], pt1[0, 1], pt2[0, 0], pt2[0, 1])
        return self

    # 获取线上的点,返回值不包含端点的GPoint列表, includeEndPt, 是否包含端点
    def getLinePts(self, includeEndPt=False):
        isX = abs(self.x1 - self.x2) > abs(self.y1 - self.y2)
        begin, end = (self.x1, self.x2) if isX else (self.y1, self.y2)
        if begin > end: begin, end = end, begin
        if isX:
            pts = [GPoint(x, self.getY(x)) for x in range(begin + 1, end)]
        else:
            pts = [GPoint(self.getX(y), y) for y in range(begin + 1, end)]
        if includeEndPt: pts.extend([self.pt1, self.pt2])
        return pts

    # 获取经过某个点的垂线
    def getVertLine(self, pt):
        line = GLine()
        line.a, line.b, line.c = -self.b, self.a, self.b * pt.fx - self.a * pt.fy
        line.d = math.sqrt(line.a ** 2 + line.b ** 2)
        return line

    # 获取平行线
    def getParaLine(self, pt):
        line = GLine()
        line.a, line.b, line.c = self.a, self.b, -self.a * pt.fx - self.b * pt.fy
        line.d = math.sqrt(line.a ** 2 + line.b ** 2)
        return line

    # 获取延长线段,延长后的线段x*ratio,y*ratio, 相当于
    def getExtendLine(self, ratio=1.5):
        midPt = self.midPoint()
        x1, y1 = self.pt1.x + (self.pt1.x - midPt.x) * ratio, self.pt1.y + (self.pt1.y - midPt.y) * ratio
        x2, y2 = self.pt2.x + (self.pt2.x - midPt.x) * ratio, self.pt2.y + (self.pt2.y - midPt.y) * ratio
        return GLine().initXY(x1, y1, x2, y2)

    # 获取pt到直线的垂直点,
    def getVertPt(self, pt):
        # https://blog.csdn.net/hsg77/article/details/90376109
        a2, b2, ab, ac, bc = self.a ** 2, self.b ** 2, self.a * self.b, self.a * self.c, self.b * self.c
        x = (b2 * pt.fx - ab * pt.fy - ac) / (a2 + b2)
        y = (a2 * pt.fy - ab * pt.fx - bc) / (a2 + b2)
        return GPoint(x, y)

    def relation(self, line):
        """
        计算线与线的关系
        :param line:
        :return: -1: line位于self的top， 1：line位于self的bottom， 0：相交
        """

        def isin(a, c):
            return (a.pt1.x < c.pt1.x < a.pt2.x or a.pt1.x < c.pt2.x < a.pt2.x) or \
                   (c.pt1.x < a.pt1.x < c.pt2.x or c.pt1.x < a.pt2.x < c.pt2.x)

        if line is None:
            return 0
        b = self.pt1.y - self.k * self.pt1.x
        r1 = np.sign(line.pt1.x * self.k + b - line.pt1.y)
        r2 = np.sign(line.pt2.x * self.k + b - line.pt2.y)
        return (r1 + r2) if not r2 or not r1 else r1 if r1 == r2 else 0

    ################################################################################
    # 判断函数
    ################################################################################

    # 线段是否相交
    def isCross(self, line):
        # https://www.cnblogs.com/wuwangchuxin0924/p/6218494.html
        return isLineSegCross(self.pt1, self.pt2, line.pt1, line.pt2)

    # 是否点在直线上
    def isSameLine(self, pt, thresh=0.1):
        p = self.a * pt.fx + self.b * pt.fy + self.c
        return p < thresh

    # 点是否在线段上
    def isPtAtLineSeg(self, pt, maxRatio=0.01, minDis=1.42):
        dis = self.distance(pt, mode=1)  # 点到线段的距离
        th = max(self.getLen() * maxRatio, minDis)
        return dis <= th

    # 与另外一条线段是否平行
    # 参数line:待检测的直线, pddr:平行距离差的阈值比例(小于1), 固定值(大于1), overlapRatio:重叠阈值比例  平行线距离(lineMindis, lineMaxdis)
    def isParallel(self, line, PDDR=0.05, overlapRatio=0.3, lineMindis=-1, lineMaxdis=-1):
        # 考虑因素,直线距离, 重合率, 线段长度差异, 平行距离差异

        longLine, shortLine = (self, line) if self.getLen() > line.getLen() else (line, self)
        llen, slen = longLine.getLen(), shortLine.getLen()
        if float(slen) / llen < overlapRatio: return False

        pts = [shortLine.pt1, shortLine.pt2]
        lineDis = [longLine.distance(p, mode=0) for p in pts]  # 直线距离
        lineDif = lineDis[0] + lineDis[1] if longLine.isCross(shortLine) else abs(lineDis[0] - lineDis[1])

        # 线是否平行
        if PDDR < 1 and lineDif > slen * PDDR: return False
        if PDDR > 1 and lineDif > PDDR: return False

        if overlapRatio <= 0: return True

        # 平行线的距离大于两线距离, 为干扰线
        if sum(lineDis) / 2 > slen + llen: return False
        if lineMindis > 0 and lineDis < lineMindis: return False
        if lineMaxdis > 0 and lineDis > lineMaxdis: return False

        # 计算重合率
        pt1 = longLine.getVertPt(shortLine.pt1)
        pt2 = longLine.getVertPt(shortLine.pt2)
        pt1.flag = longLine.isPtAtLineSeg(pt1)
        pt2.flag = longLine.isPtAtLineSeg(pt2)
        if pt1.flag and pt2.flag: return True  # 完全重合的情况
        if not pt1.flag and not pt2.flag: return False  # 完全不重合的请教

        # 计算重合率
        tmpLine = GLine(pt1, pt2)
        pt3 = longLine.pt1 if tmpLine.isPtAtLineSeg(longLine.pt1) else longLine.pt2
        dis1 = pt3.distance(pt1) if pt1.flag else pt3.distance(pt2)
        dis2 = pt1.distance(pt2)
        return dis1 / dis2 > overlapRatio

    ################################################################################
    # 拟合函数
    ################################################################################
    # 通过输入点拟合参数, mode拟合模式, 0:采样最小二乘法拟合,  1:采样随机采样拟合
    def fit(self, pts, mode=0):
        if len(pts) < 2: return None
        if mode == 0:
            N = len(pts)
            sx1, sy1, sx2, sy2, sx1y1 = 0., 0., 0., 0., 0.
            minx, maxx, miny, maxy = pts[0].fx, pts[0].fx, pts[0].fy, pts[0].fy
            for pt in pts:
                if pt.fx > maxx: maxx = pt.fx
                if pt.fx < minx: minx = pt.fx
                if pt.fy > maxy: maxy = pt.fy
                if pt.fy < miny: miny = pt.fy
                x1, y1 = pt.fx, pt.fy
                sx1 += x1
                sy1 += y1
                sx2 += x1 * x1
                sy2 += y1 * y1
                sx1y1 += x1 * y1
            if maxy - miny < maxx - minx:  #
                m = np.array([[sx2, sx1], [sx1, N]], dtype=np.float64)
                n = np.array([-sx1y1, -sy1], dtype=np.float64)
                p1, p2 = np.linalg.solve(m, n).tolist()
                self.a, self.b, self.c = p1, 1., p2
            else:
                m = np.array([[sy2, sy1], [sy1, N]], dtype=np.float64)
                n = np.array([-sx1y1, -sx1], dtype=np.float64)
                p1, p2 = np.linalg.solve(m, n).tolist()
                self.a, self.b, self.c = 1., p1, p2
            self.d = math.sqrt(self.a ** 2 + self.b ** 2)
        elif mode == 1:
            datas = PointsToNdarry(pts)
            x, y = datas[:, 0], datas[:, 1]
            ransac = linear_model.RANSACRegressor()
            if np.std(x) > np.std(y):
                x = x.reshape(-1, 1)
                ransac.fit(x, y)
                self.a, self.b, self.c = float(ransac.estimator_.coef_[0]), -1., float(ransac.estimator_.intercept_)
            else:
                y = y.reshape(-1, 1)
                ransac.fit(y, x)
                self.a, self.b, self.c = -1, float(ransac.estimator_.coef_[0]), float(ransac.estimator_.intercept_)
            self.inPts, self.outPts = [], []
            for inlier, pt in zip(ransac.inlier_mask_, pts):
                if inlier:
                    self.inPts.append(pt)
                else:
                    self.outPts.append(pt)
            pt1, pt2, dis = getPtsMaxDis(self.inPts)
            self.pt1, self.pt2 = self.getVertPt(pt1), self.getVertPt(pt2)
        return self

    # 计算直线的边框点, 返回与边框的交点
    def getBorderPts(self, xmax, ymax, xmin=0, ymin=0):
        pts = []
        y1, y2, x1, x2 = self.getY(xmin), self.getY(xmax), self.getX(ymin), self.getX(ymax)
        if ymin <= y1 <= ymax: pts.append(GPoint(xmin, y1))
        if ymin <= y2 <= ymax: pts.append(GPoint(xmax, y2))
        if xmin <= x1 <= xmax: pts.append(GPoint(x1, ymin))
        if xmin <= x2 <= xmax: pts.append(GPoint(x2, ymax))
        if not pts: return None, None
        datas = [(pt1, pt2, pt1.distance(pt2)) for pt1, pt2 in combinations(pts, 2)]
        datas.sort(key=lambda d: d[2], reverse=True)
        pt1, pt2, dis = datas[0]
        return pt1, pt2

    # 范围截取
    # 将直线的端点控制在矩形范围内
    def clip(self, xmax, ymax, xmin=0, ymin=0):
        pt1, pt2 = self.getBorderPts(xmax, ymax, xmin, ymin)
        if pt1 is None: self.pt1, self.pt2 = None, None; return
        if self.pt1 is None: self.pt1, self.pt2 = pt1, pt2; return

        def getValidPt(pt):
            if xmin <= pt.fx <= xmax and ymin <= pt.fy <= ymax: return pt
            return pt1 if pt.distance(pt1) < pt.distance(pt2) else pt2

        self.pt1 = getValidPt(self.pt1)
        self.pt2 = getValidPt(self.pt2)


# 线段转换位numpy数组, x1, y1, x2, y2
def LinesToNdarry(lines):
    datas = [[line.x1, line.y1, line.x2, line.y2] for line in lines]
    return np.asarray(datas)


# 合并多个线, 最小包围矩
def boundingLine(lines):
    pts = [line.pt1 for line in lines]
    pts.extend([line.pt2 for line in lines])
    return boundingPoint(pts)


# 对直线集合进行密度分组
# lines:线集合,   minDis:允许线断的距离  minAngle:允许直线的角度偏差  mincount:每类的最少数量
# mode: 计算模式 0:采用点到直线的距离来进行距离判断(实现效果是距离近的为聚类在一起),  1:采用点与点的距离来聚类, (效果:直线的一边会聚到一类来)
# 返回结果进行了排序, 返回结果位二维数组
def groupLines(lines, minDis=15., minAngle=8., mode=0, minAmount=2):
    if len(lines) <= 1: return [lines]

    def disLine0(a, b):
        if (a == b).all(): return 0
        linea = GLine().initXY(a[0], a[1], a[2], a[3])
        lineb = GLine().initXY(b[0], b[1], b[2], b[3])
        longline, shortline = (linea, lineb) if linea.getLen() > lineb.getLen() else (lineb, linea)
        pts = [shortline.pt1, shortline.pt2]
        dis = min([longline.distance(pt, mode=1) for pt in pts])
        angle = longline.angle(shortline)
        if angle > minAngle and dis < minDis: dis = minDis  # 当超过角度, 距离至少位mindis(使其不符号条件)
        return dis * (math.sin(math.radians(angle)) + 1)

    def disLine1(a, b):
        if (a == b).all(): return 0
        pt1, pt2, pta, ptb = GPoint(a[0], a[1]), GPoint(a[2], a[3]), GPoint(b[0], b[1]), GPoint(b[2], b[3])
        p1, pa, dis = getMinDisPts([pt1, pt2], [pta, ptb])
        if dis > minDis: return dis
        p2 = pt1 if pt2 is p1 else pt2
        pb = pta if ptb is pa else ptb
        angle = min(180 - abs(getAngle(p1, p2, pb)), 180 - abs(getAngle(pa, p2, pb)))
        return minDis + 3 if angle > minAngle else dis * math.sin(math.radians(angle))

    datas = LinesToNdarry(lines)
    if mode == 0:
        threshold = minDis * (math.sin(math.radians(minAngle)) + 1)
        labels = DBSCAN(eps=threshold, min_samples=minAmount, metric=disLine0).fit(datas).labels_
    elif mode == 1:
        labels = DBSCAN(eps=minDis, min_samples=minAmount, metric=disLine1).fit(datas).labels_
    else:
        return [lines]
    groups = groupData(labels, lines)
    return sorted(groups, key=lambda g: len(g))


# 根据三点求夹角
# 顶点:vertexPt(交点)  edgePt1,edgePt2:边点
# 返回: 角度值 -180到180度(其中符号表示旋转方向, 为正为顺时针方向, 为负为逆时针方向, 坐标远点为左上角)
def getAngle(vertexPt, edgePt1, edgePt2):
    x1, y1, x2, y2, x3, y3 = vertexPt.fx, vertexPt.fy, edgePt1.fx, edgePt1.fy, edgePt2.fx, edgePt2.fy
    x2x1, y2y1, x3x1, y3y1 = x2 - x1, y2 - y1, x3 - x1, y3 - y1
    p1 = float(x2x1 * x3x1 + y2y1 * y3y1)
    p2 = math.sqrt(x2x1 ** 2 + y2y1 ** 2)
    p3 = math.sqrt(x3x1 ** 2 + y3y1 ** 2)
    direct = 1 if x2x1 * y3y1 - y2y1 * x3x1 > 0 else -1
    val = p1 / (p2 * p3)
    if val < -1.: val = -1.
    if val > 1.: val = 1.
    return math.degrees(math.acos(val)) * direct


# 线段是否相交, pt1和pt2为线段1的端点,   pta和ptb为线段2的端点
def isLineSegCross(pt1, pt2, pta, ptb):
    # https://www.cnblogs.com/wuwangchuxin0924/p/6218494.html
    sxmin, sxmax = (pt1.fx, pt2.fx) if pt1.fx <= pt2.fx else (pt2.fx, pt1.fx)
    symin, symax = (pt1.fy, pt2.fy) if pt1.fy <= pt2.fy else (pt2.fy, pt1.fy)
    lxmin, lxmax = (pta.fx, ptb.fx) if pta.fx <= ptb.fx else (ptb.fx, pta.fx)
    lymin, lymax = (pta.fy, ptb.fy) if pta.fy <= ptb.fy else (ptb.fy, pta.fy)
    if sxmin > lxmax or lymin > symax or lxmin > sxmax or symin > lymax: return False  # 矩形不相交

    u = (pta.fx - pt1.fx) * (pt2.fy - pt1.fy) - (pt2.fx - pt1.fx) * (pta.fy - pt1.fy)
    v = (ptb.fx - pt1.fx) * (pt2.fy - pt1.fy) - (pt2.fx - pt1.fx) * (ptb.fy - pt1.fy)
    w = (pt1.fx - pta.fx) * (ptb.fy - pta.fy) - (ptb.fx - pta.fx) * (pt1.fy - pta.fy)
    z = (pt2.fx - pta.fx) * (ptb.fy - pta.fy) - (ptb.fx - pta.fx) * (pt2.fy - pta.fy)
    return u * v <= 1e-7 and w * z <= 1e-7


# 直线检测  mode:检测模式,0:霍夫直线检测,  1:直线检测器检测,    img:待检测的图像,模式0时为二值图, 模式1为灰度图
# threshold:霍夫投票阈值  minLineLength:最小直线长度  maxLineGap:最大直线断开距离
def lineDetect(img, mode=0, threshold=-1, minLineLength=0, maxLineGap=0):
    img = cv2.cvtColor(img, code=cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    lines = []
    if mode == 0:  # 霍夫直线检测
        # 参数 rho:累加器距离分辨率,  theta:累加器角度分辨率   threshold:投票阈值,  minLineLength:直线的最小长度  maxLineGap:直线合并最大断开距离
        datas = cv2.HoughLinesP(img, rho=1, theta=np.pi / 360., threshold=int(threshold), minLineLength=minLineLength,
                                maxLineGap=maxLineGap)
        if datas is None:
            return lines
        for d in datas.reshape(-1, 4):
            line = GLine().initXY(*d)
            if line.getLen() <= 0: continue
            lines.append(line)
    elif mode == 1:  # 直线检测器检测
        # 直线检测器检测 cv2.createLineSegmentDetector
        # 参数1 refine(LSD_REFINE_STD,): 调优策略, cv2.LSD_REFINE_NONE:无调优策略, cv2.LSD_REFINE_STD:标准调优策略,将弧分解成多个小线段, cv2.LSD_REFINE_ADV:使用NFA指标评价检测直线,调整参数近一步调优检测直线
        # 参数2 scale(0.8): 缩放图像比例   参数3 sigma_scale(0.6): 高斯模糊核sigma=sigma_scale/scale   参数4 quant(2.0):梯度量化误差
        # 参数5 ang_th(22.5):直线段角度容差  参数6 log_eps(0): NFA检测阈值  参数7 density_th(0.7):直线段密度阈值   参数8 n_bins(1024):梯度归一化数量
        lsd = cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD)
        datas, width, prec, nfa = lsd.detect(img)
        if datas is None and width is None and prec is None and nfa is None: print("检测失败，未能检测到直线");return lines
        datas = datas.reshape(-1, 4)
        width = width.reshape(-1).tolist()
        prec = prec.reshape(-1).tolist()
        for i in range(len(datas)):
            line = GLine().initXY(*datas[i])
            if line.getLen() <= 0: continue
            line.width, line.prec = width[i], prec[i]
            lines.append(line)
    else:
        pass
    return lines


# 线段融合算法, 满足直线距离阈值和线段距离阈值, 则两线段合并
# 参数lines:为GLine数组, maxRatio:直线间隔最大阈值比例, minDis:直线间隔最小值, segRatio:线段间隔阈值
# 返回值为融合后的线段数组(GLine),
#
#
def lineSegmentFuse(lines, maxRatio=0.01, minDis=1.414, segRatio=0.1):
    def fuse(baseline, line):
        tlen = baseline.getLen() + line.getLen()
        lineTh = max(tlen * maxRatio, minDis)  # 直线间隔阈值
        segTh = tlen * segRatio  # 线段间隔阈值

        pts = [line.pt1, line.pt2]
        lineDis = [baseline.distance(p, mode=0) for p in pts]  # 直线距离取最大
        segDis = [baseline.distance(p, mode=1) for p in pts]  # 线段距离取最小

        # 直线距离超过阈值, 或者断开距离超过阈值, 不能融合
        if max(lineDis) > lineTh or min(segDis) > segTh: return False
        pt1, pt2, dis = getMaxLinesLen([baseline, line])
        if abs(dis - baseline.getLen()) < 0.01: return True

        pts = baseline.getLinePts(True)
        pts.extend(line.getLinePts(True))

        resline = GLine().fit(pts)
        pt1 = resline.getVertPt(pt1)
        pt2 = resline.getVertPt(pt2)
        baseline.initPts(pt1, pt2)
        return True

    results = []
    datas = lines
    while len(datas) > 0:
        datas = sorted(datas, key=lambda d: d.getLen(), reverse=True)
        baseline = datas.pop(0)  # 弹出长度最长的直线
        results.append(baseline)

        # 尝试融合的优先级为, 平行 > 距离 > 长度
        datas.sort(key=lambda d: min(baseline.distance(d.pt1, 1), baseline.distance(d.pt2, 1))
                                 + abs(baseline.distance(d.pt1) - baseline.distance(d.pt2)) * 100)
        while True:  # 循环融合,直到无融合直线
            index = []
            for i, line in enumerate(datas):
                if not fuse(baseline, line): continue
                index.append(i)
            if not index: break  # 没有合并线时则跳出循环
            index.sort(reverse=True)
            for i in index: datas.pop(i)

    results.sort(key=lambda d: d.getLen(), reverse=True)
    return results


# 根据平行线进行分组
# 参数line:待检测的直线, pddr:平行距离差的阈值比例(小于1), 固定值(大于1), overlapRatio:重叠阈值比例, 平行线距离(lineMindis, lineMaxdis)
# 返回值为直线二维分组
def groupParallelLines(lines, pddr=0.05, overlapRatio=0.3, lineMindis=-1, lineMaxdis=-1):
    group = []
    datas = sorted(lines, key=lambda d: d.getLen(), reverse=True)
    while len(datas) > 0:
        baseline = datas.pop(0)
        result = [baseline]
        group.append(result)
        index = []
        for i, line in enumerate(datas):
            if not baseline.isParallel(line, pddr, overlapRatio, lineMindis, lineMaxdis): continue
            index.append(i)
            result.append(line)
        index.sort(reverse=True)
        for i in index: datas.pop(i)
    return group


# 填充平行线, lines为GLine数组, img为灰度图, fillval为填充值, thval:距离阈值<1为比例值, >1为固定值
def fillParallelLines(lines, img, thval=0.2, fillval=1):
    from geometry.point import PointsToNdarry
    if lines is None or len(lines) < 2: return
    for line1, line2 in combinations(lines, 2):
        th = max(line1.getLen(), line2.getLen()) * thval if thval < 1 else thval
        if line1.distanceWithLine(line2) > th: continue
        dis1 = line1.pt1.distance(line2.pt1) + line1.pt2.distance(line2.pt2)
        dis2 = line1.pt1.distance(line2.pt2) + line1.pt2.distance(line2.pt1)
        if dis1 < dis2:
            pts = [line1.pt1, line1.pt2, line2.pt2, line2.pt1]
        else:
            pts = [line1.pt1, line1.pt2, line2.pt1, line2.pt2]
        pts = PointsToNdarry(pts).reshape((1, -1, 2))
        cv2.fillPoly(img, pts, fillval)


def getMaxLinesLen(lines):
    pts = []
    for line in lines: pts.extend([line.pt1, line.pt2])
    return getPtsMaxDis(pts)


def calculate_theta(li):
    """
    计算theta弧度和rho
    :param li:
    :return: 直线对象
    """
    #
    #  两个向量之间的旋转角
    #  首先明确几个数学概念：
    #  1. 极轴沿逆时针转动的方向是正方向
    #  2. 两个向量之间的夹角theta， 是指(A^B)/(|A|*|B|) = cos(theta)，0<=theta<=180 度， 而且没有方向之分
    #  3. 两个向量的旋转角，是指从向量p1开始，逆时针旋转，转到向量p2时，所转过的角度， 范围是 0 ~ 360度
    #  计算向量p1到p2的旋转角，算法如下：
    #  首先通过点乘和arccosine的得到两个向量之间的夹角
    #  然后判断通过差乘来判断两个向量之间的位置关系   cross = x1 * y2 - x2 * y1
    #  如果p2在p1的顺时针方向, 返回arccose的角度值, 范围0 ~ 180.0(根据右手定理,可以构成正的面积)
    #  否则返回 360.0 - arecose的值, 返回180到360(根据右手定理,面积为负)
    #  这里的 p1 = (1.0, 1)  ,  p2 = (-li.c * li.a, -li.c * li.b)
    #

    if -0.1 < li.c < 0.1:  # 保证 li.c 不为0
        li.c += 1.0
    # n_vector = (-li.c * li.a, -li.c * li.b)  # 法向量和直线的交点
    # p0 = (1.0, 0)  # 基本向量
    # dot_product = -li.c * li.a * 1.0 + -li.c * li.b * 0
    # dot_product = -li.c * li.a
    theta = math.acos(-li.c * li.a / math.sqrt((li.c * li.a) ** 2 + (li.c * li.b) ** 2))
    # 计算向量积 cross product, 的值(cross product 是一个向量)
    # cross = 1.0 * (-li.c * li.b) - (-li.c * li.a) * 0.0
    cross_product = -li.c * li.b
    if cross_product < 0:
        theta = 2 * math.pi - theta

    li.theta = theta
    # li.theta = math.acos(abs(li.b / li.d))  # 方向向量指向直线
    li.rho = abs(li.c / li.d)


def calculate_theta_rho(li):
    """
    计算 法向量的 theta 和 rho
    :param li:
    :return: 直线对象
    """
    #
    #  两个向量之间的旋转角
    #  首先明确几个数学概念：
    #  1. 极轴沿逆时针转动的方向是正方向
    #  2. 两个向量之间的夹角theta， 是指(A^B)/(|A|*|B|) = cos(theta)，0<=theta<=180 度， 而且没有方向之分
    #  3. 两个向量的旋转角，是指从向量p1开始，逆时针旋转，转到向量p2时，所转过的角度， 范围是 0 ~ 360度
    #  计算向量p1到p2的旋转角，算法如下：
    #  首先通过点乘和arccosine的得到两个向量之间的夹角
    #  然后判断通过差乘来判断两个向量之间的位置关系   cross = x1 * y2 - x2 * y1
    #  如果p2在p1的顺时针方向, 返回arccose的角度值, 范围0 ~ 180.0(根据右手定理,可以构成正的面积)
    #  否则返回 360.0 - arecose的值, 返回180到360(根据右手定理,面积为负)
    #  这里的 p1 = (1.0, 1)  ,  p2 = (-li.c * li.a, -li.c * li.b)
    #

    if -0.1 < li.c < 0.1:  # 保证 li.c 不为0
        li.c += 1.0
    # n_vector = (-li.c * li.a, -li.c * li.b)  # 法向量和直线的交点
    # p0 = (1.0, 0)  # 基本向量
    # dot_product = -li.c * li.a * 1.0 + -li.c * li.b * 0
    # dot_product = -li.c * li.a
    theta = math.acos(-li.c * li.a / math.sqrt((li.c * li.a) ** 2 + (li.c * li.b) ** 2))
    # 计算向量积 cross product, 的值(cross product 是一个向量)
    # cross = 1.0 * (-li.c * li.b) - (-li.c * li.a) * 0.0
    cross_product = -li.c * li.b
    if cross_product < 0:
        theta = 2 * math.pi - theta

    li.theta = theta
    # li.theta = math.acos(abs(li.b / li.d))  # 方向向量指向直线
    li.rho = abs(li.c / li.d)


def auto_canny(image):
    """
    计算一个合适的canny阈值
    :param image:
    :return:
    """
    i = 1
    std = np.std(image)
    canny_ls = []  # canny_im 线条的像素个数
    # canny_contour_num = []
    while i < 100:
        canny_im = cv2.Canny(image, i, i * 2, apertureSize=3)
        canny_sum = np.sum(canny_im == 255)
        canny_ls.append(canny_sum)

        # _, contours, _ = cv2.findContours(canny_im, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # canny_contour_num.append(len(contours))
        i += 1
    grad_list = np.gradient(canny_ls, range(len(canny_ls)), edge_order=1)
    # min_exts = argrelmin(grad_list, order=2)[0]
    man_exts = argrelmax(grad_list, order=2)[0]
    # print('min_exts: {}, man_exts: {}'.format(min_exts, man_exts))
    if not man_exts.any():
        return 11
    pre_ext = man_exts[0]

    # # 使用轮廓数量
    # grad_contour_num = np.gradient(canny_contour_num, range(len(canny_contour_num)), edge_order=1)
    # min_cnt_exts = argrelmin(grad_contour_num, order=2)[0]
    # man_cnt_exts = argrelmax(grad_contour_num, order=2)[0]
    # print('min_cnt_exts: {}'.format(min_cnt_exts))
    # print('man_cnt_exts: {}'.format(man_cnt_exts))

    for ext in man_exts:
        if ext > std:
            break
        else:
            pre_ext = ext
    # print('pre_ext: {}'.format(pre_ext))

    # min_ext_X = []
    # min_ext_Y = []
    # for ext in min_cnt_exts:
    #     min_ext_X.append(ext)
    #     t = grad_contour_num[ext]
    #     min_ext_Y.append(t)
    # max_ext_X = []
    # max_ext_Y = []
    # for ext in man_cnt_exts:
    #     max_ext_X.append(ext)
    #     t = grad_contour_num[ext]
    #     max_ext_Y.append(t)
    # plt.plot(min_ext_X, min_ext_Y, 'ro')
    # plt.plot(max_ext_X, max_ext_Y, 'bo')
    # plt.plot(grad_contour_num)
    # plt.show()
    return pre_ext


def cluster_by_theta(lines):
    """
    cluster_by_theta
    :param lines: line vectors
    :return: grouped line by theta
    """
    if lines is None:
        return []
    theta_differ = 0.0348 * 2

    # 按角度排序
    lines = sorted(lines, key=lambda li: li.theta)

    theta_ls = [line.theta for line in lines]  # get theta list
    # theta_std = np.std(theta_ls)  # thetas std

    # theta 的diff
    theta_diff = np.diff(theta_ls)  # calculate diff in theta_ls which have been sorted
    diff_idx = np.where(theta_diff > theta_differ)[0] + 1  # slice by std value
    diff_idx = np.append(diff_idx, [len(lines)])  # line_ls 长度作为最后一个
    theta_groups = []  # 按 theta 分组, 没组的lines theta 都聚在一起
    group_theta_ls = []
    for i, idx in enumerate(diff_idx):  # 邻近的diff theta 分组
        if i == 0:
            theta_groups.append(lines[0: idx])
            group_theta_ls.append(theta_ls[0: idx])
        else:
            theta_groups.append(lines[diff_idx[i - 1]:idx])
            group_theta_ls.append(theta_ls[diff_idx[i - 1]:idx])
    theta_groups = [gp for gp in theta_groups if len(gp) > 0]
    return theta_groups


def cluster_by_rho(theta_groups):
    result_group = []  # 合并后的分组线
    rho_differ = 7
    for gp in theta_groups:
        # 按 rho 的大小排序的 line 列表
        rho_lines = sorted([li for li in gp], key=lambda li: li.rho)
        rho_ls = [li.rho for li in rho_lines]
        rho_diff = np.diff(rho_ls)  # calculate diff in theta_ls which have been sorted
        rho_diff_idx = np.where(rho_diff > rho_differ)[0] + 1  # slice by std value
        rho_diff_idx = np.append(rho_diff_idx, [len(rho_lines)])  # line_ls 长度作为最后一个
        rho_groups = []  # 按 rho 分组
        gp_rho_ls = []
        for i, idx in enumerate(rho_diff_idx):  # 邻近的diff theta 分组
            if i == 0:
                rho_groups.append(rho_lines[0: idx])
                gp_rho_ls.append(rho_ls[0: idx])
            else:
                rho_groups.append(rho_lines[rho_diff_idx[i - 1]:idx])
                gp_rho_ls.append(rho_ls[rho_diff_idx[i - 1]:idx])
        # 在分组内(rho 和 theta 都接近的情况下)
        new_lines = []
        for lis in rho_groups:
            if len(lis) > 1:  # 合并在同一直线上的直线段
                pts = []
                for li in lis:
                    pts.extend(li.getLinePts(includeEndPt=True))
                new_li = GLine().fit(pts)
                calculate_theta_rho(new_li)
                new_lines.append(new_li)
            elif len(lis) == 1:
                new_lines.append(lis[0])
        result_group.append(new_lines)
    return result_group


if __name__ == "__main__":
    #
    g = [1, 2, 3, 4, 5, 6]


    def test(d):
        while d:
            print d.pop()


    test(g)

    t1 = 1
