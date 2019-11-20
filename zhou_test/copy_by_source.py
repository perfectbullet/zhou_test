#!/usr/bin/python
# coding=utf-8


def move_sourcepic():
    """
    移动原图
    :return:
    """
    import os
    from shutil import copyfile, move
    from os.path import join

    src_dir = '/disk_workspace/test_images/达州至凉雾-双线201806鱼背山-罗田区间/pickledata'
    dst_dir = '/disk_workspace/test_images/达州至凉雾-双线201806鱼背山-罗田区间/棒瓶和角钢构件'

    for root, dirs, files in os.walk(src_dir):
        print('root: {}'.format(root))
        for name in files:
            if name.endswith('.jpg'):
                src = join(src_dir, name)
                dst = join(dst_dir, name)
                move(src, dst)


if __name__ == '__main__':

    import os
    from os.path import join


    def get_sourcepic():
        """
        复制原图
        :return:
        """
        import os
        from shutil import copyfile, move
        from os.path import join
        from os import listdir

        dir1 = '/disk_workspace/test_images/达州至凉雾-双线201806鱼背山-罗田区间/pickledata'
        dir2 = '/disk_workspace/test_images/达州至凉雾-双线201806鱼背山-罗田区间/棒瓶和角钢构件'

        for root, dirs, files in os.walk(dir2):
            print('root: {}'.format(root))
            for name in files:
                if '_' in name:
                    src_name = name.split('_')[0] + '.jpg'
                else:
                    src_name = name
                src = join(dir1, src_name)
                dst = join(dir2, name)
                # copyfile(src, dst)
                move(src, dst)

    move_sourcepic()
