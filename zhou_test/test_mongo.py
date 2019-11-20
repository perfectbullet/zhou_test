#!/usr/bin/python
# coding=utf-8

from pymongo import MongoClient


class TDocDB:
    def __init__(self):
        self.mongoclient = MongoClient('127.0.0.1', 27017, connect=False)
        self.db = self.mongoclient['mhsb_gt']
        self.mongostate = self.db.authenticate('zz', '123456')
        print(self.mongostate)


if __name__ == '__main__':
    mgdb = TDocDB()
    pass