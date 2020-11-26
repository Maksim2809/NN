import math
import random

import numpy as np


class loader:
    def __init__(self,
                 dimentions = 2,
                 trainPercent = 85.0):
        self.__tp = trainPercent
        self.__tr, self.__ts = self.__loadData(dimentions)


    def __loadData(self, dim):
        data = self.__get2DData() if dim == 2 else self.__get3DData()
        ln = len(data)
        lnts = int(ln * (1 - self.__tp / 100))
        lntr = ln - lnts

        random.shuffle(data)
        return sorted(data[:lntr]), sorted(data[lntr:])


    def __get2DData(self):
        return [
            [
                [i / 10],
                [math.cos(i / 10) + random.random() * 0.2 - 0.1]
            ]
            for i in range(-60, 61)
        ]


    def __get3DData(self):
        return [
            [
               [i/10, i/10],
               [math.cos(i/10+i/10) + random.random()*0.2]
            ] 
            for i in range(-60,61)
            ]


    def getTrainInp(self):
        return np.array([i[0] for i in self.__tr])


    def getTrainOut(self):
        return np.array([i[1] for i in self.__tr])


    def getTestInp(self):
        return np.array([i[0] for i in self.__ts])

    def getTestOut(self):
        return np.array([i[1] for i in self.__ts])

#--------------------------------------------------------------


    def get2TrainInp(self):
        return np.array([i[0] for i in self.__tr])


    def get2TrainOut(self):
        return np.array([i[1] for i in self.__tr])


    def get2TestInp(self):
        return np.array([i[0] for i in self.__ts])

    def get2TestOut(self):
        return np.array([i[1] for i in self.__ts])
    
