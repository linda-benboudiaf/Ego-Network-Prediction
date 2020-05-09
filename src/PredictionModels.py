from collections import *
from igraph import *
import DataReader

class PredictionModels(object):
    """docstring for PredictionModels."""

    #Loading data
    def __init__(self, g, pTr, pVal, pTest):
        self.arg = DataReader.DataReader(g, pTr, pVal, pTest)



    if __name__== '__main__':
