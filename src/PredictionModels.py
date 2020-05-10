from collections import *
from igraph import *
from sklearn import linear_model, svm
from sklearn.metrics import log_loss
import DataReader

class PredictionModels(object):
    """docstring for PredictionModels."""
    #Loading data
    def __init__(self, g, pTr, pVal, pTest, pTr_neg, pVal_neg, pTest_neg):
        #Validation data
        self.posV = DataReader.DataReader(g, pTr, pVal, pTest, pTr_neg, pVal_neg, pTest_neg)
        self.negV = DataReader.DataReader(g, pTr, pVal, pTest, pTr_neg, pVal_neg, pTest_neg)

        #Train Data
        self.pos = DataReader.DataReader(g, pTr, pVal, pTest, pTr_neg, pVal_neg, pTest_neg)
        self.neg = DataReader.DataReader(g, pTr, pVal, pTest, pTr_neg, pVal_neg, pTest_neg)


    def splitPositiveData(self):
        X_train_positive = self.pos.LabelData()
        Y_train_positive = self.neg.LabelData()

    if __name__== '__main__':
