from collections import *
from igraph import *
import DataReader

class Centrality(object):

    """ In this class we will define Betweenness,
        Eigen & Closeness centrality. """

    def __init__(self, g, pTr, pVal, pTest, pTr_neg, pVal_neg, pTest_neg):
        self.arg = DataReader.DataReader(g, pTr, pVal, pTest, pTr_neg, pVal_neg, pTest_neg)

    def BetweenCentrality(self):
        bfs = self.arg.graph.betweenness(self.arg.graph.vs) # BFS taking vertex as parameter.
        for i in range(1, 6):
            _maxValue = max(bfs)
            bfs.remove(_maxValue)
        bfs = self.arg.graph.betweenness(self.arg.graph.vs)
        return bfs

    def ClosenessCentrality(self):
        close = self.arg.graph.closeness(self.arg.graph.vs) # Closeness between nodes.
        for i in range(1, 6):
            _maxValue = max(close)
            close.remove(_maxValue)
        close = self.arg.graph.closeness(self.arg.graph.vs) # calculate new Closeness.
        return close

    def EigenVector_Centrality(self):
        eigen = self.arg.graph.evcent(directed= False) # Our graph is UNDIRECTED.
        for i in range(1,6):
            _maxValue = max(eigen)
            eigen.remove(_maxValue)
        eigen = self.arg.graph.evcent(directed = False)
        return eigen

    def DegreeCentrality(self):
        degree = self.arg.graph.degree()
        for i in range(1, 6):
            _maxValue = max(degree)
            degree.remove(_maxValue)
        degree = self.arg.graph.degree()
        return degree

if __name__ == '__main__':
    # test des methodes.
    o = Centrality(Graph(),
                    "/home/lbenboudiaf/Bureau/FacebookNetwork/dataset/SplitedData/train.edges",
                    "/home/lbenboudiaf/Bureau/FacebookNetwork/dataset/SplitedData/val.edges",
                    "/home/lbenboudiaf/Bureau/FacebookNetwork/dataset/SplitedData/test.edges",
                    "/home/lbenboudiaf/Bureau/FacebookNetwork/dataset/SplitedData/neg_train.edges",
                    "/home/lbenboudiaf/Bureau/FacebookNetwork/dataset/SplitedData/neg_val.edges",
                    "/home/lbenboudiaf/Bureau/FacebookNetwork/dataset/SplitedData/neg_test.edges")
    o.arg.LoadEdgesData() #Initialiser la données.
    o.BetweenCentrality()
    o.ClosenessCentrality()
    o.DegreeCentrality()
    #print(eigenVec)
