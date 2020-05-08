from collections import *
from igraph import *
import DataReader

class Centrality(object):

    """ In this class we will define Betweenness,
        Eigen & Closeness centrality. """
    def __init__(self, g, pTr, pVal, pTest):
        self.arg = DataReader.DataReader(g, pTr, pVal, pTest)

    def BetweenCentrality(self):
        bfs = self.arg.graph.betweenness(self.arg.graph.vs) # BFS taking vertex as parameter.
        for i in range(1, 6):
            _maxValue = max(bfs)
            #print(i,'==> node',self.arg.graph.vs[bfs.index(_maxValue)]['name'],' [Betweenness] with score of ',_maxValue)
            bfs.remove(_maxValue)
        bfs = self.arg.graph.betweenness(self.arg.graph.vs)
        return bfs

    def ClosenessCentrality(self):
        close = self.arg.graph.closeness(self.arg.graph.vs) # Closeness between nodes.
        for i in range(1, 6):
            _maxValue = max(close)
            #print(i,'==> node',self.arg.graph.vs[close.index(_maxValue)]['name'],' [Closeness] with score of ',_maxValue)
            close.remove(_maxValue)
        close = self.arg.graph.closeness(self.arg.graph.vs) # calculate new Closeness.
        return close

    def EigenVector_Centrality(self):
        eigen = self.arg.graph.evcent(directed= False) # Our graph is UNDIRECTED.
        for i in range(1,6):
            _maxValue = max(eigen)
            #print(i,'==> node',self.arg.graph.vs[eigen.index(_maxValue)]['name'],' [Eigen] with score of ',_maxValue)
            eigen.remove(_maxValue)
        eigen = self.arg.graph.evcent(directed = False)
        return eigen


if __name__ == '__main__':
    o = Centrality(Graph(),
                    "/home/lbenboudiaf/Bureau/FacebookNetwork/dataset/SplitedData/train.edges",
                    "/home/lbenboudiaf/Bureau/FacebookNetwork/dataset/SplitedData/val.edges",
                    "/home/lbenboudiaf/Bureau/FacebookNetwork/dataset/SplitedData/test.edges")
    o.arg.LoadData() #Initialiser la données.
    o.BetweenCentrality()
    o.ClosenessCentrality()
    eigenVec = o.EigenVector_Centrality()
    #print(eigenVec)
