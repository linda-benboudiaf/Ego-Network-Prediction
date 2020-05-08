from igraph import *
from collections import *
import random

class DataReader(object):

    """ Data reading and spliting to Train, Validation and Test """

    def __init__(self, graph, fluxTr,fluxV, fluxTest):
        self.graph = graph
        self.fluxV = fluxV
        self.fluxTr = fluxTr
        self.fluxTest = fluxTest

    def addVertex(self, _nodeName):
        try:
            if(_nodeName not in self.graph.vs['name']):
                #print('Inserting vertex to self.graph',name_str)
                self.graph.add_vertex(name = _nodeName)
            #else:
            #    print('Vertex exists already at index: ',self.graph.vs.find(_nodeName).index)
        except KeyError:
            self.graph.add_vertex(name =_nodeName)
        return self.graph


    def LoadData(self):
        #fileIndex=[0, 107, 348, 414, 686, 698, 1684, 1912, 3437, 3980] #All Egos id for edges.
        fileIndex = [3980]
        for i, egoId in enumerate(fileIndex):
            path = "/home/lbenboudiaf/Bureau/FacebookNetwork/dataset/edges/"+str(egoId)+".edges"
            flux = open(path, 'r')
            line=flux.readline()
            while(line!=''):
                res = (line.split()) # Will have an array of strings.
                self.addVertex(res[1]) # Adding nodes to the self.graph.
                self.addVertex(res[0])
                self.graph.add_edge(res[0],res[1])
                line=flux.readline() #line ++ read next line.
        self.graph.simplify()
        return

    def WriteTupleToFile(self, Ftrain,t):
        string=str(t[0])+' '+str(t[1])+'\n'
        Ftrain.write(string)

    def getEdgeName(self,t):
        a = (self.graph.vs[t[0]]['name'],self.graph.vs[t[1]]['name'])
        return a

    # Generate Train File 80%
    def GenerateTrain(self):
        fluxTrain = open(self.fluxTr, 'a+')
        for i in range(int(len(self.graph.es) * 0.5)):
            # self.graph.es number of edges.
            ran = random.randint(0, len(self.graph.es)-1) #Generate random number endpoints included.
            tup = self.graph.es[ran].tuple
            self.graph.delete_edges(tup)
            self.WriteTupleToFile(fluxTrain, self.getEdgeName(tup))
        fluxTrain.close()

    # Generate test file 20%
    def GenerateTest(self):
        fluxTest = open(self.fluxTest, 'a+')
        for i in range (int(len(self.graph.es)*0.25)):
            ran = random.randint(0, len(self.graph.es)-1)
            tup = self.graph.es[ran].tuple
            self.graph.delete_edges(tup)
            self.WriteTupleToFile(fluxTest, self.getEdgeName(tup))
        fluxTest.close()

   # Generate Validation file 20%
    def GenerateValidation(self):
        fluxVal = open(self.fluxV, 'a+')
        for i in range(int(len(self.graph.es)*0.15)):
            ran = random.randint(0,len(self.graph.es)-1)
            tup = self.graph.es[ran].tuple
            self.graph.delete_edges(tup)
            self.WriteTupleToFile(fluxVal, self.getEdgeName(tup))
        fluxVal.close()


if __name__ == '__main__':
    print(o.__doc__)
    o = DataReader(Graph(),
                    "/home/lbenboudiaf/Bureau/FacebookNetwork/dataset/SplitedData/train.edges",
                    "/home/lbenboudiaf/Bureau/FacebookNetwork/dataset/SplitedData/val.edges",
                    "/home/lbenboudiaf/Bureau/FacebookNetwork/dataset/SplitedData/test.edges")
    o.LoadData()
    o.GenerateTrain()
    o.GenerateTest()
    o.GenerateValidation()
