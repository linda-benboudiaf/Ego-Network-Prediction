from igraph import *
from collections import *
import random
import numpy as np

class DataReader(object):

    """ Data reading and spliting to Train, Validation and Test """

    def __init__(self,graph, fluxTr,fluxV, fluxTest, fluxTr_neg, fluxVal_neg, fluxTest_neg):
        self.graph = graph
        self.fluxV = fluxV
        self.fluxTr = fluxTr
        self.fluxTest = fluxTest
        self.fluxTr_neg = fluxTr_neg
        self.fluxVal_neg = fluxVal_neg
        self.fluxTest_neg = fluxTest_neg

    def addVertex(self, _nodeName):
        try:
            if(_nodeName not in self.graph.vs['name']):
                self.graph.add_vertex(name = _nodeName)
        except KeyError:
            self.graph.add_vertex(name =_nodeName)
        return self.graph

    def LoadEdgesData(self):
        #fileIndex=[0, 107, 348, 414, 686, 698, 1684, 1912, 3437, 3980] #All Egos id for edges.
        fileIndex = [0]
        for egoId in fileIndex:
            path = "/home/lbenboudiaf/Bureau/FacebookNetwork/dataset/edges/"+str(egoId)+".edges"
            flux = open(path)
            line=flux.readline()
            while(line!=''):
                res = (line.split()) # Will have an array of strings.
                self.addVertex(res[1]) # Adding nodes to the self.graph.
                self.addVertex(res[0])
                self.graph.add_edge(res[0],res[1])
                line=flux.readline() #line ++ read next line.
        self.graph.simplify()
        return

    def LoadGeneratedFiles(self, path):
        flux = open(path)
        line=flux.readline()
        while(line!=''):
            char = (line.split())
            self.addVertex(char[0])
            self.addVertex(char[1])
            self.graph.add_edge(char[0],char[1])
            line=flux.readline()
        self.graph.simplify()
        return

    # Load features on dictionnary
    def LoadFeaturesData(self):
        #fileIndex = [0, 107, 348, 414, 686, 698, 1684, 1912, 3437, 3980]
        fileIndex = [0]
        feats = dict()
        for i in fileIndex:
            path = '/home/lbenboudiaf/Bureau/FacebookNetwork/dataset/feat/'+str(i)+'.feat'
            flux = np.loadtxt(path)
            for f in flux :
                feats[f[0]] = np.asarray(f[1:])
        return feats

    def WriteTupleToFile(self, Ftrain,t):
        string=str(t[0])+' '+str(t[1])+'\n'
        Ftrain.write(string)

    def getEdgeName(self,t):
        edgeName = (self.graph.vs[t[0]]['name'],self.graph.vs[t[1]]['name'])
        return edgeName

    # We generate negative exemples according to graph adjacency
    def GenerateNegative(self):
        m = self.graph.get_adjacency()
        pool = list()
        count = 0
        for i, node in enumerate(m):
            for j, value in enumerate(node):
                if(value == 0 and i!=j):
                    count+=1;
                    pool.append((i,j))
        print('Count of negative sample: ', count)
        for e in pool:
            if(e[0] == 0):
                pool.remove(e)
        return pool

    # Generate Train File 60%
    def GenerateTrain(self):
        fluxTrain = open(self.fluxTr, 'a+')
        fluxTrain_neg = open(self.fluxTr_neg, 'a+')
        pool = self.GenerateNegative()
        for i in range(int(len(self.graph.es) * 0.5)):
            # self.graph.es number of edges.
            ran = random.randint(0, len(self.graph.es)-1) #Generate positive random number endpoints included.
            ran_neg = random.randint(0, len(pool)-1)
            tup = self.graph.es[ran].tuple
            tup_neg = pool[ran_neg]
            pool.remove(tup_neg)
            self.graph.delete_edges(tup)
            fluxTrain_neg.write(str(tup_neg[0])+' '+str(tup_neg[1])+'\n')
            self.WriteTupleToFile(fluxTrain, self.getEdgeName(tup))
        fluxTrain.close()
        fluxTrain_neg.close()

    # Generate test file 20%
    def GenerateTest(self):
        fluxTest = open(self.fluxTest, 'a+')
        fluxTest_neg = open(self.fluxTest_neg, 'a+')
        pool = self.GenerateNegative()
        for i in range (int(len(self.graph.es)*0.25)):
            ran = random.randint(0, len(self.graph.es)-1)
            ran_neg = random.randint(0, len(pool)-1)
            tup = self.graph.es[ran].tuple
            tup_neg = pool[ran_neg]
            pool.remove(tup_neg)
            self.graph.delete_edges(tup)
            fluxTest_neg.write(str(tup_neg[0])+' '+str(tup_neg[1])+'\n')
            self.WriteTupleToFile(fluxTest, self.getEdgeName(tup))
        fluxTest.close()
        fluxTest_neg.close()

   # Generate Validation file 20%
    def GenerateValidation(self):
        fluxVal = open(self.fluxV, 'a+')
        fluxVal_neg = open(self.fluxVal_neg, 'a+')
        pool = self.GenerateNegative()
        for i in range(int(len(self.graph.es)*0.15)):
            ran = random.randint(0,len(self.graph.es)-1)
            ran_neg = random.randint(0, len(pool)-1)
            tup = self.graph.es[ran].tuple
            tup_neg = pool[ran_neg]
            self.graph.delete_edges(tup)
            pool.remove(tup_neg)
            self.WriteTupleToFile(fluxVal, self.getEdgeName(tup))
            fluxVal_neg.write(str(tup_neg[0])+' '+str(tup_neg[1])+'\n')
        fluxVal.close()
        fluxVal_neg.close()

if __name__ == '__main__':
    o = DataReader(Graph(),
                    "/home/lbenboudiaf/Bureau/FacebookNetwork/dataset/SplitedData/train.edges",
                    "/home/lbenboudiaf/Bureau/FacebookNetwork/dataset/SplitedData/val.edges",
                    "/home/lbenboudiaf/Bureau/FacebookNetwork/dataset/SplitedData/test.edges",
                    "/home/lbenboudiaf/Bureau/FacebookNetwork/dataset/SplitedData/neg_train.edges",
                    "/home/lbenboudiaf/Bureau/FacebookNetwork/dataset/SplitedData/neg_val.edges",
                    "/home/lbenboudiaf/Bureau/FacebookNetwork/dataset/SplitedData/neg_test.edges")
    print(o.__doc__)
    o.LoadFeaturesData()
    o.LoadGeneratedFiles(o.fluxTr_neg)
    o.LoadEdgesData()
    o.GenerateTrain()
    o.GenerateTest()
    o.GenerateNegative()
