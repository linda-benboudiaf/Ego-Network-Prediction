# Load train dataset
from igraph import *
from collections import *

class DataReader(object):
    """ Data reading and spliting """

    def __init__(self, graph):
        self.graph = graph

    def addVertex(self, _nodeName):
        try:
            if(_nodeName not in self.graph.vs['name']):
                #print('Inserting vertex to self.graph',name_str)
                self.graph.add_vertex(name = _nodeName)
            else:
                print('Vertex exists already at index: ',self.graph.vs.find(_nodeName).index)
        except KeyError:
            self.graph.add_vertex(name =_nodeName)
        return self.graph


    def LoadData(self):
        fileIndex=[0]
        for i, egoId in enumerate(fileIndex):
            print(egoId)
            path = "/home/lbenboudiaf/Bureau/FacebookNetwork/dataset/edges/"+str(egoId)+".edges"
            print('File path =', path)
            flux = open(path,'r')
            nodeID=egoId
            line=flux.readline()
            while(line!=''):
                res = (line.split()) #will have an array of strings.
                self.addVertex(res[1]) # Adding nodes to the self.graph.
                self.addVertex(res[0])
                print('Adding vertex and edges to:  ',res[0],'<-->',res[1])
                self.graph.add_edge(res[0],res[1])
                line=flux.readline() #line ++ read next line.
        self.graph.simplify()
        print(self.graph)
        return


if __name__ == '__main__':
    o = DataReader(Graph())
    print(o.__doc__)
    o.LoadData()
