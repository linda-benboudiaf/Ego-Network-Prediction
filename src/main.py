from collections import *
import Centrality
from igraph import *
import plotly as py
import plotly.graph_objs as go

def main():
    centrality = Centrality.Centrality(Graph(),
                    "/home/lbenboudiaf/Bureau/FacebookNetwork/dataset/SplitedData/train.edges",
                    "/home/lbenboudiaf/Bureau/FacebookNetwork/dataset/SplitedData/val.edges",
                    "/home/lbenboudiaf/Bureau/FacebookNetwork/dataset/SplitedData/test.edges")

    centrality.arg.LoadData()
    betweenness = centrality.BetweenCentrality()
    closeness = centrality.ClosenessCentrality()
    eigenVec = centrality.EigenVector_Centrality()

    #LoadLabels.
    nodesLabels = []
    edges = []
    #nodesLabels  = [nodesLabels.append(eachEgo['name']) for eachEgo in centrality.arg.graph.vs]
    for eachEgo in centrality.arg.graph.vs:
        nodesLabels.append(eachEgo['name'])
    #edges = [edges.append(edge.tuple) for edge in centrality.arg.graph.es] #ça marche pas je ne sais pas pourquoi.
    for e in centrality.arg.graph.es:
        edges.append(e.tuple)

    layout = centrality.arg.graph.layout('kk', dim=3)

    #Prepare coordinates for Nodes and Edges.
    Xn=[layout[n][0] for n in range(len(centrality.arg.graph.vs))]# x-coordinates of nodes
    Yn=[layout[n][1] for n in range(len(centrality.arg.graph.vs))]# y-coordinates
    Zn=[layout[n][2] for n in range(len(centrality.arg.graph.vs))]# z-coordinates

    #Lists of edges.
    Xe=[]
    Ye=[]
    Ze=[]
    for e in edges:
        Xe+=[layout[e[0]][0],layout[e[1]][0], None]# x-coordinates of edge ends
        Ye+=[layout[e[0]][1],layout[e[1]][1], None]
        Ze+=[layout[e[0]][2],layout[e[1]][2], None]

    trace1=go.Scatter3d(x=Xe,
                   y=Ye,
                   z=Ze,
                   mode='lines',
                   line=go.Line(color='rgb(125,125,125)', width=1),
                   hoverinfo='none'
                   )

    trace2=go.Scatter3d(x=Xn,
                  y=Yn,
                  z=Zn,
                  mode='markers',
                  name='Alters',
                  marker=go.Marker(symbol='circle',
                                color=eigenVec,
                                size=10,colorbar=go.ColorBar(
                   title='ColorBar'
               ),
                                colorscale='Viridis',
                                line=go.Line(color='rgb(158,18,130)', width=0.5)
                                ),
                  text=nodesLabels,
                  hoverinfo='text'
                  )

    axis=dict(showbackground=True,
             showline=True,
             zeroline=False,
             showgrid=True,
             showticklabels=True,
             title=''
             )
    plan = go.Layout(
             title="Facebook Ego-Network",
             width=1000,
             height=1000,
             showlegend=True,
             scene=go.Scene(
             xaxis=go.XAxis(axis),
             yaxis=go.YAxis(axis),
             zaxis=go.ZAxis(axis),
            ),
         margin=go.Margin(t=100),
        hovermode='closest',
        annotations=go.Annotations([
               go.Annotation(
                showarrow=True,
                xref='paper',
                yref='paper',
                x=0,
                y=0.1,
                xanchor='left',
                yanchor='top',
                font=go.Font(size=14)
                )
            ]),)

    data=go.Data([trace1, trace2])
    fig=go.Figure(data=data, layout=plan)
    fig.show()

pass
main()
