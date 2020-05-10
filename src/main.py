from collections import *
import Centrality
import DataReader
import math
import numpy as np
import pandas as pd
import random
from sklearn import linear_model, svm
from sklearn.linear_model import LinearRegression
from sklearn.metrics import log_loss, classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt



# Graph moduls
from igraph import *
import plotly as py
import plotly.graph_objs as go
import sys
if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")

def main():
    # Centrality for graph representation.
    centrality = Centrality.Centrality(Graph(),
                    "../FacebookNetwork/dataset/SplitedData/train.edges",
                    "../FacebookNetwork/dataset/SplitedData/val.edges",
                    "../FacebookNetwork/dataset/SplitedData/test.edges",
                    "../FacebookNetwork/dataset/SplitedData/neg_train.edges",
                    "../FacebookNetwork/dataset/SplitedData/neg_val.edges",
                    "../FacebookNetwork/dataset/SplitedData/neg_test.edges")

    centrality.arg.LoadEdgesData()
    betweenness = centrality.BetweenCentrality()
    closeness = centrality.ClosenessCentrality()
    eigenVec = centrality.EigenVector_Centrality()

    #PredictionModels, Split positive and negative data
    pos_train = DataReader.DataReader(Graph(),
                    "../FacebookNetwork/dataset/SplitedData/train.edges",
                    "../FacebookNetwork/dataset/SplitedData/val.edges",
                    "../FacebookNetwork/dataset/SplitedData/test.edges",
                    "../FacebookNetwork/dataset/SplitedData/neg_train.edges",
                    "../FacebookNetwork/dataset/SplitedData/neg_val.edges",
                    "../FacebookNetwork/dataset/SplitedData/neg_test.edges")
    neg_train = DataReader.DataReader(Graph(),
                    "../FacebookNetwork/dataset/SplitedData/train.edges",
                    "../FacebookNetwork/dataset/SplitedData/val.edges",
                    "../FacebookNetwork/dataset/SplitedData/test.edges",
                    "../FacebookNetwork/dataset/SplitedData/neg_train.edges",
                    "../FacebookNetwork/dataset/SplitedData/neg_val.edges",
                    "../FacebookNetwork/dataset/SplitedData/neg_test.edges")
    pos_val = DataReader.DataReader(Graph(),
                    "../FacebookNetwork/dataset/SplitedData/train.edges",
                    "../FacebookNetwork/dataset/SplitedData/val.edges",
                    "../FacebookNetwork/dataset/SplitedData/test.edges",
                    "../FacebookNetwork/dataset/SplitedData/neg_train.edges",
                    "../FacebookNetwork/dataset/SplitedData/neg_val.edges",
                    "../FacebookNetwork/dataset/SplitedData/neg_test.edges")
    neg_val = DataReader.DataReader(Graph(),
                    "../FacebookNetwork/dataset/SplitedData/train.edges",
                    "../FacebookNetwork/dataset/SplitedData/val.edges",
                    "../FacebookNetwork/dataset/SplitedData/test.edges",
                    "../FacebookNetwork/dataset/SplitedData/neg_train.edges",
                    "../FacebookNetwork/dataset/SplitedData/neg_val.edges",
                    "../FacebookNetwork/dataset/SplitedData/neg_test.edges")

    #Load Edges for each graph
    pos_train.LoadEdgesData()
    neg_train.LoadEdgesData()
    pos_val.LoadEdgesData()
    neg_val.LoadEdgesData()

    #Split into negative and positive files.
    pos_train.GenerateTrain() #we will get pos and neg files.
    pos_val.GenerateValidation() #we will get pos and neg files.

    #LabelData into dictionnary for files
    pos_train.LoadGeneratedFiles(pos_train.fluxTr)
    neg_train.LoadGeneratedFiles(neg_train.fluxTr_neg)
    pos_val.LoadGeneratedFiles(pos_val.fluxV)
    neg_val.LoadGeneratedFiles(neg_val.fluxVal_neg)

    X_train_pos = pos_train.LabelData()
    X_train_neg = neg_train.LabelData()

    X_val_pos = pos_val.LabelData()
    X_val_neg = neg_val.LabelData()

    print('----------------- Spliting and labeling data X & Y------------------------- \n')
    Y_train_pos = np.full(shape=(X_train_pos.shape[0],1), fill_value=1)
    Y_train_neg = np.full(shape=(X_train_neg.shape[0],1), fill_value=0)

    Y_val_pos = np.full(shape=(X_val_pos.shape[0],1), fill_value=1)
    Y_val_neg = np.full(shape=(X_val_neg.shape[0],1), fill_value=0)

    X_train = np.append(X_train_pos,X_train_neg,axis=0)
    y_train = np.append(Y_train_pos,Y_train_neg,axis=0)

    X_val = np.append(X_val_pos, X_val_neg, axis=0)
    y_val = np.append(Y_val_pos, Y_val_neg, axis=0)

    np.random.shuffle(X_train)
    np.random.shuffle(y_train)
    np.random.shuffle(X_val)
    np.random.shuffle(y_val)
    print('----------------- Done ------------------------- \n ')
    print('\n----------------- Linear Model Predictions ------------------------- \n')

    reg = linear_model.Ridge (alpha = .5)
    reg.fit(X=X_train[:-1],y=y_train[:-1])

    reg.predict(X_train[-1:])
    len(reg.predict(X_val))

    np.mean((reg.predict(X_val) - y_val)**2)
    print('Log loss ',log_loss(y_val,reg.predict(X_val)))

    print('\n ----------------- Linear LinearRegression ------------------------- \n')
    regressor = LinearRegression()
    regressor.fit(X_train, y_train)
    print('Slope',regressor.intercept_)
    y_pred = regressor.predict(X_val)
    df = pd.DataFrame({'Actual': y_val.flatten(), 'Predicted': y_pred.flatten()})
    #print(df)
    print('\n ----------------- SVM ------------------------- \n')
    clf_svm = svm.SVC()
    clf_svm.fit(X=X_train[:-1],y=y_train[:-1])
    print(log_loss(y_val,clf_svm.predict(X_val)))

    print('\n ------------------------ Implementing Kernel SVM | Polynomial  ------------------------ \n')
    svclassifier2 = svm.SVC(kernel='poly', degree=8,C=150) # this is the degre of the polynomial.
    svclassifier2.fit(X_train, y_train)

    #making prediction
    y_predp = svclassifier2.predict(X_val)

    #evaluating the poly svm
    print(confusion_matrix(y_val, y_predp))
    print(classification_report(y_val, y_predp))
    print('\n --------------------------- Implementing Kernel SVM | Linear -------------------------- \n')

    svclassifier1 = svm.SVC(kernel='linear')
    svclassifier1.fit(X_train, y_train)
    #Make predict
    y_pred = svclassifier1.predict(X_val)
    #Evaluating the Algorithm
    print(svclassifier1.score(X_val, y_val))
    print(confusion_matrix(y_val,y_pred))
    print(classification_report(y_val,y_pred))

    print('\n ------------------------ Implementing Kernel SVM | Sigmoid ------------------------ \n')
    svclassifier4 = svm.SVC(kernel='sigmoid')
    svclassifier4.fit(X_train, y_train)
    #making predict
    y_preds = svclassifier4.predict(X_val)
    #Evaluating Algorithm
    print(confusion_matrix(y_val, y_preds))
    print(classification_report(y_val, y_preds))

    print('\n------------------------ Implementing Kernel SVM | Gaussian ------------------------\n')
    svclassifier3 = svm.SVC(kernel='rbf')
    svclassifier3.fit(X_train, y_train)
    #making predit
    y_predg = svclassifier3.predict(X_val)
    #Evaluating Algorithm
    print(confusion_matrix(y_val, y_predg))
    print(classification_report(y_val, y_predg))

    print('\n ------------------------ KNN ------------------------ \n')
    sc = StandardScaler()
    sc.fit(X_train)
    X_train = sc.transform(X_train)
    X_val = sc.transform(X_val)
    print('Value for K Math.sqrt(len of X_train) -------> ',math.sqrt(len(X_train)))
    print("Please wait for graph representation ....")

    accuracy = [] #We agregate the Accuracy averages for 18 neighbors.
    f1_scores = [] #Metrics ...
    index = range(3,81)
    for i in index:
        classifier = KNeighborsClassifier(n_neighbors = i, metric= 'euclidean', weights='uniform', leaf_size= 30) #27 classifiers
        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_val) # Predict the class labels for the provided data
        conf_matrix = confusion_matrix(y_val, y_pred) # What we predit <VS> what actually is on test data.
        res = (conf_matrix[0, 0] + conf_matrix[1, 1]) / sum(sum(conf_matrix)) # Calculate Accuracy of our predit.
        accuracy.append(res)
        f1_scores.append(list(zip(y_val, y_pred)))

    print('In the range of 3 to 39 we have this values of accuracy')
    print(accuracy)

    # Evaluate the Model.
    print('We evaluate the Matrix of Confusion')
    mc = confusion_matrix(y_val, y_pred)
    print(classification_report(y_val, y_pred))
    print(mc)
    # Graph representation

    plt.figure(figsize=(10, 6), num='Knn Algorithm Facebook Network Prediction')
    plt.plot(index, accuracy, color='green', linestyle='dashed', marker='o',
             markerfacecolor='blue', markersize=10)
    plt.title('Accuracy ratio according to K values')
    plt.xlabel('K Values')
    plt.ylabel('Accuracy average')
    plt.show()

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
                   title='Node Degree'
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
