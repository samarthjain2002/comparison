def pgm1():
    print("""
def aStarAlgo(start_node, stop_node):
    open_set = set(start_node)
    closed_set = set()
    g = {}
    parents = {}
    g[start_node] = 0
    parents[start_node] = start_node
    while open_set:
        n = None
        for v in open_set:
            if n == None or g[v] + heuristic(v) < g[n] + heuristic(n):
                n = v
        if n == stop_node or Graph_nodes[n] == None:
            pass
        else:
            for (m,weight) in get_neighbors(n):
                if m not in open_set and m not in closed_set:
                    open_set.add(m)
                    parents[m] = n
                    g[m] = g[n] + weight
                else:
                    if g[m] > g[n] + weight:
                        g[m] = g[n] + weight
                        parents[m] = n
                        if m in closed_set:
                            closed_set.remove(m)
                            open_set.add(m)
        if n == None:
            print('Path does not exist!')
            return None
        if n == stop_node:
            path = []
            while parents[n] != n:
                path.append(n)
                n = parents[n]
            path.append(start_node)
            path.reverse()
            print('Path found: {}'.format(path))
            return path
        open_set.remove(n)
        closed_set.add(n)
    print('Path does not exist!')
    return None
            
def get_neighbors(v):
    if v in Graph_nodes:
        return Graph_nodes[v]
    else:
        return None
    
def heuristic(n):
    H_dist = {
        'S': 5,
        'A': 4,
        'B': 5,
        'E': 0,
    }
    return H_dist[n]

Graph_nodes = {
    'S': [('A', 1),('B', 2)],
    'A': [('E', 13)],
    'B': [('E', 5)]
}

aStarAlgo('S','E')
    """)


def pgm2():
    print("""
class Graph:
    def __init__(self, graph, heuristicNodeList, startNode):
        self.graph = graph
        self.H = heuristicNodeList
        self.start = startNode
        self.parent = {}
        self.status = {}
        self.solutionGraph = {}
        
    def applyAOStar(self):    #starts a recursion AO* algorithm
        self.aoStar(self.start, False)
        
    def getNeighbors(self, v):    #gets the neighbors of a given node
        return self.graph.get(v,'')
    
    def getStatus(self, v):    #returns the status of a given node
        return self.status.get(v, 0)
    
    def setStatus(self, v, val):    #set the status of a given node
        self.status[v] = val
        
    def getHeuristicNodeValue(self, n):
        return self.H.get(n, 0)
        #always return the heuristic value of a given node
        
    def setHeuristicNodeValue(self,n,value):
        self.H[n] = value
        #set the revised Heuristic value of a given node
        
    def printSolution(self):
        print("FOR GRAPH SOLUTION, TRAVERSE THE GRAPH FROM THE STARTNODE: ",self.start)
        print("----------------------------------------------------------------------")
        print(self.solutionGraph)
        print("----------------------------------------------------------------------")
        
    def computeMinimumCostChildNodes(self,v):
        minimumCost = 0
        costToChildNodeListDict = {}
        costToChildNodeListDict[minimumCost] = []
        flag = True
        
        for nodeInfoTupleList in self.getNeighbors(v):
            cost = 0
            nodeList = []
            for c,weight in nodeInfoTupleList:
                cost = cost + self.getHeuristicNodeValue(c) + weight
                nodeList.append(c)
                
            if flag == True:
                minimumCost = cost
                costToChildNodeListDict[minimumCost] = nodeList
                flag = False
                
            else:
                if minimumCost > cost:
                    minimumCost = cost
                    costToChildNodeListDict[minimumCost] = nodeList

        return minimumCost,costToChildNodeListDict[minimumCost]
        
    def aoStar(self, v, backTracking):
        print("HEURISTIC VALUES: ",self.H)
        print("SOLUTION GRAPH: ",self.solutionGraph)
        print("PROCESSING NODE: ",v)
        print("-----------------------------------")
        if self.getStatus(v) >= 0:
            minimumCost, childNodeList = self.computeMinimumCostChildNodes(v)
            print(minimumCost, childNodeList)
            self.setHeuristicNodeValue(v, minimumCost)
            self.setStatus(v, len(childNodeList))
            solved = True
            for childNode in childNodeList:
                self.parent[childNode] = v
                if self.getStatus(childNode) != -1:
                    solved = False
                    
            if solved == True:
                self.setStatus(v, -1)
                self.solutionGraph[v] = childNodeList
                
            if v != self.start:
                self.aoStar(self.parent[v], True)
                
            if backTracking == False:
                for childNode in childNodeList:
                    self.setStatus(childNode, 0)
                    self.aoStar(childNode, False)

h1 = {
    'A':1,
    'B':6,
    'C':2,
    'D':12,
    'E':2,
    'F':1,
    'G':5,
    'H':7,
    'I':7,
    'J':1
}

graph1 = {
    'A':[[('B',1),('C',1)],[('D',1)]],
    'B':[[('G',1)],[('H',1)]],
    'C':[[('J',1)]],
    'D':[[('E',1),('F',1)]],
    'G':[[('I',1)]]
}

G1 = Graph(graph1, h1, 'A')
G1.applyAOStar()
G1.printSolution()
          """)


def pgm3():
    print("""
import numpy as np
import pandas as pd

data = pd.read_csv("Book3.csv", header = None)
concepts = np.array(data.iloc[ : , : -1])
print("\nInstances are:\n", concepts)
target = np.array(data.iloc[ : , -1])
print("\nTarget values are:\n", target)

def learn(concepts, target):
    specific_h = concepts[0].copy()
    print("\nInitialization of specific_h and general_h")
    print("\nSpecific Boundary: ", specific_h)
    general_h = [["?" for i in range(len(specific_h))] for i in range(len(specific_h))]
    print("\nGeneric Boundary: ", general_h)
    
    for i, h in enumerate(concepts):
        print("\nInstance", i + 1, "is", h)
        if target[i] == "Yes":
            print("Instance is positive")
            for x in range(len(specific_h)):
                if h[x] != specific_h[x]:
                    specific_h[x] = '?'
                    general_h[x][x] = '?'
        
        if target[i] == "No":
            print("Instance is negative")
            for x in range(len(specific_h)):
                if h[x] != specific_h[x]:
                    general_h[x][x] = specific_h[x]
                else:
                    general_h[x][x] = '?'
                    
        print("Specific Boundary after", i + 1, "instance is ", specific_h)
        print("Generic Boundary after", i + 1, "instance is ", general_h)
        print("\n")
    
    indeces = []
    for i in range(len(general_h)):
        if general_h[i] != ['?','?','?','?','?','?']:
            indeces.append(general_h[i])
    
    return specific_h, indeces

s_final, g_final = learn(concepts, target)
print("Final specific_h:", s_final, sep="\n")
print("Final general_h: ", g_final, sep="\n")
          """)


def pgm5():
    print("""
import numpy as np

slept_studied = np.array(([2,9],[1,5],[3,6]), dtype = float)    #Features[Hrs Slept, Hrs Studied]
marksObt = np.array(([92],[86],[89]), dtype = float)       #Labels[Marks Obtained]
slept_studied = slept_studied / np.amax(slept_studied)
print(slept_studied)
marksObt = marksObt / 100
print(marksObt)

def sigmoid(slept_studied):
    return 1 / (1 + np.exp(-slept_studied))

def sigmoid_grad(slept_studied):
    return slept_studied * (1 - slept_studied)

#Variable Initialization
epoch = 1000     #Setting training iterations
eta = 0.2     #Setting learning rate
input_neurons = 2    #Number of features in dataset
hidden_neurons = 3    #Number of hidden lamarksObter neurons
output_neurons = 1    #Number of output lamarksObter neurons
#Weight and Bias random initialization
wh = np.random.uniform(size = (input_neurons, hidden_neurons))
bh = np.random.uniform(size = (1, hidden_neurons))
wout = np.random.uniform(size = (hidden_neurons, output_neurons))
bout = np.random.uniform(size = (1, output_neurons))

for i in range(epoch):    #Forward propogation
    h_ip = np.dot(slept_studied, wh) + bh
    h_act = sigmoid(h_ip)
    o_ip = np.dot(h_act, wout) + bout
    output = sigmoid(o_ip)
    #Backward propogation
    #Error at output lamarksObter
    eo = marksObt = output
    outgrad = sigmoid_grad(output)
    d_output = eo * outgrad
    #Error at hidden lamarksObter
    eh = d_output.dot(wout.T)
    hiddengrad = sigmoid_grad(h_act)
    d_hidden = eh * hiddengrad
    wout += h_act.T.dot(d_output) * eta
    wh += slept_studied.T.dot(d_hidden) * eta
    
print("\n\n\nNormalized input: \n " + str(slept_studied))
print("Actual Output: \n" + str(marksObt))
print("Predicted Output: \n", output)
          """)


def pgm6():
    print("""
import csv,random,math
import statistics as st
from statistics import stdev

def loadcsv(filename):
    lines=csv.reader(open(filename,"r"));
    dataset=list(lines)
    for i in range(len(dataset)):
        dataset[i]=[float(x) for x in dataset[i]]
    return dataset

def splitDataset(dataset,splitRatio):
    testSize=int(len(dataset)*splitRatio);
    trainSet=list(dataset);
    testSet=[]
    while len(testSet)<testSize:
        index=random.randrange(len(trainSet));
        testSet.append(trainSet.pop(index))
    return[trainSet,testSet]

def separatebyclass(dataset):
    separated={}
    for i in range(len(dataset)):
        x=dataset[i]
        if(x[-1] not in separated):
            separated[x[-1]]=[]
        separated[x[-1]].append(x)
    return separated 

def mean(numbers):
    return sum(numbers)/float(len(numbers))

def stdev(num):
    avg=mean(num)
    variance=sum([pow(x-avg,2) for x in num])/float(len(num)-1)
    return math.sqrt(variance)

def compute_mean_std(dataset):
    mean_std=[(st.mean(attribute),st.stdev(attribute)) for attribute in zip(*dataset)];
    del mean_std[-1]
    return mean_std

def summarizebyclass(dataset):
    separated=separatebyclass(dataset)
    summary={}
    for classvalue,instances in separated.items():
        summary[classvalue]=compute_mean_std(instances)
    return summary

def estimateprobability(x,mean,stdev):
    exponent=math.exp(-(math.pow(x-mean,2)/(2*math.pow(stdev,2))))
    return(1/(math.sqrt(2*math.pi)*stdev))*exponent

def calculateclassprobabilities(summaries,testvector):
    p={}
    for classvalue,classsummaries in summaries.items():
        p[classvalue]=1
        for i in range (len(classsummaries)):
            mean,stdev=classsummaries[i]
            x=testvector[i]
            p[classvalue]*=estimateprobability(x,mean,stdev);
    return p

def predict(summaries,testvector):
    all_p=calculateclassprobabilities(summaries,testvector)
    bestlabel,bestprob=None,-1
    for lbl,p in all_p.items():
        if bestlabel is None or p>bestprob:
            bestprob=p
            bestlabel=lbl
    return bestlabel

def perform_classification(summaries,testSet):
    predictions=[]
    for i in range(len(testSet)):
        result=predict(summaries,testSet[i])
        predictions.append(result)
    return predictions

def getaccuracy(testSet,predictions):
    correct=0
    for i in range(len(testSet)):
        if testSet[i][-1]==predictions[i]:
            correct +=1
    return(correct/float(len(testSet)))*100.0 
dataset=loadcsv('pima-indians-diabetes.csv');
print("Dataset loaded----")
print("Total instances=",len(dataset))
print("Total attribute=",len(dataset[0])-1)
print("first five instances:")
for i in range(5):
    print(i+1,':',dataset[i])
splitratio=0.2
trainingset,testSet=splitDataset(dataset,splitratio)
print('\n Dataset is split into training and testing set\n')
print('\n training examples = {0}\n testing examples = {1}'.format(len(trainingset),len(testSet)))
summaries=summarizebyclass(trainingset)
predictions=perform_classification(summaries,testSet)
accuracy=getaccuracy(testSet,predictions)
print("\n Accuracy of Naive Bayesian classifier:",accuracy)
          """)
    

def pgm7():
    print("""
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.cluster import KMeans
import pandas as pd
import numpy as np

#import some data
iris = datasets.load_iris()
#print(iris)
X = pd.DataFrame(iris.data)
#print(X)
X.columns = ['Sepal_Length','Sepal_Width','Petal_Length','Petal_Width']
#print(X)
y = pd.DataFrame(iris.target)
#print(y)
y.columns = ['targets']

#Build the k-Means model
model = KMeans(n_clusters=3)
model.fit(X)    #model.labels_: Gives cluster no for which sample belongs to

#Visualize the clustering results
plt.figure(figsize=(14,14))
colormap = np.array(['red','lime','black'])

#Plot the original classification using Petal features
plt.subplot(2,2,1)
plt.scatter(X.Petal_Length,X.Petal_Width,c=colormap[y.targets],s=40)
plt.title('Real clusters')
plt.xlabel('Petal Length')
plt.ylabel('Petal Width')

#Plot the model classification
plt.subplot(2,2,2)
plt.scatter(X.Petal_Length,X.Petal_Width,c=colormap[model.labels_],s=40)
plt.title('k-Means Clustering')
plt.xlabel('Petal Length')
plt.ylabel('Petal Width')

#print(model.labels_)

#General EM for GMM
from sklearn import preprocessing
#Transform your data such that its distribution will have a mean value 0 and standard deviation of 1.
scaler = preprocessing.StandardScaler()
scaler.fit(X)
xsa = scaler.transform(X)
#print(xsa)
xs = pd.DataFrame(xsa,columns=X.columns)
#print(xs)

from sklearn.mixture import GaussianMixture
gmm=GaussianMixture(n_components=3)
gmm.fit(xs)
gmm_y = gmm.predict(xs)
#print(gmm_y)

plt.subplot(2,2,3)
plt.scatter(X.Petal_Length,X.Petal_Width,c=colormap[gmm_y],s=40)
plt.title('GMM Clustering')
plt.xlabel('Petal Length')
plt.ylabel('Petal Width')
print("Observation: The GMM using EM algorithm based clustering matched the true labels more closely than k-Means")
plt.show()
          """)
    

def pgm8():
    print("""
from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier

import numpy as np
from sklearn.model_selection import train_test_split

iris_dataset = load_iris()
#print(iris_dataset)

print("\nIRIS FEATURES \n TARGET NAMES:\n",iris_dataset.target_names)

for i in range(len(iris_dataset.target_names)):
    print("\n[{0}]:[{1}]".format(i, iris_dataset.target_names[i]))
    
print("\n IRIS DATA: \n",iris_dataset["data"])

X_train,X_test,y_train,y_test = train_test_split(iris_dataset["data"], iris_dataset["target"], random_state=0)

print("\nTarget:\n",iris_dataset["target"])
print("\nX TRAIN:\n",X_train)
print("\nX TEST:\n",X_test)
print("\nY TRAIN:\n",y_train)
print("\nY TEST:\n",y_test)
kn = KNeighborsClassifier(n_neighbors=1)
kn.fit(X_train,y_train)

for i in range(len(X_test)):
    x_new = np.array([X_test[i]])
    prediction = kn.predict(x_new)
    print("\nActual: {0} {1}, Predicted: {2} {3}"
          .format(y_test[i],iris_dataset["target_names"][y_test[i]],prediction,iris_dataset["target_names"][prediction]))
    
print("\nTEST SCORE[ACCURACY]:{:.2f}\n".format(kn.score(X_test,y_test)))
          """)
    

def pgm9():
    print("""
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def kernel(point,xmat,k):
    m,n = np.shape(xmat)
    weights = np.mat(np.eye((m)))
    for j in range(m):
        diff = point - X[j]
        weights[j,j] = np.exp(diff * diff.T / (-2.0 * k ** 2))
    return weights

def localWeight(point,xmat,ymat,k):
    wei = kernel(point, xmat, k)
    W = (X.T * (wei * X)).I * (X.T * (wei * ymat.T))
    return W

def localWeightRegression(xmat,ymat,k):
    m,n = np.shape(xmat)
    ypred = np.zeros(m)
    for i in range(m):
        ypred[i] = xmat[i] * localWeight(xmat[i],xmat,ymat,k)
    return ypred

def graphplot(X,ypred):
    sortindex = X[:,1].argsort(0)
    xsort = X[sortindex][:,0]
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.scatter(bill,tip,color='green')
    ax.plot(xsort[:,1],ypred[sortindex],color='red',linewidth=4)
    plt.xlabel('Total Bill')
    plt.ylabel('Tip')
    plt.show()
    
data = pd.read_csv('data10_tips.csv')
print(data)
bill = np.array(data.total_bill)
tip = np.array(data.tip)
mbill = np.mat(bill)
mtip = np.mat(tip)
m = np.shape(mbill)[1]
one = np.mat(np.ones(m))
X = np.hstack((one.T,mbill.T))
ypred = localWeightRegression(X,mtip,0.5)
graphplot(X,ypred)
          """)