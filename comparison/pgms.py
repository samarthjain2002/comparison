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