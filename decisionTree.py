import math
import csv
import numpy as np
import sys


inputFile=sys.argv[1]
delimiter=sys.argv[2]
impurityMeasurement=raw_input('Enter 1 for Gini and 2 for Information Gain ')

class Node:
    def __init__(self):
        nodeAttribute=''
        classLabel=''
        splitPnt=0       
        left=None
        right=None

########################################################################
#TREE RECURSION FUNCTION TO BUILD THE TREE
#PARAMETERS: 
#        noOfColumns : the no of attributes for each item.
#        dataSet: the class column from the data

#RETURNS: 
#returns the root of the decision tree
########################################################################
def RecursiveGrowth(dataSet,noOfColumns):
    if(stoppingCondition(dataSet) == True or len(dataSet) == 1):
        leaf=Node()
        
        classData=[row[noOfColumns-1] for row in dataSet]
        max=0
        for i in np.unique(classData):
            count=classData.count(i)
            if(max<count):
                max=count
                index=i
    
        leaf.classLabel=index
        leaf.left=None
        leaf.right=None
        return leaf
    else:
        root=Node()
        bestAttribute=findInternalNode(dataSet)
        root.nodeAttribute=bestAttribute[0]
        root.splitPnt=bestAttribute[1]
        dataSet=np.array([np.array(x) for x in dataSet])
        leftSubset=[]
        rightSubset=[]
        for i in range(0,len(dataSet)):
            if(float(dataSet[i,int(root.nodeAttribute)]) <= bestAttribute[1]):
                leftSubset.append(dataSet[i])
            else:
                rightSubset.append(dataSet[i])
        root.left=RecursiveGrowth(leftSubset,noOfColumns)
        root.right=RecursiveGrowth(rightSubset,noOfColumns)   
    return root




########################################################################
#CALCULATES THE BEST SPLIT POINT OF THE WHOLE DATA TO FIND BEST INTERNAL 
#NODE
#PARAMETERS: 
#        data : the data on which the decision tree is being tested

#RETURNS: 
#a list with split point and the corresponding attribute.
########################################################################
def findInternalNode(data):
    
    noOfCols = len(data[0])
    min=1
    for i in range(0,noOfCols-1):        
        sortedData= sorted(data,key=lambda x: x[i])
        sortedData=np.array(sortedData)
        
        impurityforAttributes=splitCriteriaForSingleAttribute(data, sortedData[:,i], sortedData[:,-1])
        
        if min>= impurityforAttributes[0]:
            min=impurityforAttributes[0]
            splitPoint=impurityforAttributes[1]
            minimumIndex=i
            if min==0:
                break
    return (minimumIndex,splitPoint)



   
########################################################################
#DETERMINES WHEN THE RECURSION SHOULD STOP
#PARAMETERS: 
#        dataSet : the data on which the decision tree is being tested
#        
#RETURNS: 
#a boolean to stop or not at a particular node. 
########################################################################
def stoppingCondition(dataSet):
    NoOfColumns=len(dataSet[0])
    dataSet=np.array(dataSet)
    uniqueValues=np.all(dataSet==dataSet[0,:], axis=0)
    flag=1
    
    for i in range(0, NoOfColumns):
        if uniqueValues[i]==False:
            flag=0
            break    
    classData=dataSet[:,(NoOfColumns-1)]

    noOfClasses=len(np.unique(classData))                
    if noOfClasses==1 or flag==1:
        return True
    return False

########################################################################
#CALCULATES THE BEST SPLIT POINT OF A CONTINUOUS ATTRIBUTE
#CALCULATES BASED ON GINI/INFO GAIN AS PER USER REQUEST
#PARAMETERS: 
#        data : the data on which the decision tree is being tested
#        targetAttribute: the attribute for which the split point is 
#                        being calculated.
#        classData: the class column from the data

#RETURNS: 
#a list with split point and the corresponding index of the split point.
########################################################################
def splitCriteriaForSingleAttribute(data, targetAttribute, classData):
    
    NoOfUniqueClasses= np.unique(classData)
    splitPointsList=[]
    splitPointsList.append(float(targetAttribute[0])-0.5)
    sameIndices=[]
    for i in range(0, len(targetAttribute)-1):
        weightedAverage=[]
        avg=(float(targetAttribute[i])+float(targetAttribute[i+1]))/2
        if(avg == float(targetAttribute[i])):
            sameIndices.append(i+1)
        splitPointsList.append((float(targetAttribute[i])+float(targetAttribute[i+1]))/2)
    splitPointsList.append(float(targetAttribute[-1])+0.5)
    
    for i in range(0, len(splitPointsList)):
        if(i in sameIndices):
            weightedAverage.append(1)
        else:
            entropyForEachPoint=  np.zeros((len(NoOfUniqueClasses),2), dtype=float)
            Lesser=0.0
            GreaterSum=0.0
            SumLesser=0.0
            gprobValLess=0.0
            gprobValGrt=0.0
            Greater=0.0
            for k in range(0, len(NoOfUniqueClasses)):
                countG=0
                countL=0       
                for j2 in range(i, len(classData)):
                    if classData[j2]==NoOfUniqueClasses[k]:
                        countG+=1  
                for j in range(0, i):
                    if classData[j]==NoOfUniqueClasses[k]:
                        countL+=1          
                entropyForEachPoint[k,0]=countL
                entropyForEachPoint[k,1]=countG
                GreaterSum+=entropyForEachPoint[k,1]
                SumLesser+=entropyForEachPoint[k,0]
            # calcualates Entropy /Gini based on the parameter passed by the user.
            if impurityMeasurement=="2":    
                for k in range(0, len(NoOfUniqueClasses)):
                    if SumLesser!=0:  
                        probabilityVal=entropyForEachPoint[k,0]/SumLesser
                        if probabilityVal==0:
                            probabilityVal=1.0
                        Lesser+=(-probabilityVal)*math.log(probabilityVal , 2)
                    else: 
                        Lesser=0.0
                    if GreaterSum!=0:
                        probVal=entropyForEachPoint[k,1]/GreaterSum
                        if probVal==0:
                            probVal=1.0
                        Greater+=(-entropyForEachPoint[k,1]/GreaterSum)*math.log(probVal, 2)
                    else: 
                        Greater=0.0
                weightedAverage.append((SumLesser*Lesser/len(classData))+(GreaterSum*Greater/len(classData)))
        
            elif impurityMeasurement=="1":
                for k in range(0, len(NoOfUniqueClasses)):
                    
                    if SumLesser!=0:
                        gprobValGrt+=((entropyForEachPoint[k,0])/SumLesser)**2
                    else:
                        gprobValGrt=1.0
                    giniLesser=1-gprobValGrt
                    
                    if GreaterSum!=0:
                        gprobValLess+=((entropyForEachPoint[k,1])/GreaterSum)**2
                    else:
                        gprobValLess=1.0
                    giniGreater=1-gprobValLess
                weightedAverage.append((SumLesser*giniLesser/len(classData))+(GreaterSum*giniGreater/len(classData)))
            else:
                print "Please enter only 1 or 2 ."

                
        IndexListForEntropies=weightedAverage.index(min(weightedAverage))
                
    splitPointDetailsByEntropy=[]
    splitPointDetailsByEntropy.append(min(weightedAverage))
    splitPointDetailsByEntropy.append(splitPointsList[IndexListForEntropies])
    return  splitPointDetailsByEntropy





########################################################################
#GIVES THE PREDICTED CLASS FOR ALL ROWS IN A LIST
#CALCULATES BASED ON GINI/INFO GAIN AS PER USER REQUEST
#PARAMETERS: 
#        root : the root of the recursion tree
#        dataSet: the class column from the data

#RETURNS: 
#the Test dataSet with an extra column of predicted Class
########################################################################
def predict(root,dataSet):
    NoOfRows=len(dataSet)
    dataSet=np.array(dataSet)
    matrix=[]   
    for i in range(0,NoOfRows):
        matrix.append(predictClass(root,dataSet[i]))
    return matrix   



########################################################################
#PARTITIONS THE DATA INTO TRAIN-TEST PARTITIONS 10 DIFFERENT TIMES
#USING A RANDOM NUMBER GENERATION FUNCTION
#PARAMETERS: 
#        dataSet: the class column from the data

#RETURNS: 
#returns two lists of lists with training data and corresponding
# testing data on the same index. 
#trainingData[i]'s test split will be testData[i]
########################################################################
def dataPartitioningForTrainTestSplit(dataSet):
    
    #shuffles the data by assigning random numbers 
    np.random.shuffle(dataSet)
    noOfFolds=10
    splitSize= len(dataSet)/noOfFolds    
    trainingData=[]
    testData=[]    
    for i in range(0, noOfFolds):
        trainingData.append(np.empty((len(dataSet)-splitSize, len(dataSet[0])), dtype=object))
        testData.append(dataSet[i*splitSize:(i+1)*(splitSize)])
        
    #creates partitions of the shuffled data
    for i in range(0,noOfFolds):
        
        l=0
        for j in range(0, noOfFolds):
            for k in range(0,splitSize):
                if j!=i:            
                    trainingData[i][l]=testData[j][k]
                    l+=1

    return (trainingData,testData)

########################################################################
#PREDICTS THE CLASS OF EACH ROW IN THE TEST DATA
#PARAMETERS: 
#        root : the root of the recursion tree
#        SetOfAttributes: each row of the test data 

#RETURNS: 
#    predicted classLabel for each row.
########################################################################
def predictClass(root,SetOfAttributes):

    if root.left==None and root.right==None:
        return root.classLabel
    if float(SetOfAttributes[root.nodeAttribute])<=root.splitPnt:
        classLabel=predictClass(root.left,SetOfAttributes)
    else:
        classLabel=predictClass(root.right,SetOfAttributes)
    return classLabel


########################################################################
#DISPLAYS THE DECISION TREE FOR THE DATA SET GIVEN
#PARAMETERS: 
#        root: the root of the tree

#RETURNS: 
#prints the tree. 
########################################################################
def displayTree(root):
    if(root.left == None and root.right == None):
        print "Class is:",root.classLabel
    else:
        displayTree(root.left)
        print "Node Attribute: ",root.nodeAttribute
        displayTree(root.right)

with open(inputFile) as file:
    reader = csv.reader(file,delimiter=sys.argv[2])
    fl=list(reader)

fl=np.array([row for row in fl])
trainTestPartition=dataPartitioningForTrainTestSplit(fl)
NoOfColumns=len(fl[0])
accuracy=0.0
for i in range(0,10):
    print "Training Iteration: ",i
    root=RecursiveGrowth(trainTestPartition[0][i],NoOfColumns)
    displayTree(root)
    predictClassMatrix=predict(root, trainTestPartition[1][i])
    columnNumber=len(trainTestPartition[1][i][0])
    column=[rows[columnNumber-1] for rows in trainTestPartition[1][i]]
    count=0
    for j in range(0,len(trainTestPartition[1][i])):
        if str(predictClassMatrix[j])==str(column[j]):
            count+=1
    accuracy+=float(count)/float(len(trainTestPartition[1][i]))
accuracyAverage=accuracy/10
print accuracyAverage
