import pandas as pd
import numpy as np
import math
import pyoperator
import matplotlib.pyplot as plt
dataSet=pd.read_csv('diabetes.csv')
# print(dataSet.isnull().any())
# print(dataSet.info())
# print(dataSet.values)
ft = dataSet.keys()
print(ft)
# print(dataSet.groupby('Outcome').size())



class DecisionTree(object):
    def __init__(self,algor,dataSet,Thresh=(0.01,10),threshVal=0.5):
        self.algor=algor  #算法
        self.dataSet=dataSet  #数据集
        self.Thresh=Thresh  #前剪枝的一些阈值
        self.threshVal=threshVal  #决定分类的阈值

    # 计算信息增益(ID3)
    def calcShannonEnt(self,dataSet):
       numEntries = len(dataSet)#样本数
       labelCounts = {}
       for featVec in dataSet:#遍历每个样本
          currentLabel=featVec[-1]#当前样本的类别
          if currentLabel not in labelCounts.keys():#生成类别字典
            labelCounts[currentLabel]=0
          labelCounts[currentLabel]+=1
       shannonEnt=0.0
       for key in labelCounts:#计算信息熵
          prob = float(labelCounts[key])/numEntries
          shannonEnt = shannonEnt-prob *math.log(prob,2)
       return shannonEnt

    #划分数据集,把位置axis对应的数据是否等于value进行分类，featVec[axis]已经作为分类特征，需要去除
    def splitDataSet(self,dataSet,axis,value):
        retDataSet=[]
        featVec=[]
#        if continuous==True:#连续值
#           for featVec in self.dataSet:
#                if part==0 and float(featVec[axis])<=value:
#                    reducedFeatVec=featVec[:axis]
#                    reducedFeatVec.extend(featVec[axis+1:])
#                    retDataSet.append(reducedFeatVec)
#                if part==1 and float(featVec[axis])>value:
#                    reducedFeatVec=featVec[:axis]
#                    reducedFeatVec.extend(featVec[axis+1:])
#                    retDataSet.append(reducedFeatVec)
        #离散值
        for featVec in dataSet:
            if featVec[axis]==value:
                reducedFeatVec=featVec[:axis]
                reducedFeatVec.extend(featVec[axis+1:])
                retDataSet.append(reducedFeatVec)
                # print(retDataSet)
        return retDataSet

    #选择最佳特征值
    def chooseBestFeatureToSplit(self,dataSet1):
        numFeatures=len(dataSet1[0])-1#属性
        baseEntropy=self.calcShannonEnt(dataSet1)
        bestInfoGain=0.0
        bestFeature=-1
        for i in range(numFeatures):
            featList=[example[i] for example in dataSet1]
            del(featList[0])
            uniqueVals=set(featList)
            newEntropy=0.0
            for value in uniqueVals:
                subDataSet=self.splitDataSet(dataSet1,i,value)
                prob=len(subDataSet)/float(len(dataSet1))
                newEntropy+=prob*self.calcShannonEnt(subDataSet)
            infoGain=baseEntropy-newEntropy
            if(infoGain>bestInfoGain):
                bestInfoGain=infoGain
                bestFeature=i
        return bestFeature

    #建树
    def createtree(self,dataSet1,labels):
        classList=[example[-1] for example in dataSet1]
        if classList.count(classList[0])==len(classList):
            return classList[0]#如果只剩一个类别，返回
        if len(dataSet1[0])==1:
            return self.majorityCnt(classList)# 如果所有特征都被遍历完了，调用多数表决函数
        bestFeat=self.chooseBestFeatureToSplit(dataSet1)
        # print(bestFeat)
        bestFeatLabel=labels[bestFeat]
        myTree={bestFeatLabel: {}}
        # print(bestFeat)
        del (labels[bestFeat])# 已经选择的特征不再参与分类
        featValues=[example[bestFeat] for example in dataSet1]
        uniqueValue=set(featValues)# 该属性所有可能取值，也就是节点的分支
        # print(uniqueValue)
        for value in uniqueValue:# 对每个分支，递归构建树
            subLabels=labels[:]
            myTree[bestFeatLabel][value]=self.createtree(
                self.splitDataSet(dataSet1,bestFeat, value), subLabels)
        return myTree

    #通过排序返回出现次数最多的类别,用于多数表决
    def majorityCnt(self,dataSet,classList):
        classCount = {}
        for vote in classList:
            if vote not in classCount.keys():
                classCount[vote] = 0
                classCount[vote] += 1
        sortedClassCount = sorted(classCount.iteritems(),
                                  key=pyoperator.itemgetter(1), reverse=True)
        return sortedClassCount[0][0]

    # C4.5划分数据集, axis:按第几个特征划分, value:划分特征的值, LorR: value值左侧（小于）或右侧（大于）的数据集
    def splitDataSet_c(self,dataSet, axis, value, LorR='L'):
        retDataSet = []
        featVec = []
        if LorR == 'L':
            for featVec in dataSet:
                if float(featVec[axis]) < value:
                    retDataSet.append(featVec)
        else:
            for featVec in dataSet:
                if float(featVec[axis]) > value:
                    retDataSet.append(featVec)
        return retDataSet

   # C4.5选择最好的数据集划分方式
    def chooseBestFeatureToSplit_c(self,dataSet,labelProperty):
        numFeatures = len(labelProperty)  # 特征数
        baseEntropy=self.calcShannonEnt(dataSet)  # 计算根节点的信息熵
        bestInfoGain = 0.0
        bestFeature = -1
        bestPartValue = None  # 连续的特征值，最佳划分值
        for i in range(numFeatures):  # 对每个特征循环
            featList = [example[i] for example in dataSet]
            uniqueVals = set(featList)  # 该特征包含的所有值
            newEntropy = 0.0
            bestPartValuei = None
            if labelProperty[i] == 0:  # 对离散的特征
                for value in uniqueVals:  # 对每个特征值，划分数据集, 计算各子集的信息熵
                    subDataSet = self.splitDataSet(dataSet, i, value)
                    prob = len(subDataSet) / float(len(dataSet))
                    newEntropy += prob * self.calcShannonEnt(subDataSet)
            else:  # 对连续的特征
                sortedUniqueVals = list(uniqueVals)  # 对特征值排序
                sortedUniqueVals.sort()
                listPartition = []
                minEntropy = inf
                for j in range(len(sortedUniqueVals) - 1):  # 计算划分点
                    partValue = (float(sortedUniqueVals[j]) + float(
                        sortedUniqueVals[j + 1])) / 2
                    # 对每个划分点，计算信息熵
                    dataSetLeft = self.splitDataSet_c(dataSet, i, partValue, 'L')
                    dataSetRight = self.splitDataSet_c(dataSet, i, partValue, 'R')
                    probLeft = len(dataSetLeft) / float(len(dataSet))
                    probRight = len(dataSetRight) / float(len(dataSet))
                    Entropy = probLeft * self.calcShannonEnt(
                        dataSetLeft) + probRight * self.calcShannonEnt(dataSetRight)
                    if Entropy < minEntropy:  # 取最小的信息熵
                        minEntropy = Entropy
                        bestPartValuei = partValue
                newEntropy = minEntropy
            infoGain = baseEntropy - newEntropy  # 计算信息增益
            if infoGain > bestInfoGain:  # 取最大的信息增益对应的特征
                bestInfoGain = infoGain
                bestFeature = i
                bestPartValue = bestPartValuei
        return bestFeature, bestPartValue

    def createTree_c(self,dataSet,labels,labelProperty):
        classList = [example[-1] for example in dataSet]  # 类别向量
        if classList.count(classList[0]) == len(classList):  # 如果只有一个类别，返回
            return classList[0]
        if len(dataSet[0]) == 1:# 如果所有特征都被遍历完了，返回出现次数最多的类别
            return self.majorityCnt(classList)
        bestFeat, bestPartValue = self.chooseBestFeatureToSplit_c(dataSet,labelProperty)  # 最优分类特征的索引
        if bestFeat == -1:  # 如果无法选出最优分类特征，返回出现次数最多的类别
            return self.majorityCnt(classList)
        if labelProperty[bestFeat] == 0:  # 对离散的特征
            bestFeatLabel = labels[bestFeat]
            myTree = {bestFeatLabel: {}}
            labelsNew = copy.copy(labels)
            labelPropertyNew = copy.copy(labelProperty)
            del (labelsNew[bestFeat])  # 已经选择的特征不再参与分类
            del (labelPropertyNew[bestFeat])
            featValues = [example[bestFeat] for example in dataSet]
            uniqueValue = set(featValues)  # 该特征包含的所有值
            for value in uniqueValue:  # 对每个特征值，递归构建树
                subLabels = labelsNew[:]
                subLabelProperty = labelPropertyNew[:]
                myTree[bestFeatLabel][value] = self.createTree_c(
                    self.splitDataSet(dataSet, bestFeat, value), subLabels,
                    subLabelProperty)
        else:  # 对连续的特征，不删除该特征，分别构建左子树和右子树
            bestFeatLabel = labels[bestFeat] + '<' + str(bestPartValue)
            myTree = {bestFeatLabel: {}}
            subLabels = labels[:]
            subLabelProperty = labelProperty[:]
            # 构建左子树
            valueLeft = 'Yes'
            myTree[bestFeatLabel][valueLeft] = self.createTree_c(
                self.splitDataSet_c(dataSet, bestFeat, bestPartValue, 'L'), subLabels,
                subLabelProperty)
            # 构建右子树
            valueRight = 'No'
            myTree[bestFeatLabel][valueRight] = self.createTree_c(
                self.splitDataSet_c(dataSet, bestFeat, bestPartValue, 'R'), subLabels,
                subLabelProperty)
        return myTree



# ID3决策树
class ID3(DecisionTree):
    def __init__(self,dataSet,algor='ID3',Thresh=(0.0001,8),threshVal=0.5):
        super(ID3,self).__init__(algor,dataSet)
        self.algor=algor
        self.Thresh=Thresh
        self.threshVal=threshVal

# C4.5决策树
class C4_5(DecisionTree):
   def __init__(self,dataSet,algor='C4_5',Thresh=(0.0001,8),threshVal=0.5):
        super(C4_5,self).__init__(algor,dataSet)
        self.algor=algor
        self.Thresh=Thresh
        self.threshVal=threshVal
if __name__ == '__main__':
    dataSet = pd.read_csv('diabetes.csv')

    # 缺失值处理
    col = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    dataSet.head()
    dataSet_copy = dataSet.copy(deep=True)
    dataSet_copy[['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']] = dataSet_copy[
        ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']].replace(0, np.NaN)
    for column in col:  # 平均值填充
        mean_val = dataSet_copy[column].mean()
        dataSet_copy[column].fillna(mean_val, inplace=True)
    #将pandas文件转化为列表进行处理
    ft = dataSet.keys()
    # print(len(ft))
    # print(ft)
    l = dataSet.values.tolist()
    l.insert(0, ft.values.tolist())
    # print(l)
    dataSet = l
    #输出
    Tree=ID3(dataSet)
    print(Tree.createtree(dataSet,['Pregnancies','Glucose','BloodPressure','SkinThickness','Insulin',
                                   'BMI','DiabetesPedigreeFunction','Age']))
    Tree=C4_5(dataSet)
    # print(Tree.createTree_c(dataSet,['Pregnancies','Glucose','BloodPressure','SkinThickness','Insulin',
                                  'BMI','DiabetesPedigreeFunction','Age'],[1,1,1,1,1,1,1,1]))