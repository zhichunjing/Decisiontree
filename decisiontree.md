## 数据处理

- diabetes是关于糖尿病预测的数据集，里面有9列768行，特征值有一共有九个（最后还有一个结果判断），分别是怀孕次数，葡萄糖含量，血压，皮肤厚度，胰岛素含量，身高体重指数，糖尿病遗传函数，年龄。

  ```
  RangeIndex: 768 entries, 0 to 767
  Data columns (total 9 columns)
Index(['Pregnancies', 'Glucose','BloodPressure','SkinThickness','Insulin',
  'BMI','DiabetesPedigreeFunction', 'Age', 'Outcome'],
        dtype='object')
  特征（怀孕次数，血糖，血压，皮脂厚度，胰岛素，BMI身体质量指数，糖尿病遗传函数，年龄，结果）
  Outcome
  0    500
  1    268
  dtype: int64  
  “结果”是我们将要预测的特征，0意味着未患糖尿病，1意味着患有糖尿病。在768个数据点中，500个被标记为0,268个标记为1。
  ```
  

- 数据集中存在缺失值，均采用平均值填充处理

  ```
  Glucose：5    血糖
  BloodPressure：35    血压
  SkinThickness： 227    皮脂厚度
  Insulin： 374    胰岛素
  BMI：11    身体质量指数
  
  ```

- 读入的是pandas文件，后续采用list的处理方式，把读入的数据集转为list形式

  ```
  l = dataSet.values.tolist()
  l.insert(0, ft.values.tolist())
  dataSet = l
  ```

## 构建决策树

#### 1.决策树原理

- 根据已知若干条件对事情作出判断。从根节点到叶子节点，将不同特征不断划分，最后将类别输出

#### 2.用ID3算法构造决策树

- ID3算法将信息增益做贪心算法来划分算法，总是挑选是的信息增益最大的特征来划分数据，使得数据更加有序。

- ##### ***信息增益***

  -  熵：熵是信息的期望，一般表示信息的混乱、无序程度。

    信息的定义：$l(x_i)=-{log_2{p(x_i)}}$

    信息的期望，即熵的定义：$H=-\sum_{i=1}^{n}p(x_i){log_2{p(x_i)}}$

  - 信息增益指的是熵减小的量，数据集变得有序了多少

    ```python
        def calcShannonEnt(self,dataSet):
           numEntries = len(dataSet)#样本数
           labelCounts = {}#创建一个字典
           for featVec in dataSet:#遍历每个样本
              currentLabel=featVec[-1]#当前样本的类别：最后一列
              if currentLabel not in labelCounts.keys():#生成类别字典，判断labelCount有没有currentLabel键
                labelCounts[currentLabel]=0  # 如果没有，则把这个值作为这个函数的键，并将其值初始化为0                       labelCounts[currentLabel]+=1 #将这个键的值加1
           shannonEnt=0.0
           for key in labelCounts:#计算信息熵
              prob = float(labelCounts[key])/numEntries
              shannonEnt = shannonEnt-prob *math.log(prob,2)
           return shannonEnt
    ```

  

- ##### *划分数据集*

  构造树的过程中，要将数据进行划分，为了方便，写一个子函数专门做数据划分。

  ```python
      def splitDataSet(self,dataSet,axis,value): #dataSet是二维数组，axis是要查找的位置，value是要查找的对象。
          retDataSet=[]  #创建一个空的列表
          featVec=[]  #遍历dataSet中的每一个小列表
          #离散值
          for featVec in dataSet:  #判断列表中axis对应的值是否等于value
              if featVec[axis]==value:  #去掉featVec中axis所对应的值
                  reducedFeatVec=featVec[:axis]
                  reducedFeatVec.extend(featVec[axis+1:])
                  retDataSet.append(reducedFeatVec)  #这个是为了把变化后的每个小列表添加到retDataSet中
                  # print(retDataSet)
          return retDataSet
  ```

  

- ##### *选择最佳划分特征值*

  遍历每一个特征，尝试使用每一个特征划分数据集，并计算对应的信息增益，选择最大的那一个特征来划分数据。对应的特征有多少个值就将数据集划分为几个子集。

  ```python
      def chooseBestFeatureToSplit(self,dataSet1):
          numFeatures=len(dataSet1[0])-1  #求属性的个数减1
          baseEntropy=self.calcShannonEnt(dataSet1)  #求dataSet的期望值
          bestInfoGain=0.0  #赋初始值
          bestFeature=-1
          for i in range(numFeatures):
              featList=[example[i] for example in dataSet1]  
              #把dataSet中的数据存入到example中，依次读取example中的小列表。
              del(featList[0]) #去掉列表属性值，只留下数据 
              uniqueVals=set(featList)  #去重处理
              newEntropy=0.0
              for value in uniqueVals:  #对uniqueVals里面的各个数进行遍历
                  subDataSet=self.splitDataSet(dataSet1,i,value)
                  prob=len(subDataSet)/float(len(dataSet1))
                  newEntropy+=prob*self.calcShannonEnt(subDataSet)
              infoGain=baseEntropy-newEntropy  #计算信息增益
              if(infoGain>bestInfoGain):
                  bestInfoGain=infoGain   #将最大的熵赋值bestInfoGain
                  bestFeature=i  #并将序号保存
          return bestFeature  #返回序号
  ```

  

- ##### *建树*

  - 划分数据集可以使得数据局部整体上最有序，划分得到若干个子数据集中依然可以继续划分，使用同样的方法使得子数据集最有序，依次递归进行，直到最底层子数据集中只有一个类别

  ```python
      def createtree(self,dataSet1,labels):
          classList=[example[-1] for example in dataSet1]
          if classList.count(classList[0])==len(classList):
              return classList[0]  #如果只剩一个类别，返回
          if len(dataSet1[0])==1:
              return self.majorityCnt(classList)  #如果所有特征都被遍历完了，调用多数表决函数
          bestFeat=self.chooseBestFeatureToSplit(dataSet1)   #求出最佳方案输出其列序号
          # print(bestFeat)
          bestFeatLabel=labels[bestFeat]   #特征对应的标签
          myTree={bestFeatLabel: {}}
          # print(bestFeat)
          del (labels[bestFeat]) #已经选择的特征不再参与分类
          featValues=[example[bestFeat] for example in dataSet1]
          uniqueValue=set(featValues) #该属性所有可能取值，也就是节点的分支
          # print(uniqueValue)
          for value in uniqueValue:  #对每个分支，递归构建树
              subLabels=labels[:]
              myTree[bestFeatLabel][value]=self.createtree(
                  self.splitDataSet(dataSet1,bestFeat, value), subLabels)
          return myTree
  ```

  - 当数据集只剩下一列（最后一列，也就是类别），但是无法完全分类，就返回剩下的数据集中出现最多的那个类别（多数表决）

    ```python
        def majorityCnt(dataSet,classList):
            classCount = {} #创建空字典
            for vote in classList:
                if vote not in classCount.keys():
                     #判断classCount中是否含有vote所对应的值，如果没有，将其值赋值为0；
                    classCount[vote] = 0
                    classCount[vote] += 1
            sortedClassCount = sorted(classCount.iteritems(), #排序
                                      key=pyoperator.itemgetter(1), reverse=True)
            return sortedClassCount[0][0] #返回占有率最大的那个
    ```

    

