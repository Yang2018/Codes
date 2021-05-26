import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
# 基于梯度下降的逻辑回归实现


class NaiveBayes:
    def __init__(self):
        self.model = {}  # key 为类别名 val 为字典PClass表示该类的该类，PFeature:{}对应对于各个特征的概率

    def calEntropy(self, y):  # 计算熵
        valRate = y.value_counts().apply(lambda x: x / y.size)  # 频次汇总 得到各个特征对应的概率
        valEntropy = np.inner(valRate, np.log2(valRate)) * -1
        return valEntropy

    def fit(self, xTrain, yTrain=pd.Series()):
        if not yTrain.empty:  # 如果不传，自动选择最后一列作为分类标签
            xTrain = pd.concat([xTrain, yTrain], axis=1)
        self.model = self.buildNaiveBayes(xTrain)
        return self.model

    def buildNaiveBayes(self, xTrain):
        yTrain = xTrain.iloc[:, -1]

        yTrainCounts = yTrain.value_counts()  # 频次汇总 得到各个特征对应的概率

        yTrainCounts = yTrainCounts.apply(lambda x: (
            x + 1) / (yTrain.size + yTrainCounts.size))  # 使用了拉普拉斯平滑
        retModel = {}
        for nameClass, val in yTrainCounts.items():
            retModel[nameClass] = {'PClass': val, 'PFeature': {}}

        propNamesAll = xTrain.columns[:-1]
        allPropByFeature = {}
        for nameFeature in propNamesAll:
            allPropByFeature[nameFeature] = list(
                xTrain[nameFeature].value_counts().index)
        # print(allPropByFeature)
        for nameClass, group in xTrain.groupby(xTrain.columns[-1]):
            for nameFeature in propNamesAll:
                eachClassPFeature = {}
                propDatas = group[nameFeature]
                propClassSummary = propDatas.value_counts()  # 频次汇总 得到各个特征对应的概率
                for propName in allPropByFeature[nameFeature]:
                    if not propClassSummary.get(propName):
                        propClassSummary[propName] = 0  # 如果有属性灭有，那么自动补0
                Ni = len(allPropByFeature[nameFeature])
                propClassSummary = propClassSummary.apply(
                    lambda x: (x + 1) / (propDatas.size + Ni))  # 使用了拉普拉斯平滑
                for nameFeatureProp, valP in propClassSummary.items():
                    eachClassPFeature[nameFeatureProp] = valP
                retModel[nameClass]['PFeature'][nameFeature] = eachClassPFeature

        return retModel

    def predictBySeries(self, data):
        curMaxRate = None
        curClassSelect = None
        for nameClass, infoModel in self.model.items():
            rate = 0
            rate += np.log(infoModel['PClass'])
            PFeature = infoModel['PFeature']

            for nameFeature, val in data.items():
                propsRate = PFeature.get(nameFeature)
                if not propsRate:
                    continue
                rate += np.log(propsRate.get(val, 0))  # 使用log加法避免很小的小数连续乘，接近零
            if curMaxRate == None or rate > curMaxRate:
                curMaxRate = rate
                curClassSelect = nameClass
        return curClassSelect

    def predict(self, data):
        if isinstance(data, pd.Series):
            return self.predictBySeries(data)
        return data.apply(lambda d: self.predictBySeries(d), axis=1)


data2 = pd.read_csv(r"..\datasets\D2_iris\iris.data", header=None)
mapping = {'Iris-setosa': 0, 'Iris-versicolor': 1, 'Iris-virginica': 2}
data2 = data2.replace({4: mapping})
X_train, X_test, y_train, y_test = train_test_split(
    data2.iloc[:, :-1], data2.iloc[:, -1], test_size=0.2, random_state=0)
train = pd.concat([X_train, y_train], axis=1)
test = pd.concat([X_test, y_test], axis=1)
# 下面注释的是另一个数据集
# data1 = pd.read_csv(r"..\datasets\D1_australian\australian.dat",header = None,sep=' ')
# X_train, X_test, y_train, y_test = train_test_split(data1.iloc[:,:-1], data1.iloc[:,-1], test_size = 0.2, random_state=0)
# train = pd.concat([X_train,y_train],axis=1)
# test = pd.concat([X_test,y_test],axis=1)

naiveBayes = NaiveBayes()
treeData = naiveBayes.fit(train)
result = naiveBayes.predict(test)

acc = accuracy_score(y_test, result)
f1mi = f1_score(y_test, result, average='micro')
f1ma = f1_score(y_test, result, average='macro')
rs = recall_score(y_test, result, average='macro')
ps = precision_score(y_test, result, average='macro')


y_pre = result
confus = confusion_matrix(y_test, y_pre)
print(confus)

# 对混淆矩阵画图
plt.matshow(confus, cmap=plt.cm.Blues_r)
plt.colorbar()
# 添加内部数字标签
for x in range(len(confus)):
    for y in range(len(confus)):
        plt.annotate(confus[x, y], xy=(
            x, y), horizontalalignment='center', verticalalignment='center')

# plt.title('confusion matrix')
plt.xlabel('True label')
plt.ylabel('Predocted label')
plt.savefig('./figure/mc3.svg', format='svg')

plt.figure(figsize=(12, 6))
# r2值图， X轴是参数取值
lb = ['recall_score_macro', 'accuracy_score',
      'precision_score_macro', 'f1_score_macro', 'f1_score_micro']
plt.ylabel("value(0~1)")
for x in range(5):
    plt.bar([x], [rs, acc, ps, f1ma, f1mi][x], width=0.3, label=lb[x])

plt.legend(loc='upper left')
plt.xticks([])
plt.ylim((0.5, 1))
plt.savefig('./figure/hl3.svg', format='svg')