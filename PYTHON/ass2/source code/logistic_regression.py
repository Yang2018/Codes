from math import exp
from sklearn.preprocessing import StandardScaler
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score


# 基于梯度下降的逻辑回归实现
# 梯度下降
def coefficients_sgd(train, l_rate, n_epoch):
    """
    train 是测试数据，不包含target
    l_rate 是学习率
    n_epoch 是迭代次数
    """
    coef = [0.0 for i in range(len(train[0]))]  # 初始化参数
    for epoch in range(n_epoch):
        for row in train:
            yhat = predict(row, coef)
            error = row[-1] - yhat
            coef[0] = coef[0] + l_rate * error * yhat * (1.0 - yhat)
            for i in range(len(row)-1):
                coef[i + 1] = coef[i + 1] + l_rate * \
                    error * yhat * (1.0 - yhat) * row[i]
    return coef


def predict(row, coefficients):
    yhat = coefficients[0]
    for i in range(len(row)-1):
        yhat += coefficients[i + 1] * row[i]
    return 1.0 / (1.0 + exp(-yhat))


data1 = pd.read_csv(r"..\datasets\D4_wdbc\wdbc.data", header=None)
data1[0] = 0
data1[1] = (data1[1] != 'B').astype(int)
y = data1[1]
data1.drop(1, axis=1, inplace=True)

scaler = StandardScaler()
data1 = scaler.fit_transform(data1)
X_train, X_test, y_train, y_test = train_test_split(
    data1, y, test_size=0.2, random_state=10)
coefs = coefficients_sgd(X_train, 0.1, 100)
result = []
for row in X_test:
    if predict(row, coefs) > 0.5:
        result.append(1)
    else:
        result.append(0)

print(accuracy_score(y_test, result))
print(f1_score(y_test, result))
print(recall_score(y_test, result))
print(precision_score(y_test, result))

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

plt.xlabel('True label')
plt.ylabel('Predocted label')
plt.savefig('./figure/mc1.svg', format='svg')