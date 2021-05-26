from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
data3 = pd.read_csv(r"..\datasets\D3_wine\winequality-red.csv", sep=';')
np.unique(data3['quality'], return_counts=True)
newdata = data3.where(data3['quality'] == 6).dropna()

d1 = data3.where(data3['quality'] == 5).dropna()
newdata = pd.concat([newdata, d1], axis=0)
data3 = newdata
mapping = {5: 0, 6: 1}
data3 = data3.replace({'quality': mapping})

X_train, X_test, y_train, y_test = train_test_split(
    data3.iloc[:, :-1], data3.iloc[:, -1], test_size=0.2, random_state=0)
train = pd.concat([X_train, y_train], axis=1)
test = pd.concat([X_test, y_test], axis=1)


print("SVC:")
param_range = [0.01, 0.1, 1, 10.0, 100.0]
param_grid = [{'C': param_range, 'kernel': ['linear']},
              {'C': param_range, 'kernel': ['rbf'], 'gamma': param_range},
              {'C': param_range, 'kernel': ['sigmoid'], 'gamma': param_range}]
gs = GridSearchCV(estimator=SVC(random_state=1),
                  param_grid=param_grid,
                  scoring='f1',

                  cv=10)
gs = gs.fit(X_train, y_train)
print(gs)
print(gs.best_score_)
print(gs.best_params_)
y_pre = gs.best_estimator_.predict(X_test)
print(confusion_matrix(y_test, y_pre))
print(f1_score(y_test, y_pre))
accuracy_score(y_test, y_pre)

result = y_pre
acc = accuracy_score(y_test, result)
f1mi = f1_score(y_test, result, average='micro')
f1ma = f1_score(y_test, result, average='macro')
rs = recall_score(y_test, result)
ps = precision_score(y_test, result)


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
plt.savefig('./figure/mc5.svg', format='svg')


plt.figure(figsize=(12, 6))
# r2值图， X轴是参数取值
lb = ['recall_score', 'accuracy_score',
      'precision_score', 'f1_score_macro', 'f1_score_micro']
plt.ylabel("value(0~1)")
for x in range(5):
    plt.bar([x], [rs, acc, ps, f1ma, f1mi][x], width=0.3, label=lb[x])

plt.legend(loc='upper left')
plt.xticks([])
plt.ylim((0.3, 1))
plt.savefig('./figure/hl5.svg', format='svg')