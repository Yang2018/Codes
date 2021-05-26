import pandas as pd
import numpy as np
from FCBF_module import FCBF, FCBFK, FCBFiP, get_i
from sklearn.datasets import load_digits
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
import time
from sklearn.model_selection import GridSearchCV

tmp = pd.read_csv(r'C:\Users\Yang\Documents\Tencent Files\1364131228\FileRecv\train_split_Depression_AVEC2017.csv')
tt = tmp.dropna()
y_train = tt["PHQ8_Binary"]
sex1 = tt['Gender']
tmp = pd.read_csv(r'C:\Users\Yang\Documents\Tencent Files\1364131228\FileRecv\full_test_split.csv')
tt = tmp.dropna()
sex2 = tt['Gender']
y_test = tt["PHQ_Binary"]

x_test = pd.read_csv(r'C:\Users\Yang\test_target.csv', index_col=0)
x_train = pd.read_csv(r'C:\Users\Yang\train_target.csv', index_col=0)

x_train = np.array(x_train)
y_train = np.array(y_train)

ans = []

fcbf = FCBF()
fcbf.fit(x_train[:,0:100], y_train)
n = x_train[:, fcbf.idx_sel]
print(fcbf.idx_sel)
ans.append(n)
