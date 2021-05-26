import pandas as pd
from ttt.FCBF import fcbf
import numpy as np


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
F, su = fcbf(x_train, y_train)
