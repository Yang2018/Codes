# Backprop on the Seeds Dataset
from random import seed
from random import random
from math import exp
from random import randrange
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score


def str_column_to_int(dataset, column):
    class_values = [row[column] for row in dataset]
    unique = set(class_values)
    lookup = dict()
    for i, value in enumerate(unique):
        lookup[value] = i
    for row in dataset:
        row[column] = lookup[row[column]]
    return lookup

# Find the min and max values for each column


def dataset_minmax(dataset):
    minmax = list()
    stats = [[min(column), max(column)] for column in zip(*dataset)]
    return stats

# Rescale dataset columns to the range 0-1


def normalize_dataset(dataset, minmax):
    for row in dataset:
        for i in range(len(row)-1):
            row[i] = (row[i] - minmax[i][0]) / (minmax[i][1] - minmax[i][0])

# Split a dataset into k folds


def cross_validation_split(dataset, n_folds):
    dataset_split = list()
    dataset_copy = list(dataset)
    fold_size = int(len(dataset) / n_folds)
    for i in range(n_folds):
        fold = list()
        while len(fold) < fold_size:
            index = randrange(len(dataset_copy))
            fold.append(dataset_copy.pop(index))
        dataset_split.append(fold)
    return dataset_split

# Calculate accuracy percentage


def accuracy_metric(actual, predicted):
    correct = 0
    for i in range(len(actual)):
        if actual[i] == predicted[i]:
            correct += 1
    return correct / float(len(actual)) * 100.0

# Evaluate an algorithm using a cross validation split


def evaluate_algorithm(dataset, algorithm, n_folds, *args):
    folds = cross_validation_split(dataset, n_folds)
    scores = list()
    for fold in folds:
        train_set = list(folds)
        train_set.remove(fold)
        train_set = sum(train_set, [])
        test_set = list()
        for row in fold:
            row_copy = list(row)
            test_set.append(row_copy)
            row_copy[-1] = None
        predicted = algorithm(train_set, test_set, *args)
        actual = [row[-1] for row in fold]
        accuracy = accuracy_metric(actual, predicted)
        scores.append(accuracy)
    return scores

# Calculate neuron activation for an input


def activate(weights, inputs):
    activation = weights[-1]
    for i in range(len(weights)-1):
        activation += weights[i] * inputs[i]
    return activation

# Transfer neuron activation


def transfer(activation):
    return 1.0 / (1.0 + exp(-activation))

# Forward propagate input to a network output


def forward_propagate(network, row):
    inputs = row
    for layer in network:
        new_inputs = []
        for neuron in layer:
            activation = activate(neuron['weights'], inputs)
            neuron['output'] = transfer(activation)
            new_inputs.append(neuron['output'])
        inputs = new_inputs
    return inputs

# Calculate the derivative of an neuron output


def transfer_derivative(output):
    return output * (1.0 - output)

# Backpropagate error and store in neurons


def backward_propagate_error(network, expected):
    for i in reversed(range(len(network))):
        layer = network[i]
        errors = list()
        if i != len(network)-1:
            for j in range(len(layer)):
                error = 0.0
                for neuron in network[i + 1]:
                    error += (neuron['weights'][j] * neuron['delta'])
                errors.append(error)
        else:
            for j in range(len(layer)):
                neuron = layer[j]
                errors.append(expected[j] - neuron['output'])
        for j in range(len(layer)):
            neuron = layer[j]
            neuron['delta'] = errors[j] * transfer_derivative(neuron['output'])

# Update network weights with error


def update_weights(network, row, l_rate):
    for i in range(len(network)):
        inputs = row[:-1]
        if i != 0:
            inputs = [neuron['output'] for neuron in network[i - 1]]
        for neuron in network[i]:
            for j in range(len(inputs)):
                neuron['weights'][j] += l_rate * neuron['delta'] * inputs[j]
            neuron['weights'][-1] += l_rate * neuron['delta']

# Train a network for a fixed number of epochs


def train_network(network, train, l_rate, n_epoch, n_outputs):
    for epoch in range(n_epoch):
        for row in train:
            outputs = forward_propagate(network, row)
            expected = [0 for i in range(n_outputs)]
            expected[row[-1]] = 1
            backward_propagate_error(network, expected)
            update_weights(network, row, l_rate)

# Initialize a network


def initialize_network(n_inputs, n_hidden, n_outputs):
    network = list()
    hidden_layer = [{'weights': [random() for i in range(n_inputs + 1)]}
                    for i in range(n_hidden)]
    network.append(hidden_layer)
    output_layer = [{'weights': [random() for i in range(n_hidden + 1)]}
                    for i in range(n_outputs)]
    network.append(output_layer)
    return network

# Make a prediction with a network


def predict(network, row):
    outputs = forward_propagate(network, row)
    return outputs.index(max(outputs))

# Backpropagation Algorithm With Stochastic Gradient Descent


def back_propagation(train, test, l_rate, n_epoch, n_hidden):
    n_inputs = len(train[0]) - 1
    n_outputs = len(set([row[-1] for row in train]))
    network = initialize_network(n_inputs, n_hidden, n_outputs)
    train_network(network, train, l_rate, n_epoch, n_outputs)
    predictions = list()
    for row in test:
        prediction = predict(network, row)
        predictions.append(prediction)
    return(predictions)


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

seed(1)
dataset = train.values.tolist()
test1 = test.values.tolist()
str_column_to_int(dataset, len(dataset[0])-1)
str_column_to_int(test1, len(test1[0])-1)
# normalize input variables
minmax = dataset_minmax(dataset)
normalize_dataset(dataset, minmax)

minmax = dataset_minmax(test1)
normalize_dataset(test1, minmax)
# evaluate algorithm
n_folds = 5
l_rate = 0.3
n_epoch = 500
n_hidden = 10
result = back_propagation(dataset, test1, l_rate, n_epoch, n_hidden)

acc = accuracy_score(y_test, result)
f1mi = f1_score(y_test, result, average='micro')
f1ma = f1_score(y_test, result, average='macro')
rs = recall_score(y_test, result)
ps = precision_score(y_test, result)


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
plt.savefig('./figure/mc4.svg', format='svg')

plt.figure(figsize=(12, 6))
# r2值图， X轴是参数取值
lb = ['recall_score', 'accuracy_score',
      'precision_score', 'f1_score_macro', 'f1_score_micro']
plt.ylabel("value(0~1)")
for x in range(5):
    plt.bar([x], [rs, acc, ps, f1ma, f1mi][x], width=0.3, label=lb[x])

plt.legend(loc='upper left')
plt.xticks([])
plt.ylim((0.2, 1))
plt.savefig('./figure/hl4.svg', format='svg')