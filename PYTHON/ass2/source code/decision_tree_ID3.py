import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Define the calculate entropy function


def calculate_entropy(df_label):
    classes, class_counts = np.unique(df_label, return_counts=True)
    entropy_value = np.sum([(-class_counts[i]/np.sum(class_counts))*np.log2(class_counts[i]/np.sum(class_counts))
                            for i in range(len(classes))])
    return entropy_value
# Define the calculate information gain function


def calculate_information_gain(dataset, feature, label):
    # Calculate the dataset entropy
    dataset_entropy = calculate_entropy(dataset[label])
    values, feat_counts = np.unique(dataset[feature], return_counts=True)
    # print(values, feat_counts)
    # print(dataset.where(dataset[feature]== values[i]).dropna()[label])

    # Calculate the weighted feature entropy                                # Call the calculate_entropy function
    weighted_feature_entropy = np.sum(
        [(feat_counts[i] / np.sum(feat_counts)) * calculate_entropy(dataset.where(dataset[feature]
                                                                                  == values[i]).dropna()[label]) for i
         in range(len(values))])

    feature_info_gain = dataset_entropy - weighted_feature_entropy
    return feature_info_gain


def calculate_information_gain2(dataset, feature, label):
    # Calculate the dataset entropy
    dataset_entropy = calculate_entropy(dataset[label])
    values, feat_counts = np.unique(dataset[feature], return_counts=True)

    wf = []
    for i in range(1, len(feat_counts)):
        ft = []
        ft.append(sum(feat_counts[:i]))
        ft.append(sum(feat_counts[i:]))

        ld = [dataset.where(dataset[feature] == j).dropna()[label]
              for j in values[:i]]
        rd = [dataset.where(dataset[feature] == j).dropna()[label]
              for j in values[i:]]
        ll = []
        rr = []
        for _ in ld:
            ll += _.tolist()
        for _ in rd:
            rr += _.tolist()

        weighted_feature_entropy = 0
        weighted_feature_entropy += (ft[0] /
                                     np.sum(feat_counts)) * calculate_entropy(ll)
        weighted_feature_entropy += (ft[1] /
                                     np.sum(feat_counts)) * calculate_entropy(rr)
        wf.append(weighted_feature_entropy)

    feature_info_gain = dataset_entropy - min(wf)
    return feature_info_gain, np.argmin(wf)


def create_decision_tree(dataset, features, label, parent):
    unique_data = np.unique(dataset[label])

    if len(unique_data) <= 1:
        return unique_data[0]
    elif len(dataset) == 0:
        return unique_data[0][np.argmax(unique_data[1])]
    elif len(features) == 0:
        return parent

    else:
        parent = unique_data[0][np.argmax(unique_data[1])]
        item_values = [calculate_information_gain(
            dataset, feature, label) for feature in features]
        optimum_feature = features[np.argmax(item_values)]

        decision_tree = {optimum_feature: {}}
        for value in np.unique(dataset[optimum_feature]):
            min_data = dataset.where(
                dataset[optimum_feature] == value).dropna()
            min_tree = create_decision_tree(
                min_data, features.drop(optimum_feature), label, parent)
            decision_tree[optimum_feature][value] = min_tree
    return decision_tree


# Define the create decision tree functiond
# decision_tree = {best_attr:{}}
def create_decision_tree2(dataset, features, label, parent):
    # datum = np.unique(df[label], return_counts=True)
    unique_data = np.unique(dataset[label], return_counts=True)

    if len(unique_data[0]) <= 1:
        return unique_data[0][0]
    elif len(dataset) == 0:
        return unique_data[0][np.argmax(unique_data[1])]
    elif len(features) == 0:
        return parent

    else:
        parent = unique_data[0][np.argmax(unique_data[1])]
        item_values = [(calculate_information_gain2(
            dataset, feature, label), feature) for feature in features]

        item_values = sorted(item_values, reverse=True)
        num = item_values[0][0][1]
        optimum_feature = item_values[0][1]
        of = np.unique(dataset[optimum_feature])
        opt_num = (of[num] + of[num + 1]) / 2

        decision_tree = {optimum_feature: {}}
        bigger = '>' + str(opt_num)
        min_data = dataset.where(dataset[optimum_feature] > opt_num).dropna()
        min_tree = create_decision_tree2(
            min_data, features.drop(optimum_feature), label, parent)
        decision_tree[optimum_feature][bigger] = min_tree

        minner = '<' + str(opt_num)
        min_data = dataset.where(dataset[optimum_feature] < opt_num).dropna()
        min_tree = create_decision_tree2(
            min_data, features.drop(optimum_feature), label, parent)
        decision_tree[optimum_feature][minner] = min_tree
    return decision_tree


def predict_Onepoint(point, tree):
    feature = list(tree.keys())[0]
    label = point[feature]
    if eval(str(label) + list(tree[feature].keys())[0]):
        next_tree = tree[feature][list(tree[feature].keys())[0]]
    else:
        next_tree = tree[feature][list(tree[feature].keys())[1]]

    if type(next_tree) != dict:
        return next_tree
    else:
        return predict_Onepoint(point, next_tree)


data2 = pd.read_csv(r"..\datasets\D2_iris\iris.data", header=None)
mapping = {'Iris-setosa': 0, 'Iris-versicolor': 1, 'Iris-virginica': 2}
data2 = data2.replace({4: mapping})
feature = data2.columns
X_train, X_test, y_train, y_test = train_test_split(
    data2.iloc[:, :-1], data2.iloc[:, -1], test_size=0.2, random_state=0)
train = pd.concat([X_train, y_train], axis=1)
dd = create_decision_tree2(train, train.columns[:-1], 4, -1)


result = []
for x in range(len(X_test.index)):
    result.append(predict_Onepoint(X_test.iloc[x], dd))

print("基于信息增益的决策树，得到的决策树字典：")
print(dd)
print("准确率：")
print(accuracy_score(y_test, result))