from os import path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# decisiontree
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.model_selection import train_test_split

import custom.classification.decisiontree.DecisionTree
from constants import DATA_PATH
from custom.classification.decisiontree.BinaryTreeToDotFormatConverter import BinaryTreeToDotFormatConverter

DATASET = path.join(DATA_PATH, "car.csv")

df = pd.read_csv(DATASET)

# Prints a scatter plt sepal length vs petal length
# y = df.iloc[0:100, 4].values
# X = df.iloc[0:100, [0, 2]].values
# plt.scatter(X[:50, 0], X[:50, 1], color='red', marker='o', label='setosa')
# plt.scatter(X[50:100, 0], X[50:100, 1], color='blue', marker='x', label='versicolor')
# plt.xlabel('sepal length')
# plt.ylabel('petal length')
# plt.legend(loc='upper left')
# plt.show()

# decision tree
df = pd.read_csv(DATASET)
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
# tree = DecisionTreeClassifier(criterion='entropy', max_depth=3, random_state=0)
# tree.fit(X_train, y_train)
# export_graphviz(tree, out_file='tree.dot', feature_names=['petal length', 'petal width'])

custom_tree = custom.classification.decisiontree.DecisionTree.DecisionTree(criterion_name='entropy')
# custom_tree.fit(training_x=X_train, training_y=y_train)
#BinaryTreeToDotFormatConverter.dump_tree(custom_tree._tree)

# KFOLD VALIDATION
from sklearn.model_selection import KFold
kf = KFold(n_splits=10)
for train_index, test_index in kf.split(X):
    X_train, X_test = X.take(train_index, axis=0), X.take(test_index, axis=0)
    y_train, y_test = y.take(train_index, axis=0), y.take(test_index, axis=0)

    custom_tree.fit(X_train, y_train)

    correct = 0
    for idx, instance in enumerate(X_test):
        predicted = custom_tree.predict(instance)
        if predicted == y_test[idx]:
            correct += 1

    print(f'Accuracy: {correct/len(y_test) * 100} %')

BinaryTreeToDotFormatConverter.dump_tree(custom_tree._tree)



