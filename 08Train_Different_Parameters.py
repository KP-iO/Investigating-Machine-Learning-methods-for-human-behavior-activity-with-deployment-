import warnings

import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
from sklearn.tree import *
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import  GridSearchCV
from sklearn_evaluation import plot
from pandas import ExcelWriter

warnings.filterwarnings('ignore')
filename = 'Dataset/TrainingCleaned.csv'
df = pd.read_csv(filename)
df.drop(['subject'], axis=1)
best = []


# print(df.columns)


# print(df.Activity.unique())


def decision_trees_model(x_train, y_train):
    # creating Decision Tree Classifier method
    decision_trees = DecisionTreeClassifier()
    # Create parameters to search
    max_depth = [1, 10, 50, 100, 1000]
    min_samples_leaf = [1, 25, 50, 100, 300]
    # create GridSearch Dictionary
    grid = dict(
        max_depth=max_depth, min_samples_leaf=min_samples_leaf
    )
    grid_search = GridSearchCV(estimator=decision_trees, param_grid=grid, error_score=0, cv = 5)
    grid_results = grid_search.fit(x_train, y_train)
    # Plot GridSearch Results
    ax1 = plot.grid_search(grid_results.cv_results_, change='max_depth', kind='bar')
    ax1.set_title('When maximum depth and minimum samples leaf')
    plt.show()

    # Create CSV file to hold table of results.
    df1 = pd.concat([pd.DataFrame(grid_results.cv_results_["params"]),
                     pd.DataFrame(grid_results.cv_results_["mean_test_score"], columns=["Accuracy"])], axis=1)
    print("DataFrame for Decision Tree")
    print(df1.values)
    writer = ExcelWriter('Decision Finetuned.xlsx')
    df1.to_excel(writer, 'sheet2')
    writer.save()

    return grid_results


def knn_model(x_train, y_train):
    # https://medium.datadriveninvestor.com/k-nearest-neighbors-in-python-hyperparameters-tuning-716734bc557f
    # creating linear regression object
    KNN = KNeighborsClassifier()
    n_neighbors = [1, 10, 50, 100, 500]
    metric = ['manhattan', 'euclidean']
    grid = dict(metric=metric, n_neighbors=n_neighbors)
    grid_search = GridSearchCV(estimator=KNN, param_grid=grid, error_score=0, cv = 5)
    grid_results = grid_search.fit(x_train, y_train.ravel())
    # If p = 1
    ax = plot.grid_search(grid_results.cv_results_, change='n_neighbors', kind='bar')
    ax.set_title('When distance metric and number of neighbours is changed')
    plt.show()

    # Create Table
    df1 = pd.concat([pd.DataFrame(grid_results.cv_results_["params"]),
                     pd.DataFrame(grid_results.cv_results_["mean_test_score"], columns=["Accuracy"])], axis=1)
    print("DataFrame for KNN Model")
    print(df1.values)
    writer = ExcelWriter('KNN Fine Tuned.xlsx')
    df1.to_excel(writer, 'sheet3')
    writer.save()

    return grid_results


def logistic_model(x_train, y_train):
    # creating linear regression object
    logistic = LogisticRegression()
    c_values = [0.01, 0.1, 1, 10, 100, 1000]
    solver = ['sag', 'saga']
    # Create grid to exhausive ssearch hyperparameters
    grid = dict(C=c_values, solver=solver)
    grid_search = GridSearchCV(estimator=logistic, param_grid=grid, error_score=0, cv = 5)
    grid_results = grid_search.fit(x_train, y_train.ravel())
    # If we use Newton
    ax = plot.grid_search(grid_results.cv_results_, change='C', kind='bar')
    ax.set_title("When regularization parameter and solver changes")
    plt.show()

    df1 = pd.concat([pd.DataFrame(grid_results.cv_results_["params"]),
                     pd.DataFrame(grid_results.cv_results_["mean_test_score"], columns=["Accuracy"])], axis=1)
    print("DataFrame for Logistic Model")
    print(df1.values)
    writer = ExcelWriter('Logistic Fine Tuned.xlsx')
    df1.to_excel(writer, 'sheet4')
    writer.save()

    return grid_results


def svm_model(x_train, y_train):
    # creating linear regression object
    svm = SVC()
    c_values = [0.01, 0.1, 1, 10, 100, 1000]
    gamma = [0.01, 0.1, 1, 10, 100, 1000]
    grid = dict(C=c_values, gamma=gamma)
    grid_search = GridSearchCV(estimator=svm, param_grid=grid, scoring='accuracy', error_score=0, cv = 5)
    grid_results = grid_search.fit(x_train, y_train.ravel())
    # if gamma is 100
    ax = plot.grid_search(grid_results.cv_results_, change='C', kind='bar')
    ax.set_title("When margin constant , gamma changes")
    plt.show()

    df1 = pd.concat([pd.DataFrame(grid_results.cv_results_["params"]),
                     pd.DataFrame(grid_results.cv_results_["mean_test_score"], columns=["Accuracy"])], axis=1)
    print("DataFrame for SVM Model")
    print(df1.values)
    writer = ExcelWriter('SVM FineTuned.xlsx')
    df1.to_excel(writer, 'sheet5')
    writer.save()

    return grid_results


def randomforest_model(x_train, y_train):
    # creating linear regression object
    random = RandomForestClassifier()
    n_estimator = [10, 20, 50, 100, 500]
    max_features = [1, 25, 50, 300, 500, 1000]
    grid = dict(n_estimators=n_estimator, max_features=max_features)
    grid_search = GridSearchCV(estimator=random, param_grid=grid, scoring='accuracy', error_score=0, cv = 5)
    grid_results = grid_search.fit(x_train, y_train.ravel())
    ax = plot.grid_search(grid_results.cv_results_, change='n_estimators', kind='bar')
    ax.set_title("When number of trees and maximum features changes")
    plt.show()

    df1 = pd.concat([pd.DataFrame(grid_results.cv_results_["params"]),
                     pd.DataFrame(grid_results.cv_results_["mean_test_score"], columns=["Accuracy"])], axis=1)
    print("DataFrame for Random Forest")
    print(df1.values)
    writer = ExcelWriter('Random Forest.xlsx')
    df1.to_excel(writer, 'sheet6')
    writer.save()
    return grid_results


def nn_model(x_train, y_train):
    # creating linear regression object
    nn = MLPClassifier()
    hidden_layer_sizes = [25, 50, 100, 500, 1000]
    solver = ['adam', 'sgd']
    grid = dict(hidden_layer_sizes=hidden_layer_sizes, solver=solver)
    grid_search = GridSearchCV(estimator=nn, param_grid=grid, scoring='accuracy', error_score=0, cv = 5)
    grid_results = grid_search.fit(x_train, y_train.ravel())
    ax = plot.grid_search(grid_results.cv_results_, change='hidden_layer_sizes', kind='bar')
    ax.set_title("When number of layers and solver")
    plt.show()

    df1 = pd.concat([pd.DataFrame(grid_results.cv_results_["params"]),
                     pd.DataFrame(grid_results.cv_results_["mean_test_score"], columns=["Accuracy"])], axis=1)
    print("DataFrame for NN")
    print(df1.values)
    writer = pd.ExcelWriter('NN.xlsx')
    df1.to_excel(writer, 'sheet7')
    writer.save()

    return grid_results


def build_and_train_model(df, target_name, reg_fn):
    global best

    X = df.drop(target_name, axis=1)
    Y = df[target_name]
    Y = Y.values.reshape(-1, 1)

    model = reg_fn(X, Y)

    # y_pred = model.predict(x_test)

    print("Best: %f using %s" % (model.best_score_, model.best_params_))

    means = model.cv_results_['mean_test_score']
    stds = model.cv_results_['std_test_score']
    params = model.cv_results_['params']
    for mean, stdev, param in zip(means, stds, params):
        print("%f (%f) with: %r" % (mean, stdev, param))
    best = np.append(best, model.best_score_)



#
# Print out test and training scores


print('Decision Tree')
decision_tree = build_and_train_model(df, 'Activity', decision_trees_model)
print("best: ", best)

print('K- Nearest Neighbour')
KNN = build_and_train_model(df, 'Activity', knn_model)
print("best: ", best)

print('Logistic Regression')
logistic = build_and_train_model(df, 'Activity', logistic_model)
print("best: ", best)

# NEED TO CHECK

print('Support Vector Machine')
svm = build_and_train_model(df, 'Activity', svm_model)
print("best: ", best)

print('Random Forest')
random = build_and_train_model(df, 'Activity', randomforest_model)
print("best: ", best)

print('Neural Network')
nn = build_and_train_model(df, 'Activity', nn_model)
print("best: ", best)

algorithms = ['DT', 'KNN ', 'LogR', 'SVM', 'RF', 'NN']

print(best)
# fig = plt.figure(figsize=(10, 5))
#
# # creating the bar plot
# plt.bar(algorithms, best)
#
# plt.xlabel("Algorithms")
# plt.ylabel("Accuracy")
# plt.title("Best Paremeters per Algorithm")
# plt.show()
#
fig, ax = plt.subplots()

ax.set_ylabel('Accuracy %')
ax.set_title('Best Scores Fine Tuning Model using GridSearchCV')
# ax.set_xticks(x)
# ax.set_xticklabels(algorithms)
ax.plot(algorithms, best, 'o-', color="r",
        label="Best Scores")

ax.legend(loc="best")
ax.legend()
plt.xticks(rotation=90)
plt.figure(figsize=(3, 4))
plt.show()
