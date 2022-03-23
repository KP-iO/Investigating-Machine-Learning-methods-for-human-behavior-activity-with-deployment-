import warnings

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.tree import *
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
# Loading Data
warnings.filterwarnings('ignore')
trainSet = 'Dataset/TrainingCleaned.csv'
testSet = 'Dataset/TestCleaned.csv'
df = pd.read_csv(trainSet)
df1 = pd.read_csv(testSet)
df.drop(['subject'], axis=1)
df1.drop(['subject'], axis=1)


trainingScores = []
testingScores = []
f1Scores = []
precisionScores = []
recallScores = []




#region  Create fuctions for each model

def decision_trees_model(x_train, y_train):
    # creating linear regression object
    decision_trees = DecisionTreeClassifier(max_depth =1000, min_samples_leaf = 25)
    decision_trees.fit(x_train, y_train)
    return decision_trees

def knn_model(x_train, y_train):
    # creating linear regression object
    KNN = KNeighborsClassifier(metric = 'manhattan', n_neighbors= 10)
    KNN.fit(x_train, y_train.ravel())
    return KNN

def logistic_model(x_train, y_train):
    # creating linear regression object
    logistic = LogisticRegression( C =1 , solver= 'sag')
    logistic.fit(x_train, y_train.ravel())
    return logistic

def svm_model(x_train, y_train):
    # creating linear regression object
    svm = SVC( C=1, gamma= 0.01)
    svm.fit(x_train, y_train.ravel())
    return svm

def randomforest_model(x_train, y_train):
    # creating linear regression object
    random = RandomForestClassifier(max_features= 25 , n_estimators= 50)
    random.fit(x_train, y_train.ravel())
    return random

def nn_model(x_train, y_train):
    # creating linear regression object
    nn = MLPClassifier(hidden_layer_sizes= 25, solver= 'adam')
    nn.fit(x_train, y_train.ravel())
    return nn




def build_and_train_model(df, target_name, reg_fn):
    # create global variables
    global trainingScores
    global testingScores
    global f1Scores
    global precisionScores
    global recallScores
    # Create X and Y dataframe
    # TrainSet
    x_train = df.drop(target_name, axis=1)
    y_train = df[target_name]
    y_train = y_train.values.reshape(-1, 1)
    # TestSet
    x_test = df1.drop(target_name, axis=1)
    y_test = df1[target_name]
    y_test = y_test.values.reshape(-1, 1)

    # Train models, get their score and then print the scores
    model = reg_fn(x_train, y_train)
    score = model.score(x_train, y_train)
    print("Training Score :", score)
    # Predict what the label will be bases on X test data
    y_pred = model.predict(x_test)
    # Calculate Accuracy, F1, precision and recall scores
    accuracy= accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test,y_pred, average='macro')
    precision = precision_score(y_test, y_pred, average='macro')
    recall = recall_score(y_test,y_pred, average='macro')
    print("Testing Score: ", accuracy)
    print("F1 Score: ", f1)
    print("Precision Score: ", precision)
    print("recall: ", recall)

    # append them to array created above
    testingScores = np.append(testingScores, accuracy)
    trainingScores = np.append(trainingScores,score)
    f1Scores = np.append(f1Scores, f1)
    precisionScores = np.append(precisionScores, precision)
    recallScores = np.append(recallScores, recall)

    # get parameters to look at when parameter tuning
    print(model.get_params())

# Print out model with its metrics based on it classifying activities
print('Decision Tree')
decision_tree = build_and_train_model(df, 'Activity', decision_trees_model)

print('K- Nearest Neighbour')
KNN = build_and_train_model(df, 'Activity', knn_model )

print('Logistic Regression')
logistic = build_and_train_model(df, 'Activity', logistic_model)

print('Support Vector Machine')
svm = build_and_train_model(df, 'Activity', svm_model)

print('Random Forest')
random = build_and_train_model(df, 'Activity', randomforest_model)

print('Neural Network')
nn = build_and_train_model(df, 'Activity', nn_model)

# Creating x Axis
algorithms = [ 'DT', 'KNN ', 'LogR', 'SVM', 'RF', 'NN']

# Done with help using source[1]
fig, ax = plt.subplots()
ax.set_ylabel('Accuracy %')
ax.set_title('Model Evaluation on Testset using best hyperparameters')
ax.plot(algorithms, trainingScores, 'o-', color="r",
             label="Training score")
ax.plot(algorithms, testingScores, 'o-', color="g",
             label="Testing score")
ax.legend(loc="best")
ax.legend()
plt.xticks(rotation=90)
plt.figure(figsize=(3,4))
plt.show()