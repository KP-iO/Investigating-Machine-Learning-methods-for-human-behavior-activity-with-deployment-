import warnings
# Import libraries needed
import pandas as pd
from sklearn.metrics import accuracy_score,f1_score,recall_score,precision_score
from matplotlib import pyplot as plt
import numpy as np
from sklearn.tree import *
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split

warnings.filterwarnings('ignore')
# Load in the data needed
filename = 'Dataset/TrainingCleaned.csv'
df = pd.read_csv(filename)
df.drop(['subject'], axis=1)

# Create empty arrays to store metric values
trainingScores = []
testingScores = []
f1Scores = []
precisionScores = []
recallScores = []




#region  Create fuctions for each model

def decision_trees_model(x_train, y_train):
    # creating linear regression object
    decision_trees = DecisionTreeClassifier()
    decision_trees.fit(x_train, y_train)
    return decision_trees

def knn_model(x_train, y_train):
    # creating linear regression object
    KNN = KNeighborsClassifier()
    KNN.fit(x_train, y_train.ravel())
    return KNN

def logistic_model(x_train, y_train):
    # creating linear regression object
    logistic = LogisticRegression( )
    logistic.fit(x_train, y_train.ravel())
    return logistic

def svm_model(x_train, y_train):
    # creating linear regression object
    svm = SVC()
    svm.fit(x_train, y_train.ravel())
    return svm

def randomforest_model(x_train, y_train):
    # creating linear regression object
    random = RandomForestClassifier()
    random.fit(x_train, y_train.ravel())
    return random

def nn_model(x_train, y_train):
    # creating linear regression object
    nn = MLPClassifier()
    nn.fit(x_train, y_train.ravel())
    return nn

# endregion
# build and train model
def build_and_train_model(df, target_name, reg_fn):
    # create global variables
    global trainingScores
    global testingScores
    global f1Scores
    global precisionScores
    global recallScores
    # Create X and Y dataframe
    X = df.drop(target_name, axis=1)
    Y = df[target_name]
    Y = Y.values.reshape(-1, 1)
    # use traintest split
    x_train, x_test, y_train, y_test = \
        train_test_split(X, Y, test_size=0.2, random_state=0)
    # Trai models, get their score and then print the scores
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
ax.set_title('Training Model using Train Test Split')
ax.plot(algorithms, trainingScores, 'o-', color="r",
             label="Training score")
ax.plot(algorithms, testingScores, 'o-', color="g",
             label="Testing score")
ax.legend(loc="best")
ax.legend()
plt.xticks(rotation=90)
plt.figure(figsize=(3,4))
plt.show()
# end of source[1] https://scikit-learn.org/stable/auto_examples/model_selection/plot_learning_curve.html




# print(linear_reg)
