import warnings
# Import libraries needed
import pandas as pd
import sklearn
from sklearn.metrics import accuracy_score,f1_score,recall_score,precision_score
from matplotlib import pyplot as plt
import numpy as np
from sklearn.tree import *
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score, KFold
warnings.filterwarnings('ignore')
# Load in the data needed
filename = 'Dataset/TrainingCleaned.csv'
df = pd.read_csv(filename)
df.drop(['subject'], axis=1)

# Create empty arrays to store metric values


#region Preparing Data

#endregion


X = df.drop("Activity", axis=1)
Y = df["Activity"]
Y = Y.values.reshape(-1, 1)
#region Parameters for algorith training/ Folds and scoring
num_folds = 10 # K-Value or number of folds is 10

scoring = 'accuracy'
scoring2 = 'precision_macro'
scoring3 = 'recall_macro'
scoring4 ='f1_macro'
#endregion


#region Spot check algorithms/ Create array of all algorithms you'd like to train on
models =[]
models.append(('Decision Tree', DecisionTreeClassifier()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('LR', LogisticRegression()))
models.append(('SV', SVC()))
models.append(('Random Forest', RandomForestClassifier()))
models.append(('Neural Network', MLPClassifier()))



#endregion



#region Evaluate models and print results
results =[]
name = []
accuracyScore = []
f1Scores = []
precisionScores = []
recallScores = []
for name, model in models: # look through all models in model array
    kfold = KFold( n_splits = 5, shuffle=False) #for each iteration define a KFold instance with a set K value on the random state
    # use cross_val_score four times it is a handy way to both train and evaluate data
    cv_results = cross_val_score(model, X, Y, cv=kfold, scoring = scoring)
    accuracyScore = np.append(accuracyScore, cv_results.mean())
    cv_results2 = cross_val_score(model, X, Y, cv=kfold, scoring=scoring2)
    precisionScores = np.append(precisionScores, cv_results2.mean())
    cv_results3 = cross_val_score(model, X, Y, cv=kfold, scoring=scoring3)
    recallScores = np.append(recallScores, cv_results3.mean())
    cv_results4 = cross_val_score(model, X, Y, cv=kfold, scoring=scoring4)
    f1Scores = np.append(f1Scores, cv_results4.mean())

    msg = "%s: Accuracy: %f, Precision: %f, Recall: %f,F1-Score: %f" % (name,cv_results.mean(), cv_results2.mean(), cv_results3.mean(), cv_results4.mean())

    print(msg)







    # Done with help using source[1]


#endregion

algorithms = ['DT', 'KNN ', 'LogR', 'SVM', 'RF', 'NN']
fig, ax = plt.subplots()
ax.set_ylabel('Accuracy %')
ax.set_title('Training models using KFold Cross-Validation')
ax.plot(algorithms, accuracyScore, 'o-', color="g",
            label="Accuracy")
ax.plot(algorithms, precisionScores, 'o-', color="r",
            label="Precision")

ax.plot(algorithms, recallScores, 'o-', color="b",
            label="Recall")
ax.plot(algorithms, f1Scores, 'o-', color="k",
            label="F1-Scoree")
ax.legend(loc="best")
ax.legend()
plt.xticks(rotation=90)
plt.figure(figsize=(3, 4))
plt.show()