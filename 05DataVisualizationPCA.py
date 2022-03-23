# Import pandas which allows the importing of
# data from different sources such as JSON, CSV also allows manipulation of data
import pandas as pd
# Using Standard Scaler normalizes data individually so that it has mean 0 and SD 1
from sklearn.preprocessing import StandardScaler
# this library stored as plt allows us to plot figures easily
from matplotlib import pyplot as plt
# PCA is used to reduce dimensionality
from sklearn.decomposition import PCA
# Numpy allows mathematival operations to be easily done
import numpy as np

# Load in data
filename = 'Dataset/trainsaved.csv'
df = pd.read_csv(filename)

# Removing Activity column from dataset and storing it in variable labels
labels = df.pop('Activity')
no_labels = labels.value_counts()

# scale data by assigning StandardScalar method to variable called scaler and applying it on the data
scaler = StandardScaler()
scaler.fit(df)
scaled_data = scaler.transform(df)
# carry out PCA
pca = PCA(n_components=2)
pca.fit(scaled_data)

# We are looking at where the data for certain activities exist and if they do we are
# getting their indexes and assigning to an array for that activity and storing them as an integer
# But due to the fact as we are looking the ones that do not exist there will come back as nan we
#  are saying we do not want them
# This therefore allows us to get what we want
# WALKING UPSTAIRS
walkingup = labels.index.where(labels == 'WALKING_UPSTAIRS').values
walkingupcleaned = [x for x in walkingup if str(x) != 'nan']
walkingupcleanedINT = np.array(walkingupcleaned).astype(int)

# STANDING
standing = labels.index.where(labels == 'STANDING').values
standingcleaned = [x for x in standing if str(x) != 'nan']
standingcleanedINT = np.array(standingcleaned).astype(int)

# LAYING
laying = labels.index.where(labels == 'LAYING').values
layingcleaned = [x for x in laying if str(x) != 'nan']
layingcleanedINT = np.array(layingcleaned).astype(int)

# SITTING
sitting = labels.index.where(labels == 'SITTING').values
sittingcleaned = [x for x in sitting if str(x) != 'nan']
sittingcleanedINT = np.array(sittingcleaned).astype(int)

# WALKING
walking = labels.index.where(labels == 'WALKING').values
walkingcleaned = [x for x in walking if str(x) != 'nan']
walkingcleanedINT = np.array(walkingcleaned).astype(int)

# WALKING DOWN STAIRS
walkingdown = labels.index.where(labels == 'WALKING_DOWNSTAIRS').values
walkingdowncleaned = [x for x in walkingdown if str(x) != 'nan']
walkingdowncleanedINT = np.array(walkingdowncleaned).astype(int)


# We are storing the transformed pca values as x_pca
x_pca = pca.transform(scaled_data)

# printing to see the scaled values
# print(x_pca)
# Printing to see labelled values
# print(labels.values)

# Printing to see walking cleaned o see what it looked like
# print(walkingcleaned)


fig = plt.figure()
ax1 = fig.add_subplot(111)
plt.title("PCA data visualization")
#                          PLOTTING POINTS BASED ON ACTIVITY
# Here we are going through the array of avtivties containig their index and matching them to their pca values and plotting them
# Doing this makes us able to see how distinct activities are
# Walking
for i in walkingcleanedINT:
    wlk = ax1.scatter(x=x_pca[i][0], y=x_pca[i][1], s=10, c='b', marker="s")
# WalkingUp
for i in walkingupcleanedINT:
    #
    wlkup = ax1.scatter(x=x_pca[i][0], y=x_pca[i][1], s=10, c='c', marker="s")
# WalkingDown
for i in walkingdowncleanedINT:
    wlkdwn = ax1.scatter(x=x_pca[i][0], y=x_pca[i][1], s=10, c='k', marker="s")
# Standing
for i in standingcleanedINT:
    stnding = ax1.scatter(x=x_pca[i][0], y=x_pca[i][1], s=10, c='r', marker="s")
#      Laying
for i in layingcleanedINT:
    lay = ax1.scatter(x=x_pca[i][0], y=x_pca[i][1], s=10, c='g', marker="s")
#     Sitting
for i in sittingcleanedINT:
    sit = ax1.scatter(x=x_pca[i][0], y=x_pca[i][1], s=10, c='y', marker="s")

plt.legend((wlk, wlkup, wlkdwn, stnding, lay, sit), ('Walking', 'Walking Up', 'Walking Down', 'Standing', 'Laying', 'Sitting'))
plt.show()

print('Done!')


