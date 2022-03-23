
import pandas as pd
from sklearn import preprocessing
#                                  Cleanining TestSet
filename = 'Dataset/trainsaved.csv'
df = pd.read_csv(filename)

# Getting number of null values
print(df.isna().sum())

# Removing invalid characters
columns = df.columns
columns = columns.str.replace('[-]', '')
columns = columns.str.replace('[()]','')
columns = columns.str.replace('[,]','')
columns = columns.str.replace('[.]','')
df.columns = columns
print(df.head(1))
# Savng the new changed datatframe as a .CSV file



# Label encoding our Activities to remove their string values
print("***** Unique Values *****")
print(df.Activity.unique())
le = preprocessing.LabelEncoder()
df['Activity'] = le.fit_transform(df['Activity'])
print(df.Activity)
print("***** Unique Values Encoded *****")
print(df.Activity.unique())
# Create new CSV containing cleaned training data
df.to_csv(r'Dataset/TrainingCleaned.csv')

