# Import pandas which allows the importing of
# data from different sources such as JSON, CSV also allows manipulation of data
import pandas
import matplotlib.pyplot as plt

# region Loading Data
filename = 'Dataset/trainsaved.csv'
filename1 = 'Dataset/test.csv'
df = pandas.read_csv(filename)
df1 = pandas.read_csv(filename1)
print(df)
print(df.columns.values)
print(df.dtypes.values)
# endregion


# region Print Unique Values of Activities
print("***** Unique Values *****")
print(df.Activity.unique())
# endregion


# region Used to calculate number of unique values alongside their pecentage weight for training
print(df.Activity.value_counts())
print((df.Activity.value_counts() / df.Activity.count()) * 100)
# endregion¢

# region Used to calculate number of unique values alongside their pecentage weight for test
print(df1.Activity.value_counts())
print((df1.Activity.value_counts() / df1.Activity.count()) * 100)
# endregion¢

#  Dataframes concatted to understand the entire sets and not just te individual Test and Train Set
All = pandas.concat([df, df1], axis=0)
All.info()

plt.title("Visualising Activity makeup of Training Set")
plt.rc('xtick', labelsize=8)
df.Activity.value_counts().plot.pie(autopct='%1.3f%%')
plt.show()

plt.title("Visualising Activity makeup of Test Set")
plt.rc('xtick', labelsize=8)
df1.Activity.value_counts().plot.pie(autopct='%1.3f%%')
plt.show()
