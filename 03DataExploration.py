# Import pandas which allows the importing of
# data from different sources such as JSON, CSV also allows manipulation of data
import pandas as pd
# this library stored as plt allows us to plot figures easily
import matplotlib.pyplot as plt




#region Making Printing more visible to show all collumns and rows

# pandas.set_option('display.max_columns', None)

#endregion

#region Loading Data
filename = 'Dataset/trainsaved.csv'
df = pd.read_csv(filename)
print(df)
print(df.columns.values)
#endregion

# Used to plot a bar chart to show how what the dataset is made of


# Enables us to see the data type of each column
# this is important when looking to create the API as we can use this as a form of validation
print(df.dtypes)
# Gives us the discriptive statistics of th dataframe
df.describe()