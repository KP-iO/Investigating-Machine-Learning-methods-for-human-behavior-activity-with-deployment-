# Import pandas which allows the importing of
# data from different sources such as JSON, CSV also allows manipulation of data
import pandas

# Got the Training/Test Dataset file
filename = 'Dataset/trainsaved.csv'
filename1 = 'Dataset/test.csv'

# Creating a DataFrame called DF that will store the dataframe being created from file
df = pandas.read_csv(filename)
df1 = pandas.read_csv(filename1)
# Printed the columns, head of the data and the shape of it to understand the data I'm dealing with for Training
print(df.columns.values)
print(df.head)
df.info()
# Printed the columns, head of the data and the shape of it to understand the data I'm dealing with for Testing
print(df1.columns.values)
print(df1.head)
df1.info()

