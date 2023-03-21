import pandas as pd
# import numpy as np
import os

print("Where is pandas library located?")
print(pd)
filename = os.getcwd() + "/dataset.txt"
# For Windows you should use this as
# filename = os.getcwd() + "\\dataset.txt"

df = pd.read_csv(filename) # reads into a dataframe
print("The whole data frame:")
print(df)
#print(df.axes)
#print(df.index)
#print(df.columns)

# access individual column
print("The Name column accessed individually.")
print(df["Name"])

#print("Has size:", df.Name.size, " , has objects of type", df.Name.dtype )
print("Unique names:", df.Name.unique())
print("Frequencies:")
print(df.Name.value_counts())


print("Average coffee consumption")
print("Average:", df.Cup.mean())
print("The first two lines  in the Cup column have data:")
print(df.Cup.head(2) )
print("The last two lines in the Cup column have data:")
print(df.Cup.tail(2) )

print("Unique names in the list of persons:")
for name in df.Name.unique():
    print(name)

print("Total number of cups consumed by each person:")
print( df.groupby("Name")["Cup"].sum() )
#             ^ Filter


