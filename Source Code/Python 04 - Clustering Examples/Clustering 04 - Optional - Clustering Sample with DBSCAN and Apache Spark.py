import pandas as pd
import numpy as np
import os

# imports for Schema generation
import pyspark.sql.types 
from pyspark.sql.types import *

# other spark
import pyspark.pandas as ps
# Requires PyArrow as dependency
# https://spark.apache.org/docs/latest/api/python/user_guide/pandas_on_spark/index.html
from pyspark.sql import SparkSession
spark = SparkSession.builder.appName("DBSCAN with Spark").getOrCreate()
print("Session established")

# Suppress warnings
import warnings
warnings.filterwarnings("ignore")
# plotting related
import matplotlib.pyplot as plt
import seaborn as sns# for plot styling
sns.set()  
# DBSCAN related
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN


def read_file(filename, s=","):
    file = os.getcwd() + "/" + filename
    xySchema = StructType([
                    StructField('X', FloatType(), True), 
                    StructField('Y', FloatType(), True)])
    sdf= spark.read.format("csv").option("delimiter",s).load(filename, schema=xySchema)
    df = sdf.to_pandas_on_spark()
    return df

# filename="Sample with Three Clusters.csv"
# filename="Sample With Outliers.csv"
filename="Sample With Scale Issues.csv"
# df = read_file(filename, ["X", "Y"], s=";")
df = read_file(filename, s=";")
print(df.dtypes)

# Step 1 - Check for scaling problems
# On this dataset we have no scaling problems.
print("Describing X and Y values")
print( df.describe())
# print( df.Y.describe())
input("Press Enter to continue...")

# Density-based spatial clustering of applications with noise (DBSCAN)
# Differs from k-means. We donot specify the number of clusters,
# but we specify what defines a cluster
# 1- A distance to define neighborhood. Only neighbors form a cluster. (eps)
# 2- A minimum number of points (items, records, etc) to create a cluster (minpoints)

# For an example where k-means performs poorly but dbscan performs well
# https://towardsdatascience.com/dbscan-clustering-for-data-shapes-k-means-cant-handle-well-in-python-6be89af4e6ea

# Now that we know about the steps, we flash-forward a bit
def cluster_dbscan(df):
    X = StandardScaler().fit_transform(df)
    db = DBSCAN(eps=0.3, min_samples=10).fit(X)
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    labels = db.labels_
    # Number of clusters in labels, ignoring noise if present.
    noise = list(labels).count(-1)
    noiseExists = 0
    if noise > 0:
        noiseExists = 1
    clusters = len(set(labels)) - noiseExists
    print('Estimated number of clusters: %d' % clusters)
    print('Estimated number of noise points: %d' % noise)
    return labels, clusters, noise

labels, clusters, noise = cluster_dbscan(df=df)
plt.scatter(df.X, df.Y, c=labels, cmap='viridis', edgecolor='k')
# For alternate colormaps
# https://matplotlib.org/examples/color/colormaps_reference.html
plt.title('Estimated number of clusters: %d' % clusters)
plt.xlabel("X Values")
plt.ylabel("Y Values")
plt.show()