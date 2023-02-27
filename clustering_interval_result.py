import pandas as pd
import numpy as np
#from tslearn.clustering import TimeSeriesKMeans
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
from tqdm import tqdm

def get_score_metric(X):
    # Define a list to store the within-cluster sum of squared distances for each number of clusters
    score_metric = []

    # Loop over a range of possible number of clusters
    for i in tqdm(range(2, 30)):
        kmeans = KMeans(n_clusters=i, random_state=0)
        kmeans.fit(X)
        score_metric.append([kmeans.inertia_, silhouette_score(data, kmeans.labels_)])

    return pd.DataFrame(score_metric, columns=['wcss', 'silhouette'])


data = pd.read_csv('./interval_result.csv', index_col=0, parse_dates=True)

# Convert the list to a 2D numpy array for processing
data['RangeIdx_pred'] = data.y_Rtn_upper_bound - data.y_Rtn.values
data['Precision'] = (np.sign(data.y_Rtn) == np.sign(data.Act_Rtn)) * 1

#data = data.loc['2018-01-01':]

X = data[['y_Rtn', 'RangeIdx_pred']] #'Act_Rtn', 'RangeIdx', 'y_Rtn', 'RangeIdx_pred'

#score_metric = get_score_metric(X)

# Determine the optimal number of clusters
n_clusters = 6  # In this example, the optimal number of clusters is 3, as determined from the plot

# Create an instance of KMeans
kmeans = KMeans(n_clusters=n_clusters, random_state=0)

# Fit the model to the data
kmeans.fit(X)

# Get the cluster labels for each data point
labels = kmeans.labels_
data['label'] = labels

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
a = data.groupby('label').mean()
a['count'] = data.groupby('label').count().y_Rtn.values
a['Precision'] = data.groupby('label').mean().Precision.values
print(a)
a =1
# Get the cluster centers
#cluster_centers = kmeans.cluster_centers_

# Print the results
#print("Cluster Labels: ", labels)
#print("Cluster Centers: ", cluster_centers)
