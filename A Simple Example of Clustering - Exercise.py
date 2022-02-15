from sklearn.cluster import KMeans
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

data = pd.read_csv('Countries-exercise.csv')

plt.scatter(data['Longitude'], data['Latitude'])
plt.xlim(-180, 180)
plt.ylim(-90, 90)
plt.show

x = data.iloc[:, 1:3]

kmeans = KMeans(7)
kmeans.fit(x)

clustering_results = kmeans.fit_predict(x)

data_with_clusters = data.copy()
data_with_clusters['Cluster'] = clustering_results

plt.scatter(data_with_clusters['Longitude'], data_with_clusters['Latitude'], c=data_with_clusters['Cluster'], cmap='rainbow')
plt.xlim(-180, 180)
plt.ylim(-90, 90)
plt.show()
