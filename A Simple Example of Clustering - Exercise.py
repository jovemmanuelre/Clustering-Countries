#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
from sklearn.cluster import KMeans


# In[2]:


data = pd.read_csv('Countries-exercise.csv')
data


# In[3]:


plt.scatter(data['Longitude'],data['Latitude'])
plt.xlim(-180,180)
plt.ylim(-90,90)
plt.show


# In[4]:


x = data.iloc[:,1:3]
x


# ## Clustering

# In[9]:


kmeans=KMeans(7)


# In[10]:


kmeans.fit(x)


# In[11]:


clustering_results=kmeans.fit_predict(x)
clustering_results


# In[12]:


data_with_clusters = data.copy()
data_with_clusters['Cluster'] = clustering_results
data_with_clusters


# In[13]:


plt.scatter(data_with_clusters['Longitude'],data_with_clusters['Latitude'],c=data_with_clusters['Cluster'],cmap='rainbow')
plt.xlim(-180,180)
plt.ylim(-90,90)
plt.show()

