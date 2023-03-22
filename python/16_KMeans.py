#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt

# クラスタリング用のデータセット
from sklearn.datasets import make_blobs


# In[2]:


x,y = make_blobs(n_samples=200,n_features=2,centers=3,random_state=10)


# In[3]:


# データセットを表示する関数
def plot_dataset(x,y):
    plt.plot(x[:,0][y==0],x[:,1][y==0],'bo',ms=15)
    plt.plot(x[:,0][y==1],x[:,1][y==1],'r^',ms=15)
    plt.plot(x[:,0][y==2],x[:,1][y==2],'gs',ms=15)
    plt.xlabel('$x_0$',fontsize=15)
    plt.ylabel('$x_1$',fontsize=15)


# In[4]:


plt.figure(figsize=(10,10))
plot_dataset(x,y)

plt.show()


# In[5]:


# k-meansクラスタリング
from sklearn.cluster import KMeans


# In[6]:


# 学習実行
k_means = KMeans(n_clusters=3).fit(x)

k_means.labels_


# In[7]:


def plot_dataset(x,labels):
    plt.plot(x[:,0][labels==0],x[:,1][labels==0],'bo',ms=15)
    plt.plot(x[:,0][labels==1],x[:,1][labels==1],'r^',ms=15)
    plt.plot(x[:,0][labels==2],x[:,1][labels==2],'gs',ms=15)
    plt.xlabel('$x_0$',fontsize=15)
    plt.ylabel('$x_1$',fontsize=15)


# In[8]:


plt.figure(figsize=(10,10))
plot_dataset(x,k_means.labels_)

plt.show()


# In[9]:


# 別のデータでやってみる
from sklearn.datasets import make_moons


# In[10]:


x,y = make_moons(n_samples=200,noise=0.1,random_state=0)

plt.figure(figsize=(12,8))
plot_dataset(x,y)
plt.show()


# In[11]:


k_means = KMeans(n_clusters=2).fit(x)
k_means.labels_


# In[12]:


plt.figure(figsize=(12,8))
plot_dataset(x,k_means.labels_)

plt.show()


# In[13]:


# 階層的クラスタリング
from sklearn.datasets import load_iris

# デンドログラムを書くためのライブラリ
from scipy.cluster.hierarchy import dendrogram,linkage


# In[14]:


iris = load_iris()
x = iris.data


# In[15]:


# linkageでxの距離を測る　method='average'で群平均法
z = linkage(x,method='average',metric='euclidean')


# In[16]:


dendrogram(z)
plt.show()


# In[ ]:




