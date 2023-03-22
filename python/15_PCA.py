#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


df = pd.read_csv('titanic_train.csv')
df.head()


# In[3]:


df.info()


# In[4]:


# 欠損値を埋める
df['Age'] = df['Age'].fillna(df['Age'].mean())
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])
df.info()


# In[5]:


# 使わないデータをdrop
x = df.drop(columns=['PassengerId','Survived','Name','Ticket','Cabin'])
# 予測の対象となるデータ
y = df['Survived']

x.head()


# In[6]:


# 文字データを数字に変更
from sklearn.preprocessing import LabelEncoder

cat_features = ['Sex','Embarked']

for col in cat_features:
    lbl = LabelEncoder()
    x[col] = lbl.fit_transform(list(df[col].values))

x.head()


# In[7]:


# 標準化
from sklearn.preprocessing import StandardScaler

num_features = ['Age','Fare']
for col in num_features:
    scaler = StandardScaler()
    x[col] = scaler.fit_transform(np.array(df[col].values).reshape(-1,1))

x.head()


# In[8]:


# 主成分分析
from sklearn.decomposition import PCA


# In[9]:


# 特徴量7個でそれぞれ891個のデータがある。
x.shape


# In[10]:


pca = PCA(n_components=2)

# PCA実行
x_pca = pca.fit_transform(x)

# 特徴量2個でそれぞれ891個のデータがある。
x_pca.shape


# In[11]:


pca = PCA()

# PCA実行
x_pca = pca.fit_transform(x)

# 特徴量7個でそれぞれ891個のデータがある。
x_pca.shape


# In[12]:


# 第一主成分と第二主成分を可視化する関数
def plot_2d(x,y):
    plt.plot(x[:,0][y==0],x[:,1][y==0],'bo',ms=15)
    plt.plot(x[:,0][y==1],x[:,1][y==1],'r^',ms=15)
    plt.xlabel("First Principal Component")
    plt.ylabel("Second Principal Component")
    plt.legend(['Not Survived','survived'],loc='best')
    
from mpl_toolkits.mplot3d import Axes3D

# 第1,2,3主成分を可視化する関数
def plot_3d(x,y):
    fig = plt.figure(figsize=(8,6))
    ax = fig.add_subplot(111,projection='3d')
    
    ax.plot(x[:,0][y==0],x[:,1][y==0],x[:,2][y==0],'bo',ms=15)
    ax.plot(x[:,0][y==1],x[:,1][y==1],x[:,2][y==1],'r^',ms=15)
    
    ax.set_xlabel("First Principal Component",fontsize=15)
    ax.set_ylabel("Second Principal Component",fontsize=15)
    ax.set_zlabel("Third Principal Component",fontsize=15)
    ax.legend(['Not Survived','Survived'],loc='best',fontsize=16)


# In[13]:


plt.figure(figsize=(10,10))
plot_2d(x_pca,y)

plt.show()


# In[14]:


# 以下の一行を記述することで、3Dグラフをマウスで回転できる
get_ipython().run_line_magic('matplotlib', 'notebook')

plt.figure(figsize=(5,5))
plot_3d(x_pca,y)

plt.show()


# In[15]:


# 寄与率
pca.explained_variance_ratio_


# In[16]:


# 寄与率を図示
get_ipython().run_line_magic('matplotlib', 'notebook')

plt.figure(figsize=(12,8))
plt.plot(pca.explained_variance_ratio_)
plt.xlabel('n_components')
plt.ylabel('explained_variance_ratio_')
plt.show()


# In[17]:


# 寄与率の累積を表示する
get_ipython().run_line_magic('matplotlib', 'notebook')

plt.figure(figsize=(12,8))
plt.plot(np.hstack([0,pca.explained_variance_ratio_.cumsum()]))
plt.xlabel('n_components')
plt.ylabel('explained_variance_ratio_')
plt.show()


# In[18]:


# 各主成分に影響を与えている特徴量を確認する
# 例：第一主成分で一番影響を与えているのは、4番目の0.76780369で、SibSpである。
pca.components_


# In[19]:


# 特徴量が主成分にどれくらい影響を与えているのかヒートマップで示す。

plt.matshow(pca.components_,cmap="Greys")
plt.yticks(range(len(pca.components_)),range(1,len(pca.components_) + 1))
plt.colorbar()
plt.xticks(range(x.shape[1]),x.columns.values,rotation=60,ha='left')
plt.xlabel('Features')
plt.ylabel('Principal Components')

plt.show()


# In[ ]:




