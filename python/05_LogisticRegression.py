#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from sklearn.datasets import load_iris


# In[2]:


# sklearnのiris(あやめ)データセットを利用する
iris = load_iris()

# data :データ
# feature_names: データの各列に対応する特徴量の名前
# target: どういうクラスに分類されているかを示す。
# target_names: 各クラス分類に対応する名前
iris


# In[3]:


# データを見やすくする
df = pd.DataFrame(iris.data,columns=iris.feature_names)
df['target'] = iris.target

# 4つの特徴量を持つ150個のデータがあります
df


# In[4]:


# 3つ目の特徴量（petal_length：2列目）だけを使って分類してみる
# irisデータセットは、target(データの分類)が0,1,2の3つあり、各50個ずつデータが格納されている。
# 今回は、target=1,2のみ使うので、50行目以降のデータを抽出してベクトル化
x = iris.data[50:,2].reshape(-1,1)

# target=1,2を扱うが、分類を０と１にしたいので、－１する。
y = iris.target[50:] - 1


# In[5]:


x


# In[6]:


y


# In[7]:


# ロジスティック回帰用の関数をインポート
from sklearn.linear_model import LogisticRegression
# 標準化用クラス
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


# In[8]:


scaler = StandardScaler()
# 標準化されたデータを生成
x_scaled = scaler.fit_transform(x)

x_scaled


# In[9]:


# 訓練用とテスト用データ作成
x_train,x_test,y_train,y_test = train_test_split(x_scaled,y,random_state=0)


# In[10]:


# モデルを生成して学習
log_reg = LogisticRegression().fit(x_train,y_train)


# In[11]:


# 学習結果を表示
log_reg.intercept_,log_reg.coef_


# In[12]:


# どれくらい正しく分類できたか

print("train score={}".format(log_reg.score(x_train,y_train)))
print("test score={}".format(log_reg.score(x_test,y_test)))


# In[ ]:




