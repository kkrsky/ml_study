#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits


# In[2]:


# 手書き文字の画像をロードするデータセット。64個の特徴量で手書き文字を表している。
digits = load_digits()
digits


# In[3]:


x = digits.data
y = digits.target

# 1794の文字に対して、64個の特徴がある。
x.shape


# In[4]:


images_with_labels = list(zip(digits.images,digits.target))

plt.figure(figsize=(15,6))
for idx,(image,label) in enumerate(images_with_labels[:10]):
    plt.subplot(2,5,idx + 1)
    plt.imshow(image,cmap=plt.cm.gray_r,interpolation='nearest')
    plt.axis('off')
    plt.title('{}'.format(label),fontsize=25)

# digits.dataに格納されている手書き文字をグラフ化
plt.show()


# In[5]:


# なぜ、荒い画像なのか
# 8x8 pixcel
digits.images.shape


# In[6]:


# この手書き文字10個の分類をロジスティック回帰で実現させる

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression


# In[7]:


# データをtrainとtestに分割　乱数は固定
x_train,x_test,y_train,y_test = train_test_split(x,y,random_state=0)


# In[8]:


# データを標準化(平均0 分散1)
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.fit_transform(x_test)

x_train_scaled


# In[9]:


# ロジスティック回帰学習実行
log_reg = LogisticRegression().fit(x_train_scaled, y_train)

# 学習結果
log_reg.intercept_, log_reg.coef_


# In[10]:


# 10は分類するクラスの数、それぞれ分類するためのパラメータが64個ある。
log_reg.coef_.shape


# In[11]:


# 予測してみる
prediction = log_reg.predict(x_test_scaled)

# 予測値を表示
prediction


# In[12]:


# 正解の分類を表示
y_test


# In[13]:


# 予測と正解の分類が一致する場合はTrue
prediction == y_test


# In[14]:


# 混同行列を求める
from sklearn.metrics import confusion_matrix


# In[15]:


confusion = confusion_matrix(prediction, y_test)

# 0と認識されたものが37個
# 1と認識されたものが42個、、、
confusion


# In[16]:


# 正解率
log_reg.score(x_test, y_test)


# In[ ]:




