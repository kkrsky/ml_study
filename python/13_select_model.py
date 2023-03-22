#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons


# In[2]:


moons = make_moons(n_samples=200,noise=0.2,random_state=0)
x = moons[0]
y = moons[1]


# In[3]:


from matplotlib.colors import ListedColormap

def plot_decision_boundary(model,x,y,margin=0.3):
    _x1 = np.linspace(x[:,0].min()-margin,x[:,0].max()+margin,100)
    _x2 = np.linspace(x[:,1].min()-margin,x[:,1].max()+margin,100)
    x1,x2 = np.meshgrid(_x1,_x2)
    x_new = np.c_[x1.ravel(),x2.ravel()]
    y_pred = model.predict(x_new).reshape(x1.shape)
    custom_cmap = ListedColormap(['mediumblue','orangered'])
    plt.contourf(x1,x2,y_pred,alpha=0.3,cmap=custom_cmap)

def plot_dataset(x,y):
    plt.plot(x[:,0][y==0],x[:,1][y==0],'bo',ms=15)
    plt.plot(x[:,0][y==1],x[:,1][y==1],'r^',ms=15)
    plt.xlabel("$x_0$",fontsize=30)
    plt.ylabel("$x_1$",fontsize=30,rotation=0)


# In[4]:


plt.figure(figsize=(12,8))
plot_dataset(x,y)
plt.show()


# In[5]:


# データを任意数分割してみる

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split


# In[6]:


x_train,x_test,y_train,y_test = train_test_split(x,y,random_state=0)


# In[7]:


# LogisticRegression実行
log_reg = LogisticRegression().fit(x_train,y_train)

# DecisionTreeClassifier実行
tree_clf = DecisionTreeClassifier().fit(x_train,y_train)

# このまま正解率を評価するのは好ましくない。
# トレーニング用、テスト用のデータの分割次第で正解率が異なる可能性がある。
print("LogisticRegression={:.2f}".format(log_reg.score(x_test,y_test)))
print("DecisionTreeClassifier={:.2f}".format(tree_clf.score(x_test,y_test)))


# In[8]:


from sklearn.model_selection import KFold,cross_val_score


# In[9]:


# データを分割する関数 n_splits=5で分割数を5個とする
kfold = KFold(n_splits=5,shuffle=True,random_state=0)

# cross_val_score(モデル,データ,正解ラベル,分割数)
log_reg_score = cross_val_score(log_reg,x,y,cv=kfold)
tree_clf_score = cross_val_score(tree_clf,x,y,cv=kfold)

print(log_reg_score) # 誤差10%くらい
print(tree_clf_score) # 誤差5%くらい


# In[10]:


print("LogisticRegression={:.2f}".format(log_reg_score.mean()))
print("DecisionTreeClassifier={:.2f}".format(tree_clf_score.mean()))


# In[11]:


# 混合行列を取得してみる

from sklearn.metrics import confusion_matrix


# In[12]:


# 予測値を算出
y_pred_log_reg = log_reg.predict(x_test)
y_pred_tree_clf = tree_clf.predict(x_test)

# 混合行列を取得
cm_log_reg = confusion_matrix(y_test,y_pred_log_reg)
cm_tree_clf = confusion_matrix(y_test,y_pred_tree_clf)

print(cm_log_reg)
print("\n")
print(cm_tree_clf)


# In[13]:


# 混合行列を用いて様々な評価基準を表示してみる
# precision_score: 適合率
# recall_score: 再現率
# f1_score: F値

from sklearn.metrics import precision_score,recall_score,f1_score


# In[14]:


# 適合率
print("precision log_reg:\n",precision_score(y_test,y_pred_log_reg))
print("\n")
print("precision tree_clf:\n",precision_score(y_test,y_pred_tree_clf))


# In[15]:


# 再現率
print("recall log_reg:\n",recall_score(y_test,y_pred_log_reg))
print("\n")
print("recall tree_clf:\n",recall_score(y_test,y_pred_tree_clf))


# In[16]:


# F値
print("f1 log_reg:\n",f1_score(y_test,y_pred_log_reg))
print("\n")
print("f1 tree_clf:\n",f1_score(y_test,y_pred_tree_clf))


# In[17]:


# 適合率、再現率のトレードオフの関係を確認する
from sklearn.metrics import precision_recall_curve


# In[18]:


# DecisionTreeClassifier
precision,recall,threshold = precision_recall_curve(y_test,y_pred_tree_clf)

plt.figure(figsize=(12,8))
plt.plot(precision,recall)
plt.xlabel("precision")
plt.ylabel("recall")
plt.show()


# In[19]:


# LogisticRegression
precision,recall,threshold = precision_recall_curve(y_test,y_pred_log_reg)

plt.figure(figsize=(12,8))
plt.plot(precision,recall)
plt.xlabel("precision")
plt.ylabel("recall")
plt.show()


# In[20]:


# 平均二乗誤差とR誤差について確認する

from mglearn.datasets import make_wave


# In[21]:


x,y = make_wave(n_samples=100)

plt.figure(figsize=(12,8))
plt.scatter(x,y)
plt.xlabel("x")
plt.ylabel("y")
plt.show()


# In[22]:


# 線形回帰を実行してみる

from sklearn.linear_model import LinearRegression


# In[23]:


x_train,x_test,y_train,y_test = train_test_split(x,y,random_state=0)
lin_reg = LinearRegression().fit(x_train,y_train)


# In[24]:


# R^2の値
print(lin_reg.score(x_test,y_test))


# In[25]:


# 平均二乗誤差
from sklearn.metrics import mean_squared_error

# 予測値
y_pred = lin_reg.predict(x_test)

mse = mean_squared_error(y_test,y_pred)
mse


# In[26]:


rmse = np.sqrt(mse)
rmse


# In[ ]:




