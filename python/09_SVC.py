#!/usr/bin/env python
# coding: utf-8

# In[1]:


import mglearn
import numpy as np
import matplotlib.pyplot as plt

# 学習に使用するデータセット
from sklearn.datasets import make_moons


# In[2]:


moons = make_moons(n_samples=200,noise=0.1,random_state=0)

# 最初の配列がdata,後ろの配列が正解ラベルのtarget
moons


# In[3]:


# data
x = moons[0]
x


# In[4]:


# 正解ラベル
y = moons[1]
y


# In[5]:


# データを表示
mglearn.discrete_scatter(x[:,0],x[:,1],y)
plt.plot()


# In[6]:


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# まずは、本当にSVCでは分類できないのかやってみる
from sklearn.svm import LinearSVC


# In[7]:


# テスト用とトレーニング用にデータを分離
x_train,x_test,y_train,y_test = train_test_split(x,y,stratify=y,random_state=0)

scaler=StandardScaler()

# SCV実行
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.fit_transform(x_test)

# 150個のデータと2個の特徴を持つことがわかる。
print(x_train_scaled.shape)
print(x_train_scaled)

lin_svm = LinearSVC().fit(x_train_scaled,y_train)


# In[8]:


plt.figure(figsize=(12,8))
mglearn.plots.plot_2d_separator(lin_svm,x)
mglearn.discrete_scatter(x[:,0],x[:,1],y)
plt.xlabel("$x_0$",fontsize=20)
plt.ylabel("$x_1$",fontsize=20)
# SVCでは不可だと分かる


# In[9]:


# 高次特徴空間に写像してやってみる(PolynomialFeaturesで実現できる)
from sklearn.preprocessing import PolynomialFeatures

# dgree=3で3次元
poly = PolynomialFeatures(degree=3)

# 高次元化
x_train_poly = poly.fit_transform(x_train)
x_test_poly = poly.fit_transform(x_test)

# 150個のデータと10個の特徴に拡張されたことがわかる。
print(x_train_poly.shape)
print(x_train_poly)


# In[10]:


# どのように次元を拡張したのか確認する
# 高次元化するほど、計算コストがかかる。→カーネル法を利用すると計算コストを削減できる
poly.get_feature_names_out()


# In[11]:


# 写像したデータをスケーリングして学習
x_tarin_poly_scaled = scaler.fit_transform(x_train_poly)
x_test_poly_scaled = scaler.fit_transform(x_test_poly)

lin_svm = LinearSVC().fit(x_tarin_poly_scaled,y_train)

# testデータで予測した結果と、実際の正解ラベルが一致したらTrueとする
lin_svm.predict(x_test_poly_scaled) == y_test


# In[12]:


# データの前処理として、PolynomialFeaturesやStandardScalerを実行するのが手間。
# Pipelineを使ってデータの前処理を簡素化してみる
from sklearn.pipeline import Pipeline

poly_svm = Pipeline([
    ('poly',PolynomialFeatures(degree=3)),
    ('scaler',StandardScaler()),
    ('svm',LinearSVC())
])

# Pipelineで定義した処理を実行
poly_svm.fit(x,y)


# In[13]:


# グラフ描画関数
def plot_decision_function(model):
    _x0 = np.linspace(-1.5,2.5,100)
    _x1 = np.linspace(-1.0,1.5,100)
    x0,x1 = np.meshgrid(_x0,_x1)
    x = np.c_[x0.ravel(),x1.ravel()]
    y_pred = model.predict(x).reshape(x0.shape)
    y_decision = model.decision_function(x).reshape(x0.shape)
    plt.contourf(x0,x1,y_pred,cmap=plt.cm.brg,alpha=0.2)
    plt.contourf(x0,x1,y_decision,levels=[y_decision.min(),0,y_decision.max()],alpha=0.3)

# 元のデータセットのグラフを描画する関数
def plot_dataset(x,y):
    plt.plot(x[:,0][y==0],x[:,1][y==0],'bo',ms=15)
    plt.plot(x[:,0][y==1],x[:,1][y==1],'r^',ms=15)
    plt.xlabel("$x_1$",fontsize=20)
    plt.ylabel("$x_2$",fontsize=20,rotation=0)

# グラフを描画
plt.figure(figsize=(12,8))
plot_decision_function(poly_svm)
plot_dataset(x,y)

plt.show()


# In[14]:


# 以上の機械学習では、計算コストがかかるため、計算コスト削減としてカーネル法を用いる。
from sklearn.svm import SVC

# kernelという引数にpolyを指定することで多項式回帰のカーネル関数を適応させる
kernel_svm = Pipeline([
    ('scaler',StandardScaler()),
    ('svm',SVC(kernel='poly',degree=3,coef0=1))
])

# 学習実行
kernel_svm.fit(x,y)


# In[15]:


# グラフを描画
plt.figure(figsize=(12,8))
plot_decision_function(kernel_svm)
plot_dataset(x,y)
plt.show()


# In[16]:


# degreeを2,3,5,15と増やした場合

plt.figure(figsize=(20,15))

for i,degree in enumerate([2,3,5,15]):
    poly_kernel_svm = Pipeline([
        ('scaler',StandardScaler()),
        ('svm',SVC(kernel='poly',degree=degree,coef0=1))
    ])
    
    poly_kernel_svm.fit(x,y)
    
    plt.subplot(221 + i)
    plot_decision_function(poly_kernel_svm)
    plot_dataset(x,y)
    plt.title("d = {}".format(degree),fontsize=20)
    
plt.show()


# In[ ]:




