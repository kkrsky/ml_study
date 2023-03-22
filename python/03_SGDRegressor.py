#!/usr/bin/env python
# coding: utf-8

# In[1]:


#解析的方法　正規方程式を解く

import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


# 学習データ(x軸)を生成
x = np.random.rand(100,1)

#データをみてみる
x


# In[3]:


# 学習データ(y軸)を生成
y = 5 + 3 * x + np.random.rand(100,1)

# 生成したデータを表示
plt.scatter(x,y)


# In[4]:


from sklearn.linear_model import LinearRegression


# In[5]:


# fit関数でLinearRegressionの学習を行う
lin_reg = LinearRegression().fit(x,y.ravel())

# y.ravel() : 多次元配列を1次元の配列に変換する
# 例：yy=np.random.rand(100,3)
# yy[0:5]
# yy.ravel()[0:10]

y.shape # 100行1列のデータ


# In[6]:


# 1次元の配列に変換する
y.ravel().shape


# In[7]:


# intercept_: 切片
# coef_: 傾き

# LinearRegressionの実行結果を表示する
# 学習データ y = 5 + 3 * x + randomで設定した切片、傾きに近い値を得ることができた。
lin_reg.intercept_,lin_reg.coef_


# In[8]:


# LinearRegressionで取得した直線を描画　(正規方程式を用いた描画)
x_new = np.array([[0],[1]])
plt.plot(x_new,lin_reg.intercept_ + lin_reg.coef_ * x_new,'red')
plt.scatter(x,y)


# In[9]:


# 以下、勾配降下法で傾きと切片を取得する

import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[10]:


# 学習データを再表示

plt.scatter(x,y)


# In[11]:


from sklearn.linear_model import SGDRegressor

# SGDRegressorにより学習
sgd_reg = SGDRegressor(max_iter=100).fit(x,y.ravel())

# intercept_: 切片
# coef_: 傾き
# SGDRegressorの学習結果を表示
sgd_reg.intercept_, sgd_reg.coef_


# In[12]:


# SGDRegressorで取得した直線を描画　(降下法を用いた描画)
x_new = np.array([[0],[1]])
plt.plot(x_new,sgd_reg.intercept_ + sgd_reg.coef_ * x_new,'red')
plt.scatter(x,y)


# In[13]:


# 学習率(eta0)の影響をみてみる　デフォルトは0.01
# 学習率：どの程度の大きさでパラメータの更新を行うかを決めるもの。
# 例えば　eta0=0.0001にしてみる

# SGDRegressorにより学習
sgd_reg_00001 = SGDRegressor(eta0=0.0001,max_iter=100).fit(x,y.ravel())

# SGDRegressorの学習結果を表示
# 学習率が低すぎるので、切片、傾きが想定よりも低く、適切に学習を行えていないことがわかる。
sgd_reg_00001.intercept_,sgd_reg_00001.coef_


# In[14]:


# 不適切な学習率による、SGDRegressorで取得した直線を描画
x_new = np.array([[0],[1]])
plt.plot(x_new,sgd_reg_00001.intercept_ + sgd_reg_00001.coef_ * x_new,'red')
plt.scatter(x,y)

