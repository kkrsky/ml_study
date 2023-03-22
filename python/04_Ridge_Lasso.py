#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt


# In[2]:


# 学習データ作成

# データ点数
data_size = 20

# 0~1までのデータを20個作成
x = np.linspace(0,1,data_size)
x


# In[3]:


noise = np.random.uniform(low=-1.0,high=1.0,size=data_size) * 0.2

# ノイズを加えたy軸の値
y = np.sin(2.0 * np.pi * x) + noise

plt.scatter(x,y)


# In[4]:


# 元のsin波を表示

x_line = np.linspace(0,1,1000)
sin_x = np.sin(2.0 * np.pi * x_line)
plt.plot(x_line,sin_x,'red')


# In[5]:


# 元のsin波と学習データを描画
# 何度も呼び出したいので関数化

def plot_sin():
    plt.scatter(x,y)
    plt.plot(x_line,sin_x,'red')

# 関数呼び出し 
plot_sin()


# In[6]:


# 線形回帰学習モデルをインポート
from sklearn.linear_model import LinearRegression


# In[7]:


# 以下、モデルをつくって学習

#xの形を確認
x


# In[8]:


#xは１次元配列なのでこれをベクトル形式にする必要がある、のでreshapeしておく (20,) －＞ (20,1)
x.shape


# In[9]:


# x.reshape(行,列): 例：x.reshape(-1,2) (20,) －＞ (10,2)
# -1行は全ての行を指定。

# 線形回帰 実行
lin_reg = LinearRegression().fit(x.reshape(-1,1),y)

# 学習結果を表示
lin_reg.intercept_,lin_reg.coef_


# In[10]:


# 図にする

# まったく近似できていないのを確認
plt.plot(x_line, lin_reg.intercept_ + lin_reg.coef_ * x_line)
plot_sin()


# In[11]:


# 多項式回帰をやってみる
# 二乗を追加
x_2 = x ** 2

# 学習用データを生成 axis=1で列方向に配列を1つ結合させる
x_new = np.concatenate([x.reshape(-1,1),x_2.reshape(-1,1)],axis=1)

x_new


# In[12]:


# 再びモデルをつくって学習
lin_reg_2 = LinearRegression().fit(x_new,y)

# 2つの傾きが得られる
lin_reg_2.intercept_,lin_reg_2.coef_


# In[13]:


# 図にする
# 二乗だけでは、不十分であることがわかる。
plt.plot(x_line,lin_reg_2.intercept_ + lin_reg_2.coef_[0] * x_line + lin_reg_2.coef_[1] * x_line ** 2)
plot_sin()


# In[14]:


# 上記の多項式をscikit-learnのライブラリのPolynomialFeaturesを用いると簡単に実装できる。
from sklearn.preprocessing import PolynomialFeatures


# In[15]:


# degree=3で三乗
poly = PolynomialFeatures(degree=3)

# 多項式を生成
poly.fit(x.reshape(-1,1))

# 得られたデータを格納
x_poly_3 = poly.transform(x.reshape(-1,1))

# 0列目:1のみ 1列目：元データ　2列目：2乗データ　3列目：3乗データ
x_poly_3


# In[16]:


# LinearRegressionの学習実行
lin_reg_3 = LinearRegression().fit(x_poly_3,y)


# In[17]:


# y=ax^3 + bx^2 + cxのそれぞれの係数に対してxの乗数をかける式を作成するのは面倒
# 例：y= lin_reg_2.intercept_ + lin_reg_2.coef_[0] * x_line + lin_reg_2.coef_[1] * x_line ** 2
# そこで、predictメソッドを利用すると簡単に表現できる
# 注意：predict関数の引数は、学習データと同じ形でないといけない。(今回の場合は、x_poly_3=0列目:1のみ 1列目：元データ　2列目：2乗データ　3列目：3乗データ)
# そのため、まずはx_lineを学習データと同じ形に変換する

# 変換前のデータを表示
x_line


# In[18]:


#fitとtransformをいっぺんに呼び出す関数がfit_transform
x_line_poly_3 = poly.fit_transform(x_line.reshape(-1,1))

# 学習データx_poly_3と同じ形式になった。
x_line_poly_3


# In[19]:


# グラフを描画
plt.plot(x_line,lin_reg_3.predict(x_line_poly_3))
plot_sin()

# 3乗にすると、LinearRegressionで学習させた結果は、元のsin波に近づいた。


# In[20]:


# 乗数をさらに増やしてみる。

# グラフ描画用のキャンバスを生成
fig,axes = plt.subplots(1,3,figsize=(16,4))

# 5乗、15乗、25乗を用意
for degree,ax in zip([5,15,25],axes):
    poly = PolynomialFeatures(degree=degree)
    x_poly = poly.fit_transform(x.reshape(-1,1))
    
    # 学習させる
    lin_reg = LinearRegression().fit(x_poly,y)
    
    # predict関数を実行するために、データを整形
    x_line_poly = poly.fit_transform(x_line.reshape(-1,1))
    
    # 描画
    ax.plot(x_line,lin_reg.predict(x_line_poly))
    ax.scatter(x,y)
    ax.plot(x_line,sin_x,'red')
    
    # グラフタイトルを描画
    ax.set_title("{} degree".format(degree))


# In[21]:


# 以上より、最小二乗法では、単純に次数を増やしても効果はないということが分かった
# 過学習（過剰適合）の問題がある。
# 正則化というテクニックを使って過学習を防いでみる

import mglearn
import pandas as pd
from sklearn.model_selection import train_test_split


# In[22]:


# データセットを用意 (scikit-learn 1.2でボストンのデータセットをサポートしなくなる旨のエラーが表示される)
x,y = mglearn.datasets.load_extended_boston()

# xは住宅の部屋の数などのデータ
x


# In[23]:


# yは家賃データ
y


# In[24]:


# panadasで見やすい表に変換
df_x = pd.DataFrame(x)
df_y = pd.DataFrame(y)

df_x


# In[25]:


# データ作成
x_train,x_test,y_train,y_test = train_test_split(x,y,random_state=0)


# In[26]:


# モデルを作って学習
lin_reg = LinearRegression().fit(x_train,y_train)


# In[27]:


# 訓練データでは95%適合しているが、未知のデータ(テストデータ)を与えると適合率は60%になる。
# 104項目もあるためテストデータのscoreは低い
# つまり、どの項目を使うか選択する必要がある

print("train score={}".format(round(lin_reg.score(x_train,y_train),3)))
print("test score={}".format(round(lin_reg.score(x_test,y_test),3)))


# In[28]:


# リッジ回帰とラッソ回帰をやって過学習を防いでみる
# リッジ回帰、ラッソ回帰用の関数をインポート
from sklearn.linear_model import Ridge,Lasso


# In[29]:


# リッジ回帰で学習
ridge = Ridge().fit(x_train,y_train)


# In[30]:


# モデルのトレーニングとテストデータの正解率を表示する関数
def print_score(model):
    print("train score={}".format(round(model.score(x_train,y_train),3)))
    print("test score={}".format(round(model.score(x_test,y_test),3)))


# In[31]:


# リッジ回帰解析結果
# LinearRegressionよりテストの正答率が向上した。
# リッジ解析：係数の絶対値を小さくして学習

print_score(ridge)


# In[32]:


# alpha: alphaが大きいと、リッジ回帰の係数の絶対値が小さくなる
# デフォルト係数は１、alpha=10としてみる。
# テストデータの正解率は低下した。
ridge_10 = Ridge(alpha=10).fit(x_train,y_train)
print_score(ridge_10)


# In[33]:


# alpha=0.1としてみる。
# テストデータの正解率は増加した。
ridge_01 = Ridge(alpha=0.1).fit(x_train,y_train)
print_score(ridge_01)


# In[34]:


# リッジ回帰における、係数を比較
coefficients = pd.DataFrame({'lin_reg':lin_reg.coef_,'ridge':ridge.coef_,'ridge_10':ridge_10.coef_,'ridge_01':ridge_01.coef_})

# 行は、データの次元数
# 通常の線形回帰では、全体的に各項目の係数が大きくなっているが、リッジ解析では、係数の絶対値が小さくなっている。
# リッジ回帰はalphaを適切な値に設定する必要がある。
coefficients


# In[35]:


# ラッソ回帰

# ラッソ回帰で学習
lasso = Lasso().fit(x_train,y_train)

# 結果を表示
# 訓練とテストデータの正答率がどちらも低い
# これは、データが足りないことが想定される＝ほとんどの係数を0にしてしまっている
print_score(lasso)


# In[36]:


# ラッソ回帰のalphaで調整する
lasso_001 = Lasso(alpha=0.01, max_iter=10000).fit(x_train,y_train)
print_score(lasso_001)


# In[37]:


# ラッソ回帰における、係数を比較
coefficients = pd.DataFrame({'lin_reg':lin_reg.coef_,'ridge':lasso.coef_,'lasso_001':lasso_001.coef_})

# 行は、データの次元数
# 通常の線形回帰では、全体的に各項目の係数が大きくなっているが、リッジ解析では、係数の絶対値が小さくなっている。
# ラッソ回帰はalphaを適切な値に設定する必要がある。
coefficients


# In[ ]:




