#!/usr/bin/env python
# coding: utf-8

# In[1]:


import mglearn
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC

# 学習に用いるあやめのデータセット

from sklearn.datasets import load_iris


# In[2]:


iris = load_iris()

# data :データ(4列)
# feature_names: データの各列に対応する特徴量の名前
# ['sepal length (cm)',  'sepal width (cm)',  'petal length (cm)',  'petal width (cm)']
# target: どういうクラスに分類されているかを示す。
# target_names: 各クラス分類に対応する名前
# ['setosa', 'versicolor', 'virginica']
iris


# In[3]:


# irisデータセットは、target(データの分類)が0,1,2の3つあり、各50個ずつデータが格納されている。
# 今回は、target=1,2('versicolor', 'virginica')のみ使うので、50行目以降のデータを抽出してベクトル化
# さらに、'petal length (cm)',  'petal width (cm)'のみを扱うので2列以降のデータを取得
x = iris.data[50:,2:]

# target=1,2を扱うが、分類を０と１にしたいので、－１する。
y = iris.target[50:] -1


# In[4]:


x


# In[5]:


y


# In[6]:


# mglearnを使って散布図を描画

# xの0列目は、'petal length (cm)', 1列目は 'petal width (cm)'のデータ
mglearn.discrete_scatter(x[:,0],x[:,1],y)

# yに格納されている値で、0は、'versicolor', 1は'virginica'を意味する
plt.legend(['versicolor','virginica'],loc='best')
plt.xlabel("petal length (cm)")
plt.ylabel("petal width (cm)")

plt.show()


# In[7]:


# テストデータとトレーニングデータに分割する
x_train,x_test,y_train,y_test = train_test_split(x,y,stratify=y,random_state=0)


# In[8]:


# LinearSVC学習を実行
svm = LinearSVC().fit(x_train,y_train)


# In[9]:


# どういう風に分類したか図示する関数
def plot_separator(model):
    #グラフのサイズを指定
    plt.figure(figsize=(10,6))
    
    # どういう風に分類したか図示する関数
    mglearn.plots.plot_2d_separator(model,x)
    
    #散布図を描画
    mglearn.discrete_scatter(x[:,0],x[:,1],y)
    
    #軸の名前
    plt.xlabel('petal length')
    plt.ylabel('petal width')
    
    #表示範囲
    plt.xlim(2.8,7.0)
    plt.ylim(0.8,2.6)
    
    plt.show()


# In[10]:


# グラフを描画
plot_separator(svm)


# In[11]:


# ペナルティの度合い：ハイパーパラメータＣをいじってどれくらい予測の違いがでるのか
# デフォルトはＣ＝１
# 再度、学習を実行

# 下記のメッセージが表示される場合は、max_iterでiterationsを調整できる。
# ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
svm_100 = LinearSVC(C=100, max_iter=100000).fit(x_train,y_train)

plot_separator(svm_100)


# In[12]:


# score(正しく分類できた割合)を見てみる
# C=1の場合
print('score on training set : {:.2f}'.format(svm.score(x_train,y_train)))
print('score on test set : {:.2f}'.format(svm.score(x_test,y_test)))


# In[13]:


# C=100の場合
print('score on training set : {:.2f}'.format(svm_100.score(x_train,y_train)))
print('score on test set : {:.2f}'.format(svm_100.score(x_test,y_test)))


# In[14]:


# C=0.01の場合
svm_001 = LinearSVC(C=0.01).fit(x_train,y_train)

plot_separator(svm_001)


# In[15]:


# C=0.01の場合
print('score on training set : {:.2f}'.format(svm_001.score(x_train,y_train)))
print('score on test set : {:.2f}'.format(svm_001.score(x_test,y_test)))


# In[ ]:




