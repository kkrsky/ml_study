#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons


# In[2]:


moons = make_moons(n_samples=200,noise=0.2,random_state=0)
x = moons[0] # data
y = moons[1] # target (正解ラベル)


# In[3]:


x


# In[4]:


from matplotlib.colors import ListedColormap

# 境界をプロットする関数
def plot_decision_boundary(model,x,y,margin=0.3):
    _x1 = np.linspace(x[:,0].min()-margin,x[:,0].max()+margin,100)
    _x2 = np.linspace(x[:,1].min()-margin,x[:,1].max()+margin,100)
    x1,x2 = np.meshgrid(_x1,_x2)
    x_new = np.c_[x1.ravel(),x2.ravel()]
    y_pred = model.predict(x_new).reshape(x1.shape)
    custom_cmap = ListedColormap(['mediumblue','orangered'])
    plt.contourf(x1,x2,y_pred,alpha=0.3,cmap=custom_cmap)

# 散布図をプロットする関数
def plot_dataset(x,y):
    plt.plot(x[:,0][y==0],x[:,1][y==0],'bo',ms=15)
    plt.plot(x[:,0][y==1],x[:,1][y==1],'r^',ms=15)
    plt.xlabel("$x_0$",fontsize=30)
    plt.ylabel("$x_1$",fontsize=30,rotation=0)


# In[5]:


plt.figure(figsize=(12,8))
plot_dataset(x,y)
plt.show()


# In[6]:


# 分類を実行
# 決定木の場合
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

x_train,x_test,y_train,y_test = train_test_split(x,y,random_state=0)

tree_clf = DecisionTreeClassifier().fit(x_train,y_train)


# In[7]:


plt.figure(figsize=(12,8))
plot_decision_boundary(tree_clf,x,y)
plot_dataset(x,y)

# 決定木では、分類面が直線になってしまう。
plt.show()


# In[8]:


# ランダムフォレストの場合
from sklearn.ensemble import RandomForestClassifier


# In[9]:


# ランダムフォレストを実行
# n_estimators: 木の組み合わせ数
random_forest = RandomForestClassifier(n_estimators=100,random_state=0).fit(x_train,y_train)

# 分類面で曲線も表現できるようになった。
plt.figure(figsize=(12,8))
plot_decision_boundary(random_forest,x,y)
plot_dataset(x,y)
plt.show()


# In[10]:


# irisデータで実行してみる
from sklearn.datasets import load_iris


# In[11]:


iris = load_iris()
x_iris = iris.data
y_iris = iris.target


# In[12]:


# irisデータセットに保存されている特徴名
iris.feature_names


# In[13]:


# ランダムフォレストを実行
random_forest_iris = RandomForestClassifier(n_estimators=100,random_state=0).fit(x_iris,y_iris)


# In[14]:


# 学習したデータの特徴量の重要度を表示できる。
# 各データ番号は、特徴名にそれぞれ対応する。
random_forest_iris.feature_importances_


# In[15]:


plt.figure(figsize=(12,8))
plt.barh(range(iris.data.shape[1]),random_forest_iris.feature_importances_,height=0.5)
plt.yticks(range(iris.data.shape[1]),iris.feature_names,fontsize=20)
plt.xlabel('Feature Importance',fontsize=20)
plt.show()


# In[16]:


# さらに現実的なデータセットで学習してみる
# titanicデータセットのSurvived(0 or 1)を予測してみる。
import pandas as pd

df = pd.read_csv('titanic_train.csv')
df.head()


# In[17]:


df.info()


# In[18]:


# Survivedを予測することが目標
# RangeIndex: 891なので、891個のデータがある。
# ageの欠損値をageの平均値で埋める
df['Age'] = df['Age'].fillna(df['Age'].mean())
# Embarkedの欠損値をEmbarkedの最頻値で埋める
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])

# ageとEmbarkedのデータ数が891個になったことを確認する
df.info()


# In[19]:


# カテゴリー名を文字列から数値に変換する
# 今回のデータセットでは、sexとEmbarkedを数値に変換する

# 文字データを数字に変更
from sklearn.preprocessing import LabelEncoder

cat_features = ['Sex','Embarked']

for col in cat_features:
    lbl = LabelEncoder()
    df[col] = lbl.fit_transform(list(df[col].values))


# In[20]:


# sexとEmbarkedが数値で表現されていることを確認する
df.head()


# In[21]:


# 以上がデータの前処理
# 使わないデータをdrop
x = df.drop(columns=['PassengerId','Survived','Name','Ticket','Cabin'])

#予測の対象となるデータ
y = df['Survived']


# In[22]:


x.head()


# In[23]:


x_train,x_test,y_train,y_test = train_test_split(x,y,random_state=0)


# In[24]:


# 決定木を実行
tree = DecisionTreeClassifier().fit(x_train,y_train)

# 正答率
tree.score(x_test,y_test)


# In[25]:


# ランダムフォレストを実行
rnd_forest = RandomForestClassifier(n_estimators=500,max_depth=5,random_state=0).fit(x_train,y_train)

# 正答率
rnd_forest.score(x_test,y_test)


# In[26]:


# 機械学習コンテスト用のテストデータを学習させてみる。
test_df = pd.read_csv('titanic_test.csv')
test_df.head()


# In[27]:


test_df.info()


# In[28]:


# 欠損値を埋める
test_df['Age'] = test_df['Age'].fillna(test_df['Age'].mean())
test_df['Fare'] = test_df['Fare'].fillna(test_df['Fare'].mean())
test_df.info()


# In[29]:


# カテゴリー名を文字列から数値に変換する
cat_features = ['Sex','Embarked']

for col in cat_features:
    lbl = LabelEncoder()
    test_df[col] = lbl.fit_transform(list(test_df[col].values))


# In[30]:


# 学習に使用しない特徴を除く
x_pred = test_df.drop(columns=['PassengerId','Name','Ticket','Cabin'])
ID = test_df['PassengerId']


# In[31]:


prediction = rnd_forest.predict(x_pred)
prediction


# In[32]:


# ファイルとして出力するフォーマットを指定
submission = pd.DataFrame({
    'PassengerId': ID,
    'Survived': prediction
})

submission.head()


# In[33]:


# index=False :左端のデータ番号を書き出すファイルに含めない
submission.to_csv('11_titanic_test_output.csv',index=False)


# In[ ]:




