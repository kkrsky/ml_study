#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd


# In[2]:


df = pd.read_csv('titanic_train.csv')
df.head()


# In[3]:


# 891個のデータと12個の特徴量があることを確認。
df.info()


# In[4]:


# 欠損値を埋める
df['Age'] = df['Age'].fillna(df['Age'].mean())
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])

df.head()


# In[5]:


df.info()


# In[6]:


# 予測用データ
x = df.drop(columns=['PassengerId','Survived','Name','Ticket','Cabin'])

# 正解データ
y = df['Survived']


# In[7]:


x.head()


# In[8]:


# 文字列のカテゴリーデータを数値に変換する

from sklearn.preprocessing import LabelEncoder


# In[9]:


cat_features = ['Sex','Embarked']

for col in cat_features:
    lbl = LabelEncoder()
    x[col] = lbl.fit_transform(list(df[col].values))

x.head()


# In[10]:


#9 SVC
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split


# In[11]:


x_train,x_test,y_train,y_test = train_test_split(x,y,random_state=0)

# SVC学習実行
svm = SVC().fit(x_train,y_train)

# 正解率
svm.score(x_test,y_test)


# In[12]:


# スケーリング対象となり得るか、データを確認してみる

# カテゴリーなデータであることがわかる
x['Pclass'].value_counts()


# In[13]:


# カテゴリーなデータであることがわかる
x['SibSp'].value_counts()


# In[14]:


# スケーリング実施
# 連続的な値であるAgeとFareにStandardScalerを実施

from sklearn.preprocessing import StandardScaler


# In[15]:


num_features = ['Age','Fare']
for col in num_features:
    scaler = StandardScaler()
    x[col] = scaler.fit_transform(np.array(df[col].values).reshape(-1,1))

# 平均0, 分散1のデータにスケーリング
x.head()


# In[16]:


# ダミー変数を用意
# 数値1,2,3を用いると1<2<3の情報も含まれてしまうので、分類データをダミー変数化することで分類として情報を用意する。
x = pd.get_dummies(x,columns=['Pclass','SibSp','Embarked'])

x.head()


# In[17]:


# drop_first=Trueの効果を確認する言うのデータを生成
x_drop = df.drop(columns=['PassengerId','Survived','Name','Ticket','Cabin'])

# カテゴリーを数値化
cat_features = ['Sex','Embarked']
for col in cat_features:
    lbl = LabelEncoder()
    x_drop[col] = lbl.fit_transform(list(df[col].values))

# 平均0, 分散1のデータにスケーリング
num_features = ['Age','Fare']
for col in num_features:
    scaler = StandardScaler()
    x_drop[col] = scaler.fit_transform(np.array(df[col].values).reshape(-1,1))
    
# drop_first=Trueと設定するとダミー変数が1つ減る。＝多重共線性。
x_drop = pd.get_dummies(x_drop,columns=['Pclass','SibSp','Embarked'], drop_first=True)

x_drop.head()


# In[18]:


# スケーリングしたデータを再度学習をしてみる
x_train,x_test,y_train,y_test = train_test_split(x,y,random_state=0)

svm = SVC().fit(x_train,y_train)

svm.score(x_test,y_test)


# In[19]:


# drop_first=Trueでスケーリングしたデータを再度学習をしてみる
x_drop_train,x_drop_test,y_drop_train,y_drop_test = train_test_split(x_drop,y,random_state=0)

svm_drop = SVC().fit(x_drop_train,y_drop_train)

svm_drop.score(x_drop_test,y_drop_test)


# In[20]:


# SVCの学習パラメータ調整してみる

# ペナルティ項 Cをいじってみる、デフォルトは１
svm = SVC(C=10).fit(x_train,y_train)
svm.score(x_test,y_test)


# In[21]:


svm = SVC(C=100).fit(x_train,y_train)
svm.score(x_test,y_test)


# In[22]:


# Cのパラメータを振ってみる

best_score = 0
for C in [1,10,100,1000,10000]:
    svm = SVC(C=C).fit(x_train,y_train)
    
    # 本来、テストデータをパラメータ調整に利用するのはよろしくない。->検証用のデータを用意する必要がある。
    score = svm.score(x_test,y_test)
    if score > best_score:
        best_score = score
        best_parameter = C

print(best_parameter)
print(best_score)


# In[23]:


# 新たに検証用データセットを作成
x_train_,x_validation,y_train_,y_validation = train_test_split(x_train,y_train,random_state=0)


# In[24]:


best_score = 0
for C in [1,10,100,1000,10000]:
    svm = SVC(C=C).fit(x_train_,y_train_)
    score = svm.score(x_validation,y_validation)
    if score > best_score:
        best_score = score
        best_parameter = C
        
# 再訓練 (グリッドサーチ手法)
svm_best = SVC(C=best_parameter).fit(x_train,y_train)
svm_best.score(x_test,y_test)


# In[25]:


# 上記のグリッドサーチ処理をsklearnライブラリ内で容易に行える。
from sklearn.model_selection import GridSearchCV


# In[26]:


param = {
'C':[0.01,0.1,1,10,100]
}

# GridSearchCV(学習モデル,param_grid=調整するパラメータオブジェクト,cv=交差検証回数)
grid_search = GridSearchCV(SVC(),param_grid=param,cv=5)
grid_search.fit(x_train,y_train)

grid_search.best_params_


# In[27]:


grid_search.score(x_test,y_test)


# In[28]:


# 元データも交差検証する
from sklearn.model_selection import cross_val_score


# In[29]:


scores = cross_val_score(GridSearchCV(SVC(),param_grid=param,cv=5),x,y,cv=5)
print(scores)


# In[30]:


scores.mean()


# In[ ]:




