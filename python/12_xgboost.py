#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


# In[2]:


train_df = pd.read_csv('titanic_train.csv')
test_df = pd.read_csv('titanic_test.csv')


# In[3]:


train_df.head()


# In[4]:


# trainとtestデータの違いは、正解ラベル(Survived)の有無
test_df.head()


# In[5]:


# 今回はまとめてデータを処理(欠損値の修正など)したいので、データを結合する
# 基本的にテストデータとトレーニングデータをまとめて処理を行うことは欠損値を正しく埋めることができない可能性があるのであまりよろしくない。
all_df = pd.concat((train_df.loc[:,'Pclass':'Embarked'],test_df.loc[:,'Pclass':'Embarked']))

# 欠損値を確認(Age, Fare, Cabin, Embarked)
all_df.info()


# In[6]:


# 欠損値を埋める
all_df['Age'] = all_df['Age'].fillna(all_df['Age'].mean())
all_df['Fare'] = all_df['Fare'].fillna(all_df['Fare'].mean())
all_df['Embarked'] = all_df['Embarked'].fillna(all_df['Embarked'].mode()[0])

all_df.info()


# In[7]:


# カテゴリ名(文字列)を数値に変換する(SexとEmbarked)
all_df.head()


# In[8]:


# 文字データを数字に変更
cat_features = ['Sex','Embarked']

# LabelEncoderを適用
for col in cat_features:
    lbl = LabelEncoder()
    all_df[col] = lbl.fit_transform(list(all_df[col].values))

all_df.head()


# In[9]:


# 今回学習に使わないデータを除く
all_df = all_df.drop(columns=['Name','Ticket','Cabin'])

all_df.head()


# In[10]:


print(train_df.shape[0])
train = all_df[:train_df.shape[0]]
test = all_df[train_df.shape[0]:]

train.info()


# In[11]:


# 正解ラベルを取得
y = train_df['Survived']

ID = test_df['PassengerId']


# In[12]:


x_train,x_test,y_train,y_test = train_test_split(train,y,random_state=0)


# In[13]:


# 学習ライブラリxgboost
import xgboost as xgb

# xgboostのパラメータ
params = {
    # どのような分析を行うのか。 binary=0 or 1を出力
    "objective":"binary:logistic",
    
    # 評価方法は正解率を用いる
    "eval_metric":"auc",
    
    "eta":0.1,
    "max_depth":6,
    "subsample":1,
    "colsample_bytree":1,
}

dtrain = xgb.DMatrix(x_train,label=y_train)
dtest = xgb.DMatrix(x_test,label=y_test)

# 学習実行
model = xgb.train(params=params,
                  dtrain=dtrain,num_boost_round=100,
                  early_stopping_rounds=10, # 10回連続で数値が更新されなかったら計算を終了する
                  evals=[(dtest,'test')])


# In[14]:


prediction = model.predict(xgb.DMatrix(test),ntree_limit=model.best_ntree_limit)

# 正解の確率を表示
prediction


# In[15]:


# 正解の確率が0.5未満であれば0とする
prediction = np.where(prediction < 0.5,0,1)

prediction


# In[16]:


# csvファイル作成
submission = pd.DataFrame({
    'PassengerId': ID,
    'Survived': prediction
})

submission.to_csv('12_titanic_test_output.csv',index=False)


# In[ ]:




