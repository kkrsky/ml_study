#!/usr/bin/env python
# coding: utf-8

# In[1]:


# setup
import mglearn
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


mglearn.plots.plot_knn_classification(n_neighbors=3)


# In[3]:


# mglearnライブラリのデータセットを使用
x,y = mglearn.datasets.make_forge()


# In[4]:


x


# In[5]:


# 26 person 2 height and weight data.
x.shape


# In[6]:


# answer label data
y


# In[7]:


y.shape


# In[8]:


# drow scatter data
# mglearn.discrete_scatter(x axis, y axis, answer label data)
# データの可視化
# discreate_scatter: 行と列、グルーピングを指定する
# 第１引数： 散布図に描写する各データのX値
# 第２引数： 散布図に描写する各データのY値
# 第３引数： 散布図に描写する各データのLABEL 
mglearn.discrete_scatter(x[:,0],x[:,1],y)
plt.show()


# In[9]:


#最近傍法の準備
from sklearn.neighbors import KNeighborsClassifier

# 自動で訓練データとテストデータを生成するライブラリ
from sklearn.model_selection import train_test_split


# In[10]:


#データを訓練用（train）とテスト用（test）に分割
# random_state: 乱数シード。0にセットすることで同じ結果が得られる
x_train,x_test,y_train,y_test = train_test_split(x,y,random_state=0)


# In[11]:


#データを目視
x_train


# In[12]:


# x_trainの行数と列数
x_train.shape


# In[13]:


# x_testの行数と列数
x_test.shape


# In[14]:


#最近傍法の実装
clf = KNeighborsClassifier(n_neighbors=3)

#訓練データで学習の実行
clf.fit(x_train,y_train)


# In[15]:


#テストデータで予測の実行
clf.predict(x_test)


# In[16]:


#実際のラベルと比較
y_test


# In[17]:


#正しさの割合を出してくれる関数を実行してみる
clf.score(x_test,y_test)


# In[18]:


#データを丸めてみやすくする
round(clf.score(x_test,y_test),3)


# In[19]:


#[他の方法] 正規表現を使ってみる
print("{:.2f}".format(clf.score(x_test,y_test)))


# In[20]:


#Kの値を変えてみる (n数を変えても今回のデータでは変化しなかった)
for n_neighbors in range(1,16):
    KNeighborsClassifier(n_neighbors=n_neighbors).fit(x_train,y_train)
    print("Test set accuracy : n_neighbors={},{:.2f}".format(n_neighbors,clf.score(x_test,y_test)))


# In[21]:


#Kの値を変えてみた時、境界線がどう変化するのか図で見てみる
# →kの値が増えると境界が滑らかになる


# plt.subplots (グラフの行､ 列､ 大きさ) :グラフを書くためのキャンパスを用意。1行5列のキャンバスを生成する。
# イメージとして、figはブラウザー, axisはブラウザータブ
fig,axes = plt.subplots(1,5,figsize=(15,3))

for n_neighbors,ax in zip([1,3,5,10,15],axes):
    # モデルの作成＆学習
    clf = KNeighborsClassifier(n_neighbors=n_neighbors).fit(x_train,y_train)
    
    # mglearn.plots.plot_2d_separator(分類器, 入力データ) :グラフを2つの領域に分割する関数
    # fill:True 2つの領域を塗りつぶす
    # ax: 表示する軸を指定
    # alpha: 透明度
    mglearn.plots.plot_2d_separator(clf,x,fill=True,ax=ax,alpha=0.5)
    
    # 散布図を描画
    mglearn.discrete_scatter(x[:,0],x[:,1],y,ax=ax)
    
    # グラフタイトルを描画
    ax.set_title("{} neighbors".format(n_neighbors))

#plt.show()は表示に少々時間がかかるかも
plt.show()


# In[22]:


# 以下、skleanのデータセットを利用して学習を行う

from sklearn.datasets import load_breast_cancer


# In[23]:


cancer = load_breast_cancer()


# In[24]:


# データセットの中身を確認
# data, target, frame, target_names, DESCR, feature_names, filename, data_moduleのラベルを確認できる

cancer


# In[25]:


# cancer.dataは、569行30列存在する(569人分のデータがあり、それぞれ30項目のデータがある)
cancer.data.shape


# In[26]:


# 上記の各人のデータの30項目の中身

cancer.feature_names


# In[27]:


# 癌患者のラベル(1:癌)

cancer.target


# In[28]:


# 癌データの0列のデータ('mean radius'), 1列のデータ('mean texture')について、散布図で色分けをしてみる

mglearn.discrete_scatter(cancer.data[:,0],cancer.data[:,1],cancer.target)


# In[29]:


# pltライブラリで散布図を描画してみる(ラベルデータで色分けさせるためには、工夫が必要)

plt.scatter(cancer.data[:,0],cancer.data[:,1])
plt.show()


# In[30]:


# cancer.dataの0列全データに対して、cancer.target==0の条件と一致する値だけ取り出す
x_1_blue=cancer.data[:,0][cancer.target==0]
# 同上
x_2_blue=cancer.data[:,1][cancer.target==0]

# 青データのみ描画してみる
plt.scatter(x_1_blue,x_2_blue)


# In[31]:


# cancer.dataの0列全データに対して、cancer.target==1の条件と一致する値だけ取り出す
x_1_red=cancer.data[:,0][cancer.target==1]
# 同上
x_2_red=cancer.data[:,1][cancer.target==1]

# 赤データのみ描画してみる
plt.scatter(x_1_red,x_2_red,c='red')


# In[32]:


# 赤青を重ねて描画
plt.scatter(x_1_blue,x_2_blue,c='blue')
plt.scatter(x_1_red,x_2_red,c='red')


# In[33]:


# 以上がデータの確認
# データを学習用に加工する(今回はデータセットで既に整理済みなので省略)
# モデルを試す

# stratify:母集団のラベルの割合を保ったまま、トレーニングとテストのデータに分割する。
x_train, x_test, y_train, y_test = train_test_split(cancer.data[:,0:2],cancer.target,stratify=cancer.target, random_state=0)


# In[34]:


x_train.shape


# In[35]:


# K近接法の学習を実行
clf=KNeighborsClassifier(n_neighbors=3).fit(x_train,y_train)

# テスト用データを予測した結果
clf.predict(x_test)


# In[36]:


# 正解のラベル
y_test


# In[37]:


# 予測結果と正解ラベルで一致する場合はTrueとして表示
clf.predict(x_test)==y_test


# In[38]:


# 正解率
round(clf.score(x_test,y_test),3)


# In[39]:


cancer.data[:,0:2]


# In[40]:


print(clf)


# In[41]:


# # K近接法の学習を実行
clf = KNeighborsClassifier(n_neighbors=3).fit(x_train,y_train)

# 描画用キャンバス生成
fig,axes = plt.subplots(1,1,figsize=(15,3))

# グラフを2つの領域に分割する関数
mglearn.plots.plot_2d_separator(clf,cancer.data[:,0:2] , fill=True, alpha=0.5)

# 散布図を描画
# mglearn.discrete_scatter(cancer.data[:,0], cancer.data[:,1], cancer.target, ax=axes)


plt.show()


# In[42]:


cancer.target

