# coding: utf-8

# KMeans clustering
# https://tech-clips.com/article/421516

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Wholesale customers Data Set (卸売業者の顧客データ)
cust_df = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/00292/Wholesale%20customers%20data.csv")

del(cust_df['Channel'])
del(cust_df['Region'])
cust_array = np.array([
    cust_df['Fresh'].tolist(),
    cust_df['Milk'].tolist(),
    cust_df['Grocery'].tolist(),
    cust_df['Frozen'].tolist(),
    cust_df['Milk'].tolist(),
    cust_df['Detergents_Paper'].tolist(),
    cust_df['Delicassen'].tolist()
], np.int32)
cust_array = cust_array.T

pred = KMeans(n_clusters=4).fit_predict(cust_array)

# クラスタリングの紐付け
cust_df['cluster_id'] = pred
# print(cust_df)

# 各クラスタに属する「サンプル数の分布」、各クラスタの「各部門商品の購買額の平均値」
value_counts = cust_df['cluster_id'].value_counts()
# print(value_counts)

mean0 = cust_df[cust_df['cluster_id']==0].mean()
mean1 = cust_df[cust_df['cluster_id']==1].mean()
mean2 = cust_df[cust_df['cluster_id']==2].mean()
mean3 = cust_df[cust_df['cluster_id']==3].mean()
# print(mean0)

# グラフ化
clusterinfo = pd.DataFrame()
for i in range(4):
    clusterinfo['cluster' + str(i)] = cust_df[cust_df['cluster_id'] == i].mean()
clusterinfo = clusterinfo.drop('cluster_id')
my_plot = clusterinfo.T.plot(kind='bar', stacked=True, title="Mean Value of 4 Clusters")
my_plot.set_xticklabels(my_plot.xaxis.get_majorticklabels(), rotation=0)

# my_plot.show()


# ・クラスター番号 = 0 に分類された顧客(293人)は、全体的に購買額が低い傾向にあります。
# ・クラスター番号 = 1 に分類された顧客(63人)は、Fresh(生鮮食品)の購買額が比較的高いことがわかります。
# ・クラスター番号 = 2 に分類された顧客(77人)は、Grocery(食料雑貨品)とDetergents_Paper(衛生用品と紙類)の購買額が比較的高いことがわかります。
# ・クラスター番号 = 3 に分類された顧客(7人)は、全てのジャンルで購買額が高いと言えます。

#