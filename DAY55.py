# -*- coding: utf-8 -*-
"""
Created on Tue Feb  4 19:59:52 2020

@author: tribology2020
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
# Though the following import is not directly being used, it is required
# for 3D projection to work
from mpl_toolkits.mplot3d import Axes3D

from sklearn.cluster import KMeans
from sklearn import datasets

np.random.seed(5)
#載入 toy example 資料集
toy = datasets.make_blobs(centers=3, n_features=4)
X = toy[0]
y = toy[1]
#設定 模型 估計參數
estimators = [('k_means_8', KMeans(n_clusters=8)),
              ('k_means_3', KMeans(n_clusters=3)),
              ('k_means_bad_init', KMeans(n_clusters=3, n_init=1,
                                               init='random'))]
#init  : 設定初始化方式
#n_init  : 使用不同 centroid seeds 運行 k-means 算法的時間
              
#資料建模 並 視覺化 結果      
fignum = 1
titles = ['8 clusters', '3 clusters', '3 clusters, bad initialization']
for name, est in estimators:
    fig = plt.figure(fignum, figsize=(4, 3))# 設定圖片大小
    ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)
    #參數??
    ## fit data
    est.fit(X)
    #訓練
    labels = est.labels_

    ax.scatter(X[:, 3], X[:, 0], X[:, 2],
               c=labels.astype(np.float), edgecolor='k')

    ax.w_xaxis.set_ticklabels([])
    ax.w_yaxis.set_ticklabels([])
    ax.w_zaxis.set_ticklabels([])
    ax.set_title(titles[fignum - 1])
    ax.dist = 12
    fignum = fignum + 1

# Plot the ground truth
fig = plt.figure(fignum, figsize=(4, 3))
ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)

for name, label in [('cls0', 0),
                    ('cls1', 1),
                    ('cls2', 2)]:
    ax.text3D(X[y == label, 3].mean(),    #繪製3D圖形
              X[y == label, 0].mean(),
              X[y == label, 2].mean() + 2, name,
              horizontalalignment='center',
              bbox=dict(alpha=.2, edgecolor='w', facecolor='w'))
# Reorder the labels to have colors matching the cluster results
y = np.choose(y, [1, 2, 0]).astype(np.float) #將原本 label 順序的(0 1 2)改成(1 2 0) 
ax.scatter(X[:, 3], X[:, 0], X[:, 2], c=y, edgecolor='k') #畫散點圖，後面的參數用來調整顏色

ax.w_xaxis.set_ticklabels([])
ax.w_yaxis.set_ticklabels([])
ax.w_zaxis.set_ticklabels([])
ax.set_title('Ground Truth')
ax.dist = 12