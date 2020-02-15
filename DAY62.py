# -*- coding: utf-8 -*-
"""
觀察雙同心圓使用 t-SNE 在不同困惑度 (perplexity) 下的分群效果
因為非監督模型的效果, 較難以簡單的範例看出來
所以非監督偶數日提供的範例與作業, 主要目的在於觀察非監督模型的效果,
同學只要能感受到模型效果即可, 不用執著於搞懂程式的每一個部分 
on Sat Feb 15 15:54:46 2020

@author: tribology2020
"""
# 載入套件
import numpy as np     #calculate 
import matplotlib.pyplot as plt   #draw

from matplotlib.ticker import NullFormatter  #tick 刻度  #No labels on the ticks
from sklearn import manifold, datasets
from time import time

# 設定模型與繪圖參數
n_samples = 300
n_components = 2
(fig, subplots) = plt.subplots(2, 5, figsize=(15, 6))
perplexities = [4, 6, 9, 14, 21, 30, 45, 66, 100]

# 設定同心圓資料點 
X, y = datasets.make_circles(n_samples=n_samples, factor=.05, noise=.05)
#                                               Scale factor between inner and outer circle.
red = y == 0 # 將 y 為 0 的 index set 存成變數 red
green = y == 1 # 將 y 為 1 的 index set 存成變數 green

# 繪製資料原圖
ax = subplots[0][0]
ax.set_title("Original")
ax.scatter(X[red, 0], X[red, 1], c="r")
ax.scatter(X[green, 0], X[green, 1], c="g")
ax.xaxis.set_major_formatter(NullFormatter())
ax.yaxis.set_major_formatter(NullFormatter())
plt.axis('tight')

# 繪製不同 perplexity 下的 t-SNE 分群圖
for i, perplexity in enumerate(perplexities):
    if i<4:
        ax = subplots[0][i+1]
    else:
        ax = subplots[1][i-4]

    t0 = time()
    tsne = manifold.TSNE(n_components=n_components, init='random',
                         random_state=0, perplexity=perplexity)
    Y = tsne.fit_transform(X)
    t1 = time()
    print("circles, perplexity=%d in %.2g sec" % (perplexity, t1 - t0))
    ax.set_title("Perplexity=%d" % perplexity)
    ax.scatter(Y[red, 0], Y[red, 1], c="r")
    ax.scatter(Y[green, 0], Y[green, 1], c="g")
    ax.xaxis.set_major_formatter(NullFormatter())
    ax.yaxis.set_major_formatter(NullFormatter())
    ax.axis('tight')
    
plt.show()