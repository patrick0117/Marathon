# -*- coding: utf-8 -*-
"""
Created on Sat Nov 30 14:29:22 2019

@author: 91812
"""
import numpy as np
np.random.seed(1)

import matplotlib
import matplotlib.pyplot as plt
# 隨機生成兩組 1000 個介於 0~50 的數的整數 x, y, 看看相關矩陣如何
x = np.random.randint(0, 50, 1000)
y = np.random.randint(0, 50, 1000)

# 呼叫 numpy 裡的相關矩陣函數 (corrcoef)
np.corrcoef(x, y)


# 隨機生成 1000 個介於 0~50 的數 x
x = np.random.randint(0, 50, 1000)

# 這次讓 y 與 x 正相關，再增加一些雜訊
y = -x +np.random.normal(0, 10, 1000)

# 再次用 numpy 裡的函數來計算相關係數
np.corrcoef(x, y)
plt.scatter(x, y)
print(np.corrcoef(x, y))