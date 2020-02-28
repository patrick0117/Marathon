# -*- coding: utf-8 -*-
"""
Created on Fri Feb 28 20:17:52 2020

@author: 88697
"""

import numpy as np
from numpy import *
import matplotlib.pylab as plt

#Sigmoid 數學函數表示方式
#sigmoid = lambda x: 1 / (1 + np.exp(-x))
def sigmoid(x):
    return (1 / (1 + np.exp(-x)))

#Sigmoid 微分
def dsigmoid(x):
    return (x * (1 - x))

# linespace generate an array from start and stop value
# with requested number of elements. Example 10 elements or 100 elements.
x = plt.linspace(-10,10,100)

# prepare the plot, associate the color r(ed) or b(lue) and the label 
plt.plot(x, sigmoid(x), 'b', label='linspace(-10,10,10)')

# Draw the grid line in background.
plt.grid()

# 顯現圖示的Title
plt.title('Sigmoid Function')

# 顯現 the Sigmoid formula
plt.text(4, 0.8, r'$\sigma(x)=\frac{1}{1+e^{-x}}$', fontsize=15)

#resize the X and Y axes
plt.gca().xaxis.set_major_locator(plt.MultipleLocator(1))
plt.gca().yaxis.set_major_locator(plt.MultipleLocator(0.1))

# create the graph
plt.show()


#Softmax 數學函數表示方式
def softmax(x):
     return np.exp(x) / float(sum(np.exp(x)))

#x=np.arange(0,1.0,0.01)
x = plt.linspace(-5,5,100)

#resize the X and Y axes
plt.gca().xaxis.set_major_locator(plt.MultipleLocator(1))
#plt.gca().yaxis.set_major_locator(plt.MultipleLocator(1))

#列印所有Softmax 值並輸出成一陣列
print(softmax(x))
plt.plot(x, softmax(x), 'r')
plt.show()