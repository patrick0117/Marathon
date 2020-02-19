# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 11:34:18 2020

@author: tribology2020
"""
import keras
from keras import backend as K
from keras.layers import Layer
print(keras.__version__)

#  GPU加速测试, True - Windows用戶得到True也沒有關係，因為Anaconda中已經內置了MKL加速庫
import numpy 
id(numpy.dot) == id(numpy.core.multiarray.dot)
#檢查Keras float 
K.floatx()

#設定浮點運算值
K.set_floatx('float16')
K.floatx()
