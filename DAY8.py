# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 16:06:32 2019

@author: 91812
試著顯示特定欄位的基礎統計數值 (In[4], Out[4], Hint : describe())
試著顯示特定欄位的直方圖 (In[5], Out[5], Hint : .hist())
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# 設定 data_path
dir_data = 'C:/Users/91812/Documents/GitHub/Marathon/downloads/'
f_app_train = os.path.join(dir_data, 'D7house_train.csv.gz')
app_train = pd.read_csv(f_app_train)
A=app_train.MSSubClass.mean()
B=app_train.MSSubClass.std()
p = app_train.plot.hist(x='Id',y='MSSubClass',color='Blue',label='MSSubClass')
#plt.show()