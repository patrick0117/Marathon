# -*- coding: utf-8 -*-
"""
Created on Mon Dec  2 18:14:03 2019
pandas.cut 的等寬劃分效果 (In[3], Out[4])
pandas.qcut 的等頻劃分效果 (In[5], Out[6])
@author: 91812
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 初始設定 Ages 的資料
ages = pd.DataFrame({"age": [18,22,25,27,7,21,23,37,30,61,45,41,9,18,80,100]})

# 新增欄位 "equal_width_age", 對年齡做等寬劃分，分成四組
ages["equal_width_age"] = pd.cut(ages["age"], 4)
#ages["equal_width_age"].value_counts()

# 新增欄位 "equal_freq_age", 對年齡做等頻劃分
ages["equal_freq_age"] = pd.qcut(ages["age"], 4)
#ages["equal_freq_age"].value_counts()

#HW
#新增一個欄位 customized_age_grp，把 age 分為 (0, 10], (10, 20],
# (20, 30], (30, 50], (50, 100] 五組，'(' 表示不包含, ']' 表示包含
bins = [0,10,20,30,50,100]
ages["nonequal_width_age"] = pd.cut(ages["age"], bins)