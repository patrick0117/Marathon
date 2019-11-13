# -*- coding: utf-8 -*-
"""
Created on Wed Nov 13 09:57:34 2019

@author: 91812
"""

import os
import numpy as np
import pandas as pd

dir_data = 'C:/Users/91812/Documents/GitHub/Marathon/downloads/'

f_app = os.path.join(dir_data, 'application_test.csv')

print('Path of read in data: %s' % (f_app))

app_train = pd.read_csv(f_app)

#app_train.head()