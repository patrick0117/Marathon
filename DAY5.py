# -*- coding: utf-8 -*-
"""
Created on Sat Nov 16 10:56:16 2019

@author: 91812
"""

import pandas as pd
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import cv2
import scipy.io as sio
import json    #可輸出jason

data = {'國家':np.random.rand(1) , 
        '人口':np.random.rand(1) }
data = pd.DataFrame()
'''
img1 = skio.imread('C:/Users/91812/Documents/GitHub/Marathon/downloads/D5.jpg')
#skio import 有問題
plt.imshow(img1)
plt.show()
'''
##開圖
img1 = Image.open('C:/Users/91812/Documents/GitHub/Marathon/downloads/D5.jpg') # 這時候還是 PIL object
img1 = np.array(img1)
plt.imshow(img1)
plt.show()
###開圖
img2 = cv2.imread('C:/Users/91812/Documents/GitHub/Marathon/downloads/D5.jpg')
plt.imshow(img2)
plt.show()
#開圖
img3 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
plt.imshow(img3)
plt.show()
#
N_times = 1000
#for _ in range(N_times)   迴圈
#im = np.array([np.array(Image.open('C:/Users/91812/Documents/GitHub/Marathon/downloads/D5.jpg')) for _ in range(N_times)])
##存檔

sio.savemat(file_name='C:/Users/91812/Documents/GitHub/Marathon/downloads/D5.mat', mdict={'img': img1})
#mat_arr = sio.loadmat('data/examples/example.mat')

with open("C:/Users/91812/Documents/GitHub/Marathon/downloads/D5.txt", encoding="utf-8") as f:
    data = f.readlines()
print(data)

dada = []
with open("C:/Users/91812/Documents/GitHub/Marathon/downloads/D5.txt", 'r') as f:
    for line in f:
        line = line.replace('\n', '').split(',') # 將每句最後的 /n 取代成空值後，再以逗號斷句
        dada.append(line)
df = pd.DataFrame(data[1:])
df.columns = data[0]

#存成np
np.save(arr=array, file='data/example.npy')




