# -*- coding: utf-8 -*-
"""
Created on Tue Jan  7 19:10:12 2020
 1.
 隨機森林中的每一棵樹，是希望能夠

(X)沒有任何限制，讓樹可以持續生長 (讓樹生成很深，讓模型變得複雜)
(O)不要過度生長，避免 Overfitting
 2.
 假設總共有 N 筆資料，每棵樹用取後放回的方式抽了總共 N 筆資料生成，
 請問這棵樹大約使用了多少 % 不重複的原資料生成?
 單一個個體被抽重的機率是0.632
 大約使用63.2%之原始資料不重複生成
@author: tribology2020
"""

