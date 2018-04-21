#! usr/bin/env python
# coding:utf-8

"""
@author: Jinzhong Xu
@test
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
from entropy import filtration_entropy

# plt.rcParams['font.sans-serif'] = ['SimHei']        # 用来正常显示中文标签
# plt.rcParams['axes.unicode_minus'] = False        # 用来正常显示负号

circle100 = pd.read_csv('c100.csv')
circle100 = circle100[['x1', 'x2']]
# print(circle100.head())
# plt.plot(circle100['x1'], circle100['x2'], 'k.')  # 画出数据的分布图
# plt.title(u'circle with Guass noise')
# plt.show()

tic = time.time()
filtration_entropy(circle100)
toc = time.time()

print("Total time consuming :" + str(toc - tic) + 's')

