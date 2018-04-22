#! usr/bin/env python
# coding:utf-8

"""
@author: Jinzhong Xu (xujin)
Fri Apr 20 17:03 2018
"""

import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
from scipy import stats
import pandas as pd
import time


def filtration_entropy(data, tail=0.0):
    """
    用以计算数据集的熵，并画出熵图
    :param data: 数据集data，以列表形式表示，每个元素为元组形式
    :param tail: 参数M用以设置熵图最右侧边距
    :return: 得到数据的距离向量，熵向量和熵图
    """

    dist_matrix = np.zeros((len(data), len(data)))  # 用来存储距离矩阵

    tic_matrix = time.time()
    for i in np.arange(len(data)):
        for j in np.arange(len(data)):
            dist_matrix[i][j] = np.linalg.norm(data.iloc[i] - data.iloc[j], ord=2)  # 2-norm
    toc_matrix = time.time()
    print('Matrix time consuming:' + str(toc_matrix - tic_matrix) + 's')

    length = (len(data) - 1) * len(data) / 2  # 只用存储距离矩阵上对角的元素，剔除对角线上的全0值
    dist_vector = [0] * int(length)  # 定义一个数组，用来存储上对角距离值（可能包含0）
    flag = 0

    tic_vector = time.time()
    for i in np.arange(len(data) - 1):
        for j in np.arange(i + 1, len(data)):
            dist_vector[flag] = dist_matrix[i][j]
            flag += 1
    dist_vector = np.array(dist_vector)  # 把距离向量list转化为数组，用以剔除0和重复值，最后排序
    dist_vector = dist_vector[np.nonzero(dist_vector)]  # 把可能出现的0值剔除掉
    dist_vector = np.unique(dist_vector)  # 剔除重复值并排序
    toc_vector = time.time()
    print('vector time consuming: ' + str(toc_vector - tic_vector) + 's')

    #     print("Distance Matrix Convert to Distance Vector: {0}".format(distVector))  # test 打印距离向量

    plt.figure(figsize=(12, 8))
    point_in_circle = np.array([1] * len(data))  # 记录以每个点为圆心的圆不同半径包含的周围点的个数
    entropy = [0.0] * (len(dist_vector))  # 不同半径时，计算得到的熵值

    tic_circle = time.time()
    # 去除首尾两点后的距离向量画除的熵图
    for i in np.arange(1, len(dist_vector)):  # 去除首尾距离后，其他距离值对应的熵值
        for j in np.arange(len(data)):  # 每个点在不同半径下的圆包含的点
            point_in_circle[j] = np.sum(dist_matrix[j] <= dist_vector[i])
        ratio = np.array(point_in_circle) / np.sum(point_in_circle)  # 在此时的半径下，密度点的比率
        entropy[i] = sp.stats.entropy(ratio, base=np.e)  # 计算此时的半径下，熵值
        plt.plot([dist_vector[i - 1], dist_vector[i]], [entropy[i], entropy[i]], 'r')
    toc_circle = time.time()
    print('circle time consuming: ' + str(toc_circle - tic_circle) + 's')

    tic_head_tail = time.time()
    # 当距离小于距离向量首点（最小距离）时的熵图
    entropy[0] = np.log(len(data))
    plt.plot([0, dist_vector[0]], [entropy[0], entropy[0]], 'r')

    # 当距离大于距离向量末尾（最大距离）时的熵图
    entropy[-1] = np.log(len(data))
    plt.plot([dist_vector[-1], dist_vector[-1] + tail], [entropy[-1], entropy[-1]], 'r')
    toc_head_tail = time.time()
    print('headTail time consuming: ' + str(toc_head_tail - tic_head_tail) + 's')

    #     print("Complex Entropy Vector: {}".format(entropy))
    print("Min(h) = {}".format(np.min(entropy)))
    print("Max(h) = {}".format(np.max(entropy)))
    print("Ave(h) = {}".format(np.sum(entropy) / len(entropy)))
    plt.title("$Filtration\ Entropy\ of\ Simplicial\ Complexes(FESC)$", fontsize=20)
    plt.xlabel("$Distance\ Parameter(or\ time): \epsilon$", fontsize=20)
    plt.ylabel("$Filtration\ Entropy: h_X(\epsilon)$", fontsize=20)

    plt.show()
