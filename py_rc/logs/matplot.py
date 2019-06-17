#!/usr/bin/env python
# encoding: utf-8
'''
@author: tianxiaomo
@license: (C) Apache.
@contact: huguanghao520@gmail.com
@software: PyCharm
@file: matplot.py
@time: 2019/6/15 15:30
@desc:
'''
import matplotlib.pyplot as plt
import numpy as np

# log_path = 'log_2019-06-14_11-56-36.log'
log_path = 'log_2019-06-15_09-44-59.log'

data = open(log_path, 'r', encoding='utf-8').readlines()

t_loss = []
for i in data:
    if 'loss           :' in i:
        t_loss.append(float(i.split(':')[-1]))

plt.plot(np.arange(0,len(t_loss[:1000])),np.asarray(t_loss[:1000]))
plt.show()

v_loss = []
for i in data:
    if 'val_loss' in i:
        v_loss.append(float(i.split(':')[-1]))

plt.plot(np.arange(0,len(v_loss[:1000])),np.asarray(v_loss[:1000]))
plt.show()


t_acc = []
for i in data:
    if 'metrics        : [' in i:
        t_acc.append(float(i.split(',')[-1][:-2]))
plt.plot(np.arange(0,len(t_acc[:1000])),np.asarray(t_acc[:1000]))
plt.show()


v_acc = []
for i in data:
    if 'val_metrics' in i:
        v_acc.append(float(i.split(',')[-1][:-2]))
plt.plot(np.arange(0,len(v_acc[:1000])),np.asarray(v_acc[:1000]))
plt.show()

