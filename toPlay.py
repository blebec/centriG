#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  2 11:24:28 2020

@author: cdesbois
"""

import matplotlib.pyplot as plt

#%%
plt.close('all')

fig = plt.figure()
axes = []
ax = fig.add_subplot(421)
axes.append(ax)
for i in range(3, 8, 2):
    print(i)
    axes.append(fig.add_subplot(4, 2, i))
for i in range(2, 9, 2):
    axes.append(fig.add_subplot(4, 2, i))

fig, axes = plt.subplots(nrows=4, ncols=2)
axes = axes.T.flatten().tolist()
axr = axes[0]
for ax in axes[0::2]:
    ax.get_shared_y_axes().join(ax, axr)
axr = axes[1]
for ax in axes[1::2]:
    ax.get_shared_y_axes().join(ax, axr)
