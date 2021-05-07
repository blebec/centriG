#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  7 15:12:49 2021

@author: cdesbois
"""

from math import ceil, floor
import os

from importlib import reload

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
#from mpl_toolkits.axes_grid1.inset_locator import inset_axes

import config
import general_functions as gfunc
import load.load_data as ldat

# import itertools


# nb description with pandas:
pd.options.display.max_columns = 30

#===========================
# global setup
# NB fig size : 8.5, 11.6 or 17.6 cm

anot = True           # to draw the date and name on the bottom of the plot
std_colors = config.std_colors()
speed_colors = config.speed_colors()
plt.rcParams.update(config.rc_params())
paths = config.build_paths()
os.chdir(paths['pg'])




#%%%
latEngy50_v_df = ldat.load_cell_contributions(rec='vm', amp='engy', age='new')
latEngy50_s_df = ldat.load_cell_contributions(rec='spk', amp='engy', age='new')


loop = [{'time' : 't'}, {'engy' : 'e'} ]

df = latEner50_v_df
cols = [_ for _ in df.columns if 'time' in _]
sigs = [_ for _ in cols if '_sig' in _]
cols = [_ for _ in cols if '_sig' not in _]

# all cells
resdf = df[cols].agg(['mean', 'std'])

temp = df[cols].agg(['mean', 'std']).T
temp = temp.reset_index()
temp['index'].apply(lambda st: st.split('_')[0])
temp.index = temp['index'].apply(lambda st: st.split('_')[0])
del temp['index']
temp.columns = ['vm_t_' + st for st in temp.columns]

# sigs
col = cols[0]
sig = sigs[0]
means = {}
stds = {}
for col, sig in zip(cols, sigs):
    means[col.split('_')[0]], stds[col.split('_')[0]] = \
                                   df.loc[df[sig]==1, col].agg(['mean', 'std']).values
temp['sigVm_t_mean'] = pd.Series(means)
temp['sigVm_t_std'] = pd.Series(stds)







fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(16, 16))
axes = axes.flatten()

ax = axes[0]

ax.hist(latEngy50_v_df.cfisosect_time50.hist(bins=15))


mini, maxi = latEner50_v_df.cpisosect_time50.agg(['min', 'max'])
mini = 5 * floor(mini/5)
maxi = 5 * floor(maxi/5)
