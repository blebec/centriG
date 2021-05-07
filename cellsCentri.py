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

def extract_cg_values():
    latEngy50_v_df = ldat.load_cell_contributions(rec='vm', amp='engy', age='new')
    latEngy50_s_df = ldat.load_cell_contributions(rec='spk', amp='engy', age='new')

    #build container
    cols = [_.split('_')[0] for _ in latEngy50_v_df.columns]
    cols = sorted(set(cols), key=cols.index)
    res_df = pd.DataFrame(index=cols)

    for i, kind in enumerate(['vm', 'spk']):
        for mes in ['time', 'engy']:
            df = [latEngy50_v_df ,latEngy50_s_df][i]
            # df = latEner50_v_df
            cols = [_ for _ in df.columns if mes in _]
            sigs = [_ for _ in cols if '_sig' in _]
            cols = [_ for _ in cols if '_sig' not in _]
            
            # all cells
            temp = df[cols].agg(['mean', 'std', 'min', 'max']).T
            temp = temp.reset_index()
            temp['index'].apply(lambda st: st.split('_')[0])
            temp.index = temp['index'].apply(lambda st: st.split('_')[0])
            del temp['index']
            txt = '{}_{}_'.format(kind, mes[0])
            temp.columns = [txt + st for st in temp.columns]
            res_df = res_df.join(temp)

            # sigs
            col = cols[0]
            sig = sigs[0]
            means = {}
            stds = {}
            minis = {}
            maxis = {}
            for col, sig in zip(cols, sigs):
                means[col.split('_')[0]], stds[col.split('_')[0]], \
                    minis[col.split('_')[0]], maxis[col.split('_')[0]] = \
                        df.loc[df[sig]==1, col].agg(['mean', 'std', 'min', 'max']).values
            txt = 'sig{}_{}_'.format(kind.title(), mes[0])
            res_df[txt + 'mean'] = pd.Series(means)
            res_df[txt + 'std'] = pd.Series(stds)
            res_df[txt + 'min'] = pd.Series(minis)
            res_df[txt + 'max'] = pd.Series(maxis)
    return res_df

res_df = extract_cg_values()

