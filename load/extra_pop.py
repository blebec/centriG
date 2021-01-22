#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 21 12:42:53 2021

@author: cdesbois
"""

import os
# from datetime import datetime
# from importlib import reload

# import matplotlib.gridspec as gridspec
# import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
# #from mpl_toolkits.axes_grid1.inset_locator import inset_axes
# from matplotlib import markers
# from matplotlib.patches import Rectangle
# from pandas.plotting import table

import config
# import fig_proposal as figp
# import general_functions as gfunc
# import load.load_data as ldat
# import load.load_traces as ltra
# import old.old_figs as ofig

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

#%%
def load_latences(sheet=0):
    """
    load the xcel file
    Parameters
    ----------
    sheet : the sheet number in the scel file, int, optional
        DESCRIPTION. The default is 0.

    Returns
    -------
    pandas dataframe
    """
    sheet = ['EXP 1', 'EXP 2'][sheet]
    paths['data'] = os.path.join(paths['owncFig'], 'infos_extra')
    file = 'Tableau_info_integrales_latences.xlsx'
    filename = os.path.join(paths['data'], file)
    df = pd.read_excel(filename, sheet_name=sheet, header=1)
    # adapt columns
    cols = [_.lower().strip() for _ in df.columns]
    cols = [_.replace(' ', '_') for _ in cols]
    cols = [_.replace('__', '_') for _ in cols]
    cols = [_.replace('topsynchro', 'top') for _ in cols]
    cols = [_.replace('latency', 'lat') for _ in cols]
    cols = [_.replace('half_height', 'hh') for _ in cols]
    cols = [_.replace('latence', 'lat') for _ in cols]
    cols = [_.replace('lat_onset', 'latOn') for _ in cols]
    cols = [_.replace('-_top', '-top') for _ in cols]
    cols = [_.replace('hh_lat', 'hhlat') for _ in cols]
    cols = [_.replace('lat_hh', 'hhlat') for _ in cols]
    
    
    
    
    cols[0] = 'channel'
    # clean row1 replace exp and pre
    # print message
    print('='*10)
    print( 'NB messages removed : {}'.format([_ for _ in df.loc[0].dropna()]))
    print('='*10)
    df.drop(df.index[0], inplace=True)
    # rename columns
    df.columns = cols
    # remove empty columns
    df = df.dropna(how='all', axis=1)
    # clean columns
    df[df.columns[0]] = df[df.columns[0]].apply(lambda x: x.split(' ')[1])
    df.layers = df.layers.apply(lambda x: x.split(' ')[1])
    return df

df = load_latences(1)

#%%
# pltconfig = config.rc_params()
# pltconfig['axes.titlesize'] = 'small'
plt.rcParams.update({'axes.titlesize': 'small'})

plt.close('all')

fig, axes = plt.subplots(nrows=4, ncols=6, figsize=(21, 16))
cols = df.mean().index.tolist()
cols = cols[1:]
axes = axes.flatten()
for i, col in enumerate(cols):
    ax = axes[i]
    df[col].hist( bins=20 ,ax=ax)
    ax.set_title(col)
    med = df[col].median()
    ax.axvline(med, color='tab:orange')
    ax.text(0.6, 0.6, f'{med:.1f}', ha='left', va='bottom', 
            transform=ax.transAxes, size='small', color='tab:orange',
            backgroundcolor='w')
    q0 = df[col].quantile(q=0.02)
    q1 = df[col].quantile(q=0.98)
    ax.set_xlim(q0, q1)
    
fig.tight_layout()
config.rc_params
config.rc_params()
