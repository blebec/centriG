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
sheet = ['EXP 1', 'EXP 2'][0]
paths['data'] = os.path.join(paths['owncFig'], 'infos_extra')
file = 'Tableau_info_integrales_latences.xlsx'
filename = os.path.join(paths['data'], file)
df = pd.read_excel(filename, sheet_name=sheet, header=1)
# adapt columns
cols = [_.lower().strip() for _ in df.columns]
datacols = [_.replace(' ', '_') for _ in cols]
cols = [_.replace('__', '_') for _ in cols]
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
