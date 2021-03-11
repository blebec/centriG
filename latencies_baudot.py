#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 11 09:35:00 2021

@author: cdesbois
"""

import os
from datetime import datetime
from importlib import reload
from math import ceil, floor

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import linear_model


import config


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

paths['data'] = os.path.join(paths['owncFig'], 'data')


plt.rcParams.update(
    {'font.sans-serif': ['Arial'],
     'font.size': 14,
     'legend.fontsize': 'medium',
     'figure.figsize': (11.6, 5),
     'figure.dpi': 100,
     'axes.labelsize': 'medium',
     'axes.titlesize': 'medium',
     'xtick.labelsize': 'medium',
     'ytick.labelsize': 'medium',
     'axes.xmargin': 0}
)

filename = '/Users/cdesbois/ownCloud/cgFigures/data/baudot/scatterData/scatLat.xlsx'
#%%

# filename = '/Users/cdesbois/ownCloud/cgFigures/data/baudot/Figure latence GABY + histo.xls'

# #%% all   ... and forget
# filename = '/Users/cdesbois/ownCloud/cgFigures/data/baudot/traitement_latencepiero_S-C_final_papier_2.xls'
# data_dict = pd.read_excel(filename, None)
# for key in data_dict:
#     print(key)

# df = data_dict['DITRIB latence min max calcul 1']
# df = data_dict['LATENCE PIERO S-C']
#%%
filename = '/Users/cdesbois/ownCloud/cgFigures/data/baudot/Figure latence GABY + histo.xls'
data_dict = pd.read_excel(filename, None)
for k in data_dict:
    print(k)
#
df = data_dict['DITRIB latence min max calcul 1']
# remove empty columns
df = df.dropna(axis=1, how='all')
df = df.drop(0)

# kinds of stimulation formating
df[df.columns[0]] = df[df.columns[0]].fillna(method='ffill')
df[df.columns[0]] = df[df.columns[0]].apply(lambda st: '_'.join(st.lower().split(' ')))
df[df.columns[-1]] = df[df.columns[-1]].fillna(method='ffill')
df[df.columns[-1]] = df[df.columns[-1]].apply(lambda st: st.lower())
df = df.rename(columns = {df.columns[0]: 'stim', df.columns[-1]:'repr'})

# manage columns names
cols = [st.lower() for st in df.columns]
cols = [_.replace('correlation', 'corr') for _ in cols]
cols = [_.replace('airsign', 'sig') for _ in cols]
cols = ['_'.join(_.split(' ')) for _ in cols]
cols[-4] = cols[-4].replace('lat_sig_', '').split('.')[0]
cols[-4] = cols[-4].replace('_s', '_seq').split('.')[0]
cols[-3] = cols[-3].replace('lat_sig_', '').split('.')[0]
cols[-3] = cols[-3].replace('_s', '_seq').split('.')[0]
df.columns = cols
