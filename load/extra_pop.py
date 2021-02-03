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
import statsmodels.api as sm
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

def plot_all_histo(df):
    
    fig, axes = plt.subplots(nrows=4, ncols=6, figsize=(21, 16))
    cols = df.mean().index.tolist()
    cols = cols[1:]
    axes = axes.flatten()
    for i, col in enumerate(cols):
        ax = axes[i]
        df[col].hist(bins=20, ax=ax, density=True)
        ax.set_title(col)
        med = df[col].median()
        ax.axvline(med, color='tab:orange')
        ax.text(0.6, 0.6, f'{med:.1f}', ha='left', va='bottom', 
                transform=ax.transAxes, size='small', color='tab:orange',
                backgroundcolor='w')
        for spine in ['left', 'top', 'right']:
            ax.spines[spine].set_visible(False)
            ax.set_yticks([])
            # q0 = df[col].quantile(q=0.02)
            # q1 = df[col].quantile(q=0.98)
            # ax.set_xlim(q0, q1)
    
    fig.tight_layout()
    return fig

fig = plot_all_histo(df)

#%% test dotplot  

#TODO use floats not integers

plt.close('all')
fig = plt.figure()
ax = fig.add_subplot(211)

colors = ['tab:blue', 'tab:red', 'tab:orange']
for i in range(3):
    col = df.columns[i+3]

    x = df[col].values.tolist()
    y = df.index.tolist()

    ax.plot(x, y, '.', color=colors[i], markersize=10, alpha=0.5, label=col)
    ax.axvline(df[col].median(), color=colors[i], alpha=0.5)
    ax.legend()
ax.set_ylim(ax.get_ylim()[::-1])

ax = fig.add_subplot(212)
colors = ['tab:blue', 'tab:red', 'tab:orange']
cols = df.columns[[8,6,7]]
for i, col in enumerate(cols):

    x = df[col].values.tolist()
    y = df.index.tolist()

    ax.plot(x, y, '.', color=colors[i], markersize=10, alpha=0.5, label=col)
    ax.axvline(df[col].median(), color=colors[i], alpha=0.5)
    ax.legend()
    
ax.set_ylim(ax.get_ylim()[::-1])

#%%
df.loc[57, [df.columns[3]]] = np.nan
df.loc[57, df.columns[4]] = np.nan

df.loc[25:26, df.columns[4]] = np.nan

df.loc[17, df.columns[5]] = np.nan
df.loc[19, df.columns[5]] = np.nan

df.loc[60, df.columns[6]] = np.nan
#pb latence excessive = 100 ? à supprimer
df.loc[[3, 5, 6, 7, 8,  9, 10,11, 35, 36, 37, 40,  41, 42, 43, 44, 45, 47, 51,
 53, 54, 57, 59, 62, 64], df.columns[6]] = np.nan

#df.loc[57] = np.nan

df.loc[9, df.columns[25]] = np.nan
df.loc[46, df.columns[25]] = np.nan
df.loc[40, df.columns[25]] = np.nan



#%% filter
plt.close(fig)

i = 3
i = 4

pyperclip.copy(i)
col = df.columns[i]
print(col)
print(df[col].value_counts().sort_index())

fig = plt.figure()
fig.suptitle(col)
ax = fig.add_subplot(111)
ax.hist(df[col], bins=30)

#%%
df.loc[df[col] <20, [col]]

df.loc[df[col] >30, [col]].index.tolist()


#%%
print('col 2 col δori_pref-ori_gabor : 2 values -20 and 0')
print('electrode 57 seems bad')
#%%  to be adpated

def statsmodel_diff_mean(df, param=params):
    df = df.dropna()
    # extract correlation
    y = df.diffe
    x = df.moy
    # build the model & apply the fit
    x = sm.add_constant(x) # constant intercept term
    model = sm.OLS(y, x)
    fitted = model.fit()
    print(fitted.summary())

    #make prediction
    x_pred = np.linspace(x.min()[1], x.max()[1], 50)
    x_pred2 = sm.add_constant(x_pred) # constant intercept term
    y_pred = fitted.predict(x_pred2)
    print(y_pred)

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111)
    # x = 'ip1m'  # PVC
    # y = 'ip2m'  # jug
    sm.graphics.mean_diff_plot(m1=df.jug, m2=df.cvp, ax=ax)
    ax.plot(x_pred, y_pred, color='tab:red', linewidth=2, alpha=0.8)
    txt = 'difference = {:.2f} + {:.2f} mean'.format(
        fitted.params['const'], fitted.params['moy'])
    ax.text(13.5, -1, txt, va='bottom', ha='right', color='tab:red')

    ax.axvline(df.moy.mean(), color='tab:orange', linewidth=2, alpha=0.6)
    txt = 'measures = \n {:.2f} ± {:.2f}'.format(
        df.moy.mean(), df.moy.std())
    ax.text(8.7, -2.7, txt, color='tab:orange', va='center', ha='right')

    ax.axhline(df.diffe.mean(), color='tab:orange', linewidth=2, alpha=0.6)
    txt = 'differences = \n {:.2f} ± {:.2f}'.format(
        df.diffe.mean(), df.diffe.std())
    ax.text(13.5, 0.6, txt, color='tab:orange', va='center', ha='right')

    ax.set_ylabel('jug - cvp    (mmHg)')  # ip2m - ip1m
    ax.set_xlabel('mean jug|cvp    (mmHg)')
    ax.axhline(0, color='tab:blue', alpha=0.6)
    for spine in ['left', 'top', 'right', 'bottom']:
        ax.spines[spine].set_visible(False)
    fig.text(0.99, 0.01, 'cDesbois', ha='right', va='bottom', alpha=0.4, size=12)
    fig.text(0.01, 0.01, param['file'], ha='left', va='bottom', alpha=0.4)
    return fig