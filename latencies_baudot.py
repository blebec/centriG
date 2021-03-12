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

file_name = '/Users/cdesbois/ownCloud/cgFigures/data/baudot/scatterData/scatLat.xlsx'

# filename = '/Users/cdesbois/ownCloud/cgFigures/data/baudot/Figure latence GABY + histo.xls'

# #%% all   ... and forget
# filename = '/Users/cdesbois/ownCloud/cgFigures/data/baudot/traitement_latencepiero_S-C_final_papier_2.xls'
# data_dict = pd.read_excel(filename, None)
# for key in data_dict:
#     print(key)

# df = data_dict['DITRIB latence min max calcul 1']
# df = data_dict['LATENCE PIERO S-C']


def load_onsets():
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
    df[df.columns[0]] = df[df.columns[0]].apply(lambda st: '_'.join(st.lower().split(' ')[::-1]))
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
    df['moy_c-p'] *= (-1) # correction to obtain relative latency
    return df


data_df = load_onsets()
#%%

def plot_onsetTransfertFunc(df):
    """
    plot the vm -> time onset transfert function
    """
    values = ['moy_c-p', 'psth_seq-c']
    stims = df.stim.unique()
    markers = {'cf' : '^', 'cp' : 'v'}
    colors = ['tab:brown', std_colors['green'],
              std_colors['yellow'],std_colors['red']]

    fig = plt.figure(figsize=(8, 6))
    fig.suptitle('spk Vm onset-time transfert function')
    ax = fig.add_subplot(111)

    for i, stim in enumerate(stims):
        temp = df.loc[df.stim == stim, values].dropna()
        # x = -1 * temp[values[0]].values
        x = temp[values[0]].values
        y = temp[values[1]].values
        ax.scatter(x, y, color=colors[i], marker=markers[stim.split('_')[0]],
                   s=100, alpha=0.8, label=stim, edgecolor='w')
        # regress:
        x = x.reshape(len(x), 1)
        y = y.reshape(len(x), 1)
        regr = linear_model.LinearRegression()
        regr.fit(x,y)
        ax.plot(x, regr.predict(x), color=colors[i], linestyle= ':',
                linewidth=3, alpha=0.7)
    ax.legend()
    ax.axhline(0, color='tab:blue', linewidth=2, alpha=0.8)
    ax.axvline(0, color='tab:blue', linewidth=2, alpha=0.8)
    ax.set_ylabel('spikes onset relative latency (msec)')
    ax.set_xlabel('Vm onset relative latency (msec)')
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)

    ax.set_ylim(-30, 30)
    ax.set_xlim(-30, 70)

    if anot:
        date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        fig.text(0.99, 0.01, 'latencies_baudot.py:plot_onsetTransfertFunc',
                 ha='right', va='bottom', alpha=0.4)
        fig.text(0.01, 0.01, date, ha='left', va='bottom', alpha=0.4)
    fig.tight_layout()
    return fig


plt.close('all')
fig = plot_onsetTransfertFunc(data_df)

save = False
if save:
    file = 'onsetTransfertFunc.png'
    dirname = os.path.join(paths['owncFig'], 'pythonPreview', 'baudot')
    filename = os.path.join(dirname, file)
    fig.savefig(filename)

#%% histogrammes

plt.close('all')

def histo_inAndOut(df):

    # df = data_df.copy()

    fig, axes = plt.subplots(nrows=4, ncols=2, figsize=(8,6),
                          sharex=True, sharey=True)
    axes = axes.flatten(order='F')

    values = ['moy_c-p', 'psth_seq-c']
    stims = df.stim.unique()
    colors = ['tab:brown', std_colors['green'],
              std_colors['yellow'],std_colors['red']]

    # define bins
    maxi = max(data_df[values[0]].quantile(q=.95),
               data_df[values[1]].quantile(q=.95))
    mini = min(data_df[values[0]].quantile(q=.05),
               data_df[values[1]].quantile(q=.05))
    maxi = ceil(maxi/5)*5
    mini = floor(mini/5)*5
    # in = VM
    for i, stim in enumerate(stims):
        ax = axes[i]
        temp = df.loc[df.stim == stim, values[0]].dropna()
        a, b, c = temp.quantile([.25, .5, .75])
        ax.axvline(b, color=colors[i], alpha=0.6, linewidth=2)
        ax.axvspan(a, c, color=colors[i], alpha=0.3)
        height, x = np.histogram(temp.values, bins=range(mini, maxi, 5),
                                 density=True)
        txt = '{:.1f} ({:.0f} {:.0f})'.format(a, b, c)
        ax.text(x=1, y=0.5, s=txt, color='tab:grey', fontsize='small',
                va='bottom', ha='right', transform=ax.transAxes)
        x = x[:-1]
        align='edge' # ie right edge
        # NB ax.bar, x value = lower
        ax.bar(x, height=height, width=5, align=align,
               color=colors[i], edgecolor='k', alpha=0.6, label='stim')
        ax.axvline(0, color='tab:blue', alpha=0.7, linewidth=2)
    # out = Spk
    for i, stim in enumerate(stims):
        ax = axes[i+4]
        temp = df.loc[df.stim == stim, values[1]].dropna()
        a, b, c = temp.quantile([.25, .5, .75])
        ax.axvline(b, color=colors[i], alpha=0.6, linewidth=2)
        ax.axvspan(a, c, color=colors[i], alpha=0.3)
        txt = '{:.1f} ({:.0f} {:.0f})'.format(a, b, c)
        ax.text(x=1, y=0.5, s=txt, color='tab:grey', fontsize='small',
                va='bottom', ha='right', transform=ax.transAxes)
        height, x = np.histogram(temp.values, bins=range(mini, maxi, 5),
                                 density=True)
        x = x[:-1]
        align='edge' # ie right edge
        # NB ax.bar, x value = lower
        ax.bar(x, height=height, width=5, align=align,
               color=colors[i], edgecolor='k', alpha=0.6, label='stim')
        ax.axvline(0, color='tab:blue', alpha=0.7, linewidth=2)

    for ax in axes:
        for spine in ['left', 'top', 'right']:
            ax.spines[spine].set_visible(False)
        ax.set_yticks([])

    axes[0].set_title('Input (Vm))')
    axes[4].set_title('Output (Spk)')
    axes[3].set_xlabel('Onset Relative Latency (msec)')
    axes[7].set_xlabel('Onset Relative Latency (msec)')


    if anot:
        date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        fig.text(0.99, 0.01, 'latencies_baudot.py:plot_onsetTransfertFunc',
                 ha='right', va='bottom', alpha=0.4)
        fig.text(0.01, 0.01, date, ha='left', va='bottom', alpha=0.4)
    fig.tight_layout()
    return fig

plt.close('all')
fig = histo_inAndOut(data_df)

save = False
if save:
    file = 'histo_inAndOut.png'
    dirname = os.path.join(paths['owncFig'], 'pythonPreview', 'baudot')
    filename = os.path.join(dirname, file)
    fig.savefig(filename)

#%% diff /ref
plt.close('all')


def plot_diffMean(df):
# df = data_df.copy()

    values = ['moy_c-p', 'psth_seq-c']
    stims = df.stim.unique()
    markers = {'cf' : '^', 'cp' : 'v'}
    colors = ['tab:brown', std_colors['green'],
              std_colors['yellow'],std_colors['red']]

    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 12), 
                             sharex=True, sharey=True)
    axes = axes.flatten()

    for i, stim in enumerate(stims):
        ax = axes[i]
        temp = df.loc[df.stim == stim, values].dropna().copy()
        x = temp.mean(axis=1)
        y = temp.diff(axis=1)[values[1]]
        # y = (temp[values[1]] - temp[values[0]])
        # quantiles        
        a, b, c = x.quantile([.25, .5, .75])
        ax.axvline(b, color=colors[i], alpha=0.6, linewidth=2)
        ax.axvspan(a, c, color=colors[i], alpha=0.3)
        a, b, c = y.quantile([.25, .5, .75])
        ax.axhline(b, color=colors[i], alpha=0.6, linewidth=2)
        ax.axhspan(a, c, color=colors[i], alpha=0.3)
        # plot
        x = x.values
        y = y.values
        ax.scatter(x, y, color=colors[i], marker=markers[stim.split('_')[0]],
                   s=100, alpha=0.8, label=stim, edgecolor='w')
        # regress:
        x = x.reshape(len(x), 1)
        y = y.reshape(len(x), 1)
        regr = linear_model.LinearRegression()
        regr.fit(x,y)
        ax.plot(x, regr.predict(x), color=colors[i], linestyle= ':',
                linewidth=3, alpha=0.7)
        ax.legend()
        txt = '{} cells'.format(len(temp))
        ax.text(x=0.05, y=0.7, s=txt, color='tab:grey', fontsize='small',
                va='bottom', ha='left', transform=ax.transAxes)
        ax.axhline(0, color='tab:blue', linewidth=2, alpha=0.8)
        ax.axvline(0, color='tab:blue', linewidth=2, alpha=0.8)
        ax.set_ylabel('spikes - vm (msec)')
        ax.set_xlabel('Vm Spk mean onset relative latency (msec)')
        for spine in ['top', 'right']:
            ax.spines[spine].set_visible(False)

    ax.set_ylim(-60, 60)
    # ax.set_xlim(-25, 60)

    if anot:
        date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        fig.text(0.99, 0.01, 'latencies_baudot.py:plot_diffMean',
                 ha='right', va='bottom', alpha=0.4)
        fig.text(0.01, 0.01, date, ha='left', va='bottom', alpha=0.4)
    fig.tight_layout()
    return fig

plt.close('all')
fig = plot_diffMean(data_df)
save = False
if save:
    file = 'diffMean.png'
    dirname = os.path.join(paths['owncFig'], 'pythonPreview', 'baudot')
    filename = os.path.join(dirname, file)
    fig.savefig(filename)
