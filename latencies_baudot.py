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
from scipy import stats


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
    """
    load the excel data and format it

    Returns
    -------
    df : pandas dataframe
    """
    filename = '/Users/cdesbois/ownCloud/cgFigures/data/baudot/Figure latence GABY + histo.xls'
    data_dict = pd.read_excel(filename, None)
    for k in data_dict:
        print(k)
    #
    df = data_dict['DITRIB latence min max calcul 1']
    # remove empty columns
    df = df.dropna(axis=1, how='all')
    df = df.drop(0)
    # remove empty rows
    df = df.dropna(axis=0, how='all')

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
    for col in ['moy_c-p', 'psth_seq-c']:
        df[col] = df[col].astype(float)
        return df


data_df = load_onsets()

#%%

def printLenOfRecording(df):
    names = [_ for _ in df.nom.unique() if _[:3].isdigit()]
    cells = {_[:5] for _ in names}

    print('='*15)
    print('nb of recordings for latencies = {}'.format(len(names)))
    print (names)
    print('='*15)
    print('nb of unique numId = {}'.format(len(cells)))
    print (cells)

printLenOfRecording(data_df)
#%%

def plot_onsetTransfertFunc(inputdf):
    """
    plot the vm -> time onset transfert function
    """
    datadf = inputdf.copy()
    cols = ['moy_c-p', 'psth_seq-c']
    stims = datadf.stim.unique()
    markers = {'cf' : 'o', 'cp' : 'v'}
    colors = ['tab:brown', std_colors['green'],
              std_colors['yellow'],std_colors['red']]
    #xscales
    xscales = [-30, 70]

    fig = plt.figure(figsize=(8, 6))
    fig.suptitle('spk Vm onset-time transfert function')
    ax = fig.add_subplot(111)

    for i, stim in enumerate(stims):
        df = datadf.loc[datadf.stim == stim, cols]
        # remove outliers
        df.loc[df[cols[0]] < xscales[0]] = np.nan
        df.loc[df[cols[0]] > xscales[1]] = np.nan
        df = df.dropna()
        # x = -1 * temp[values[0]].values
        x = df[cols[0]].values.astype(float)
        y = df[cols[1]].values.astype(float)
        # corr
        # r2 = stats.pearsonr(x.flatten(),y.flatten())[0]**2
        lregr = stats.linregress(x,y)
        r2 = lregr.rvalue ** 2
        print('{} \t r2= {:.3f} \t stdErrFit= {:.3f}'.format(stim, r2, lregr.stderr))
        # label = '{} {}  r2={:.3f}'.format(len(df), stim, r2)
        label = '{} cells, {}'.format(len(df), stim)
        ax.scatter(x, y, color=colors[i], marker=markers[stim.split('_')[0]],
                   s=100, alpha=0.8, label=label, edgecolor='w')
        # regress:
        x = x.reshape(len(x), 1)
        y = y.reshape(len(x), 1)
        regr = linear_model.LinearRegression()
        regr.fit(x,y)
        # if r2 > 0.001:
        #     ax.plot(x, regr.predict(x), color=colors[i], linestyle= ':',
        #             linewidth=3, alpha=0.7)
        
    mini = min(ax.get_xlim()[0], ax.get_ylim()[0])
    maxi = min(ax.get_xlim()[1], ax.get_ylim()[1])
    ax.plot([mini, maxi], [mini, maxi], '-', color='tab:grey', alpha=0.5)
    ax.legend()
    ax.axhline(0, color='tab:blue', linewidth=2, alpha=0.8)
    ax.axvline(0, color='tab:blue', linewidth=2, alpha=0.8)
    ax.set_ylabel('spikes onset relative latency (msec)')
    ax.set_xlabel('Vm onset relative latency (msec)')
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)

    ax.set_ylim(-30, 30)
    ax.set_xlim(xscales)

    if anot:
        date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        fig.text(0.99, 0.01, 'latencies_baudot.py:plot_onsetTransfertFunc',
                 ha='right', va='bottom', alpha=0.4)
        fig.text(0.01, 0.01, date, ha='left', va='bottom', alpha=0.4)
        txt = 'only {} range'.format(xscales)
        fig.text(0.5, 0.01, txt, ha='right', va='bottom', alpha=0.4)
    fig.tight_layout()
    return fig


plt.close('all')
figure = plot_onsetTransfertFunc(data_df)

save = False
if save:
    file = 'onsetTransfertFunc.png'
    dirname = os.path.join(paths['owncFig'], 'pythonPreview', 'baudot')
    file_name = os.path.join(dirname, file)
    figure.savefig(file_name)

#%% histogrammes

plt.close('all')

def histo_inAndOut(inputdf, removeOutliers=True):

    datadf = inputdf.copy()

    fig, axes = plt.subplots(nrows=4, ncols=2, figsize=(8,6),
                          sharex=True, sharey=True)
    axes = axes.flatten(order='F')

    cols = ['moy_c-p', 'psth_seq-c']
    stims = datadf.stim.unique()
    colors = ['tab:brown', std_colors['green'],
              std_colors['yellow'],std_colors['red']]
    if removeOutliers:
        # xscales
        xscales = [-30, 70]
        datadf.loc[datadf[cols[0]] < xscales[0]] = np.nan
        datadf.loc[datadf[cols[0]] > xscales[1]] = np.nan

    # define bins
    maxi = max(datadf[cols[0]].quantile(q=.95),
               datadf[cols[1]].quantile(q=.95))
    mini = min(datadf[cols[0]].quantile(q=.05),
               datadf[cols[1]].quantile(q=.05))
    maxi = ceil(maxi/5)*5
    mini = floor(mini/5)*5
    # plot
    for k in range(2):      # [vm, spk]
        for i, stim in enumerate(stims):
            ax = axes[i + 4*k]
            df = datadf.loc[datadf.stim == stim, cols[k]].dropna()
            a, b, c = df.quantile([.25, .5, .75])
            ax.axvline(b, color=colors[i], alpha=0.6, linewidth=2)
            ax.axvspan(a, c, color=colors[i], alpha=0.3)
            txt = 'med= {:.0f}'.format(b)
            ax.text(x=1, y=0.8, s=txt, color='tab:grey', fontsize='small',
                    va='bottom', ha='right', transform=ax.transAxes)
            txt = '{} cells'.format(len(df))
            ax.text(x=0, y=0.8, s=txt, color='tab:grey', fontsize='small',
                    va='bottom', ha='left', transform=ax.transAxes)
            # histo
            height, x = np.histogram(df.values, bins=range(mini, maxi, 5),
                                     density=True)
            x = x[:-1]
            align='edge' # ie right edge
            # NB ax.bar, x value = lower
            ax.bar(x, height=height, width=5, align=align,
                   color=colors[i], edgecolor='k', alpha=0.6, label='stim')
            ax.axvline(0, color='tab:blue', alpha=0.7, linewidth=2)

    for ax in axes:
        ax.set_yticks([])
        for spine in ['left', 'top', 'right']:
            ax.spines[spine].set_visible(False)
    axes[0].set_title('Input (Vm)')
    axes[4].set_title('Output (Spk)')
    axes[3].set_xlabel('Onset Relative Latency (msec)')
    axes[7].set_xlabel('Onset Relative Latency (msec)')

    if anot:
        date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        fig.text(0.99, 0.01, 'latencies_baudot.py:histo_inOut',
                 ha='right', va='bottom', alpha=0.4)
        fig.text(0.01, 0.01, date, ha='left', va='bottom', alpha=0.4)
        if removeOutliers:
            txt = 'only {} range '.format(xscales)
            fig.text(0.5, 0.01, txt, ha='center', va='bottom', alpha=0.4)
    fig.tight_layout()
    return fig

plt.close('all')
figure = histo_inAndOut(data_df, removeOutliers=True)

save = False
if save:
    file = 'histo_inAndOut.png'
    dirname = os.path.join(paths['owncFig'], 'pythonPreview', 'baudot')
    file_name = os.path.join(dirname, file)
    figure.savefig(file_name)

#%% diff /ref
plt.close('all')


def plot_diffMean(inputdf, removeOutliers=True, refMean=True):
# datadf = data_df.copy()
    datadf = inputdf.copy()
    cols = ['moy_c-p', 'psth_seq-c']
    stims = datadf.stim.unique()
    markers = {'cf' : 'o', 'cp' : 'v'}
    colors = ['tab:brown', std_colors['green'],
              std_colors['yellow'],std_colors['red']]
    if removeOutliers:
        #xscales
        xscales = [-30, 70]
        datadf.loc[datadf[cols[0]] < xscales[0]] = np.nan
        datadf.loc[datadf[cols[0]] > xscales[1]] = np.nan

    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 12),
                             sharex=True, sharey=True)
    axes = axes.flatten()

    for i, stim in enumerate(stims):
        ax = axes[i]
        df = datadf.loc[datadf.stim == stim, cols].dropna().copy()
        if refMean:
            x = df.mean(axis=1)
        else:
            x = df[cols[0]]
        y = df.diff(axis=1)[cols[1]]
        # y = (df[cols[1]] - df[cols[0]])
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
        txt = '{} cells'.format(len(df))
        ax.text(x=0.05, y=0.7, s=txt, color='tab:grey', fontsize='small',
                va='bottom', ha='left', transform=ax.transAxes)
        ax.axhline(0, color='tab:blue', linewidth=2, alpha=0.8)
        ax.axvline(0, color='tab:blue', linewidth=2, alpha=0.8)
        ax.set_ylabel('spikes - vm (msec)')
        if refMean:
            ax.set_xlabel('mean(Vm, Spk) onset relative latency (msec)')
        else:
            ax.set_xlabel('Vm onset relative latency (msec)')
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
ref_mean = True
figure = plot_diffMean(data_df, refMean=ref_mean)
save = False
if save:
    file = 'diffMean.png'
    dirname = os.path.join(paths['owncFig'], 'pythonPreview', 'baudot')
    file_name = os.path.join(dirname, file)
    figure.savefig(file_name)
