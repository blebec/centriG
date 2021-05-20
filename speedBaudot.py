#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Thu Jan 28 16:55:32 2021

@author: cdesbois
"""

import os
from datetime import datetime
from importlib import reload
from bisect import bisect
# from itertools import zip_longest

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats

import config

std_colors = config.std_colors()
speed_colors = config.speed_colors()

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

# NB all limits are lower ones!

def load_speed_data():
    file = 'baudot.csv'
    filename = os.path.join(paths['data'], file)
    bddf = pd.read_csv(filename)
    bddf.columns = [_.strip() for _ in bddf.columns]
    # to have the lower interval limit
    bddf.optiMax = bddf.optiMax - 25

    file = 'neuron_props_speed.xlsx'
    filename = os.path.join(paths['data'], 'averageTraces', file)
    cgdf = pd.read_excel(filename)
    cols = [st.strip() for st in cgdf.columns]
    cols = [st.lower() for st in cols]
    cols = [st.replace('(', '_') for st in cols]
    cols = [st.replace(')', '') for st in cols]
    cgdf.columns = cols
    return bddf, cgdf

def build_summary(bddf, cgdf):
    """
    class the number of cells
    input = baudot dataframe & centrigabor dataframe
    output = summary dataframe (nb of cells, upper interval spped limit)
    """
    # count number of cells
    def class_speed(s):
        "return lower limit in class"
        lims = range(100, 500, 25)
        i = bisect(lims, s)
        return lims[i-1]
    cgdf['optiMax'] = cgdf.speed_isi0.apply(lambda x: class_speed(x))
    cgNumb = dict(cgdf.optiMax.value_counts())
    bdNumb = dict(bddf.set_index('optiMax'))

    summary_df = pd.DataFrame(index=range(50, 525, 25))
    summary_df = summary_df.join(pd.DataFrame(bdNumb))
    summary_df.columns = ['bd_cells']
    summary_df['cg_cells'] = pd.Series(cgNumb)
    summary_df = summary_df.fillna(0)
    summary_df = summary_df.astype(int)
    return summary_df


def load_cgpopdf():
    """
    load the whole centrigabor population  (n=37)

    Returns
    -------
    pandas dataframe
    """
    file = 'centrigabor_pop_db.xlsx'

    filename = os.path.join(paths['data'], 'averageTraces', file)
    df = pd.read_excel(filename)
    df.columns = [col.lower().strip() for col in df.columns]
    return df

def load_bringuier():
    filename = os.path.join(paths['data'], 'bringuier.csv')
    brdf = pd.read_csv(filename, sep='\t', decimal=',')
    brdf.speed_upper = brdf.speed_upper ## * 1000  # 1mm / visual°
    # unstack
    temp = brdf.impulse.fillna(0) - brdf.long_bar.fillna(0)
    temp = temp.apply(lambda x: x if x>=0 else 0)
    brdf.impulse = temp
    brdf = brdf.fillna(0)
    brdf = brdf.rename(columns={'speed_upper': 'speed_lower'})
    return brdf

def load_gmercier():
    file = 'gmercier.csv'
    filename = os.path.join(paths['data'], file)
    df = pd.read_csv(filename, sep='\t')
    del df['speed_high']
    return df

def load_gmercier2():
    file = 'gmercier2.csv'
    filename = os.path.join(paths['data'], file)
    df = pd.read_csv(filename, sep='\t', decimal=',')
    df = df.set_index('cell')
    # remove empty lines
    df = df.dropna(how='all')
    return df


# speed and baudot
bddf, spdf = load_speed_data()
summary_df = build_summary(bddf, spdf)
# replaced cgdf by spdf (speeddf)

# gmercier
gmdf = load_gmercier()
gmdf2 = load_gmercier2()

# binguier
brdf = load_bringuier()

# cg population
popdf = load_cgpopdf()

#%
def samebin(popdf=popdf, spdf=spdf, bddf=bddf, gmdf=gmdf, brdf=brdf):
    def compute_speed_histo(speeds, bins=40):
        #speeds = df.speed/1000
        height, x = np.histogram(speeds, bins=bins, range=(0,1))
        # normalize
        # height = height/np.sum(height)
        width = (max(x) - min(x))/(len(x) -1)
        return x[:-1], height, width

    # res dataframe 20 bins 0 to 1 range
    df = pd.DataFrame(index = [_/20 for _ in range(0, 20, 1)])
    # centrigabor population
    x, height_cgpop, width = compute_speed_histo(popdf.speed/1000, bins=20)
    df['cgpop'] = height_cgpop
    # speed centrigabor pop
    _, height_cgspeed, _ = compute_speed_histo(spdf.speed_isi0/1000, bins=20)
    df['cgspeed'] = height_cgspeed
    # baudot -> resample to double bin width
    temp = bddf.copy()
    temp.loc[-1] = [150, 0] # append row to match the future scale
    temp.index = temp.index + 1
    temp = temp.sort_index()
    temp = temp.set_index('optiMax')
    temp.index = temp.index/1000
    temp['cells'] = temp.nbCell.shift(-1).fillna(0)
    temp.cells += temp.nbCell
    temp = temp.drop(temp.index[1::2])
    temp = temp.drop(columns=['nbCell'])
    df['bd'] = temp
    # gmercier
    temp = gmdf.set_index('speed_low')
    temp.index = temp.index/1000
    df['gm'] = temp
    # bringuier
    temp = brdf.set_index('speed_lower')
    # remove upper limit
    temp = temp.drop(index=[100])
    #rename columns
    cols = ['br_' + st for st in temp.columns]
    df[cols[0]] = temp[temp.columns[0]]
    df[cols[1]] = temp[temp.columns[1]]
    # fill and convert to integer
    df = df.fillna(0).astype('int')
    return df

bined_df = samebin()


#%%

def plot_optimal_speed(df):

    # prepare data
    height_cg, x = np.histogram(popdf.speed, bins=18, range=(50, 500))
    x = x[:-1]
    df = pd.DataFrame(index=range(50, 525, 25))
    df['popcg'] = pd.Series(data=height_cg, index=x)
    df['bd'] = bddf.set_index('optiMax')
    df = df.fillna(0)
    align = 'edge' # ie right edge
    width = (df.index.max() - df.index.min())/(len(df) - 1)

    # plot
    fig = plt.figure(figsize=(11.6, 5))
    ax = fig.add_subplot(111)
    # NB ax.bar, x value = lower
    ax.bar(df.index, height=df.popcg, width=width, align=align,
           color='w', edgecolor='k', alpha=0.6, label='cengrigabor')
    ax.bar(df.index, height=df.bd, bottom=df.popcg, width=width, align=align,
           color='tab:gray', edgecolor='k', alpha=0.6, label='baudot')
    txt = 'n = {:.0f} cells'.format(df.popcg.sum())
    ax.text(x=0.8, y= 0.6, s=txt, color='k',
            va='bottom', ha='left', transform=ax.transAxes)
    txt = 'n = {:.0f} cells'.format(df.bd.sum())
    ax.text(x=0.8, y= 0.5, s=txt, color='tab:gray',
            va='bottom', ha='left', transform=ax.transAxes)
    ax.set_xlabel('{} (°/sec)'.format('optimal apparent speed'.title()))
    ax.set_ylabel('nb of Cells')
    ax.legend()

    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)

    fig.tight_layout()
    if anot:
        date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        fig.text(0.99, 0.01, 'speedBaudot.py:plot_optimal_speed',
                 ha='right', va='bottom', alpha=0.4)
        fig.text(0.01, 0.01, date, ha='left', va='bottom', alpha=0.4)
    return fig


plt.close('all')
anot = True

fig = plot_optimal_speed(summary_df)
save = False
if save:
    file = 'optSpeed.pdf'
    dirname = os.path.join(paths['owncFig'], 'pythonPreview', 'baudot')
    filename = os.path.join(dirname, file)
    fig.savefig(filename)

#%%
plt.close('all')

def plot_optimal_bringuier(gdf=bined_df):
    """
    plot an histogram of the optimal cortical horizontal speed
    Parameters
    ----------
    df = pandas dataFrame
    Returns
    -------
    fig = matplotlib.pyplot.figure

    """
    fig, ax = plt.subplots(figsize=(4.3, 4), nrows=1, ncols=1)
    # gerenal features
    x = gdf.index
    width = (max(x) - min(x))/(len(x) - 1)*.98
    align = 'edge'

    txt = 'Bar n={:.0f}'.format(gdf.br_long_bar.sum())
    ax.bar(x, height=gdf.br_long_bar, width=width, align=align, alpha=0.8,
           color=std_colors['blue'], edgecolor='k',
           label=txt)
    txt = 'SN n={:.0f}'.format(gdf.br_impulse.sum())
    ax.bar(x, height=gdf.br_impulse, bottom=gdf.br_long_bar, width=width,
           align=align, color=std_colors['green'],
           edgecolor='k', alpha=0.8, label=txt)
    # txt = 'Apparent Speed of Horizontal Propagation (ASHP) m/s'
    txt = 'Propagation Speed (mm/ms)'
    ax.set_xlabel(txt)
    ax.set_ylabel('Nb of measures')
    ax.legend()

    for ax in fig.get_axes():
        for spine in ['top', 'right']:
            ax.spines[spine].set_visible(False)

    fig.tight_layout()
    lims = ax.get_xlim()
    ax.set_xlim(0, lims[1])   
    if anot:
        date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        fig.text(0.99, 0.01, 'speedBaudot.py:plot_optimal_bringuier',
                 ha='right', va='bottom', alpha=0.4)
        fig.text(0.01, 0.01, date, ha='left', va='bottom', alpha=0.4)
    return fig


plt.close('all')
fig = plot_optimal_bringuier(bined_df)

save = False
if save:
    file = 'optSpreedBringuier.pdf'
    dirname = os.path.join(paths['owncFig'], 'pythonPreview', 'baudot')
    filename = os.path.join(dirname, file)
    fig.savefig(filename)
    #update current
    dirname = os.path.join(paths['owncFig'], 'pythonPreview', 'current', 'fig')
    file = 'o1_optSpeedBringuier'
    for ext in ['.png', '.pdf', '.svg']:
        filename = os.path.join(dirname, (file + ext))
        fig.savefig(filename)

#%%
# def plot_both(df0, df1, df2, df3):
def plot_both(gdf=bined_df):

    # fig = plt.figure(figsize=(4.3, 8))
    # axes = []
    # ax = fig.add_subplot(211)
    # axes.append(ax)
    # ax = fig.add_subplot(212, sharex=ax, sharey=ax)
    # axes.append(ax)
    
    fig, ax = plt.subplots(figsize=(4.3, 8))
    # gerenal features
    x = gdf.index
    width = (max(x) - min(x))/(len(x) - 1)*.98
    align = 'edge'

    # top
    # ax = axes[0]

    ax.bar(x, height=gdf.br_impulse + gdf.br_long_bar, width=width, 
            align=align,  color='w', alpha=0.7, 
            edgecolor='tab:grey')
    # ax.bar(x, height=gdf.br_long_bar, width=width, 
    #        align=align, alpha=0.5,
    #        color='w', edgecolor='tab:grey')
    # ax.bar(x, height=gdf.br_impulse, bottom=gdf.br_long_bar, width=width,
    #        align=align, alpha=0.5, 
    #        color='w', edgecolor='tab:grey')

    gdf['pool'] = 0
    txt = 'Radial n={:.0f}'.format(gdf.cgpop.sum())
    ax.bar(x, gdf.cgpop, bottom=gdf.pool, width=width, align=align,
           color=speed_colors['red'], alpha=0.6, edgecolor='k', label=txt)
    gdf.pool += gdf.cgpop
    txt = 'Cardinal n={:.0f}'.format(gdf.bd.sum())
    ax.bar(x, gdf.bd, bottom=gdf.pool, width=width, align=align,
           color=speed_colors['yellow'], alpha=0.6, edgecolor='k', label=txt)
    gdf.pool += gdf.bd
    txt = '2-stroke n={:.0f}'.format(gdf.gm.sum())
    ax.bar(x, gdf.gm, bottom=gdf.pool, width=width, align=align,
           color=speed_colors['orange'], alpha=0.6, edgecolor='k', label=txt)
    gdf.pool += gdf.gm

    txt = 'Inferred Cortical Speed (mm/ms)'
    ax.set_xlabel(txt)
    ax.set_ylabel('Nb of cells')
    ax.legend()
    # # bottom
    # ax = axes[1]
    # txt = 'Bar n={:.0f}'.format(gdf.br_long_bar.sum())
    # ax.bar(x, height=gdf.br_long_bar, width=width, align=align, alpha=0.8,
    #        color=std_colors['blue'], edgecolor='k',
    #        label=txt)
    # txt = 'SN n={:.0f}'.format(gdf.br_impulse.sum())
    # ax.bar(x, height=gdf.br_impulse, bottom=gdf.br_long_bar, width=width,
    #        align=align, color=std_colors['green'],
    #        edgecolor='k', alpha=0.8, label=txt)
    # # txt = 'Apparent Speed of Horizontal Propagation (ASHP) m/s'
    # txt = 'Propagation Speed (mm/ms)'
    # ax.set_xlabel(txt)
    # ax.set_ylabel('Nb of measures')
    # ax.legend()
   
    for ax in fig.get_axes():
        for spine in ['top', 'right']:
            ax.spines[spine].set_visible(False)

    fig.tight_layout()
    lims = ax.get_xlim()
    ax.set_xlim(0, lims[1])
    if anot:
        date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        fig.text(0.99, 0.01, 'speedBaudot.py:plot_both',
                 ha='right', va='bottom', alpha=0.4)
        fig.text(0.01, 0.01, date, ha='left', va='bottom', alpha=0.4)
        # fig.text(0.5, 0.01, 'cortical speed',
        #          ha='center', va='bottom', alpha=0.4)
    return fig


plt.close('all')
fig = plot_both(bined_df)
save = False
if save:
    file = 'optSpreedBoth.pdf'
    dirname = os.path.join(paths['owncFig'], 'pythonPreview', 'baudot')
    filename = os.path.join(dirname, file)
    fig.savefig(filename)
    #update current
    dirname = os.path.join(paths['owncFig'], 'pythonPreview', 'current', 'fig')
    file = 'f9_optSpreedBoth'
    for ext in ['.png', '.pdf']:
        filename = os.path.join(dirname, (file + ext))
        fig.savefig(filename)


#%%
# targetbins = list(np.linspace(0, 1, 21))
# targetbins = [round(_, 2) for _ in targetbins]


def hist_summary(brdf, summary_df, cgdf, df, gmdf, maxx=1):
    """
    histogramme distribution of the main results from the lab

    """
    def compute_speed_histo(speeds, bins=40):
        #speeds = df.speed/1000
        height, x = np.histogram(speeds, bins=bins, range=(0,1))
        # normalize
        height = height/np.sum(height)
        width = (max(x) - min(x))/(len(x) -1)
        return x[:-1], height, width

    fig, axes = plt.subplots(figsize=(8.6,12), nrows=2, ncols=3,
                             sharey=True, sharex=True)
    # ax.bar(x[:-1], height, width=width, color='tab:red', alpha=0.6)
    axes = axes.flatten()

    # bringuier flash
    ax = axes[0]
    x = brdf.speed_lower.tolist()
    x = [round(_, 2) for _ in x]
    height_bar = (brdf.impulse)/brdf.impulse.sum().tolist()
    align = 'edge'
    x[-1] = 1.05     # for continuous range
    width = max(x)/(len(x) -1)
    ax.bar(x, height=height_bar, width=width, align=align, alpha=1,
           color='tab:green', edgecolor='k', label='impulse bringuier')
    txt = 'n = 37 \n ({} measures)'.format(brdf.impulse.sum())
    ax.text(x=0.6, y= 0.8, s=txt, va='top', ha='left', transform=ax.transAxes)
    moy = (brdf.impulse * (brdf.speed_lower+0.025))[:-1].sum()/brdf.impulse.sum()
    txt = 'mean ~ {:.2f}'.format(moy)
    ax.text(x=0.7, y= 0.6, s=txt, va='top', ha='center', color='tab:green',
            transform=ax.transAxes)
    ax.axvline(moy, color='tab:green')
    ax.legend()

    # bringuier bar
    ax = axes[1]
    x = brdf.speed_lower.tolist()
    x = [round(_, 2) for _ in x]
    height_bar = (brdf.long_bar)/brdf.long_bar.sum().tolist()
    align = 'edge'
    x[-1] = 1.05     # for continuous range
    width = max(x)/(len(x) -1)
    ax.bar(x, height=height_bar, width=width, align=align, alpha=1,
           color='tab:blue', edgecolor='k', label='bar bringuier')
    txt = 'n = 27 \n ({} measures)'.format(brdf.long_bar.sum())
    ax.text(x=0.6, y= 0.8, s=txt, va='top', ha='left', transform=ax.transAxes)
    moy = (brdf.long_bar * (brdf.speed_lower+0.025)[:-1]).sum()/brdf.long_bar.sum()
    txt = 'mean ~ {:.2f}'.format(moy)
    ax.text(x=0.7, y= 0.6, s=txt, va='top', ha='center', color='tab:blue',
            transform=ax.transAxes)
    ax.axvline(moy, color='tab:blue')
    ax.legend()

    # baudot
    ax = axes[2]
    x = summary_df.index/1000
    height = summary_df.bd_cells / summary_df.bd_cells.sum()
    width = (max(x) - min(x))/(len(x) -1)
    # width = 0.02
    ax.bar(x, height, width=width, color='tab:purple', edgecolor='k', alpha=0.8,
           label='baudot')
    txt = 'n = {}'.format(int(summary_df.bd_cells.sum()))
    ax.text(x=0.6, y= 0.8, s=txt, va='top', ha='left', transform=ax.transAxes)
    moy = (summary_df.bd_cells * x).sum()/summary_df.bd_cells.sum()
    txt = 'mean ~ {:.2f}'.format(moy)
    ax.text(x=0.7, y= 0.6, s=txt, va='top', ha='center', color='tab:purple',
            transform=ax.transAxes)
    ax.axvline(moy, color='tab:purple')
    ax.legend()

    # gerard mercier
    ax = axes[3]
    x = gmdf.speed_low/1000
    height = gmdf.cells/ gmdf.cells.sum()
    width = (max(x) - min(x))/(len(x) -1)
    ax.bar(x, height, width=width, color='tab:brown', edgecolor='k', align='edge',
       alpha=0.8, label='gmercier')
    txt = 'n = {}'.format(gmdf.cells.sum())
    ax.text(x=0.6, y= 0.8, s=txt, va='top', ha='left', transform=ax.transAxes)
    moy = (gmdf.cells * x).sum()/gmdf.cells.sum()
    txt = 'mean ~ {:.2f}'.format(moy)
    ax.text(x=0.7, y= 0.6, s=txt, va='top', ha='center', color='tab:brown',
            transform=ax.transAxes)
    ax.axvline(moy, color='tab:brown')
    ax.legend()

    # centripop
    ax = axes[4]
    ax.bar(*compute_speed_histo(df.speed/1000), color='tab:red', edgecolor='k',
           alpha=0.8, label='centrigabor population')
    txt = 'n = {}'.format(len(df))
    ax.text(x=0.6, y= 0.8, s=txt, va='top', ha='left', transform=ax.transAxes)

    txt = 'mean ± std : \n {:.2f} ± {:.2f}'.format(
        (df.speed/1000).mean(), (df.speed/1000).std())
    ax.text(x=0.7, y= 0.7, s=txt, va='top', ha='center', color='tab:red',
            transform=ax.transAxes)
    ax.axvline((df.speed/1000).mean(), color='tab:red')
    ax.legend()

    # speed pop
    ax = axes[5]
    ax.bar(*compute_speed_histo(cgdf.speed_isi0/1000), color='tab:orange',
           edgecolor='k', alpha=0.8, label='speed population')
    txt = 'n = {}'.format(len(cgdf))
    ax.text(x=0.6, y= 0.8, s=txt, va='top', ha='left', transform=ax.transAxes)
    txt = 'mean ± std : \n {:.2f} ± {:.2f}'.format(
        (cgdf.speed_isi0/1000).mean(), (cgdf.speed_isi0/1000).std())
    ax.text(x=0.7, y= 0.7, s=txt, va='top', ha='center',
            color='tab:orange', transform=ax.transAxes)
    ax.axvline((cgdf.speed_isi0/1000).mean(), color='tab:orange')
    ax.legend()

    fig.suptitle('summary for speed')
    ax.set_xlim(0, maxx)
    for ax in fig.get_axes():
        for spine in ['left', 'top', 'right']:
            ax.spines[spine].set_visible(False)
            ax.tick_params(left=False, labelleft=False)
    for ax in axes[2:]:
        ax.set_xlabel('speed (m/s)')

    if anot:
        date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        fig.text(0.99, 0.01, 'speedBaudot.py:hist_summary',
                 ha='right', va='bottom', alpha=0.4)
        fig.text(0.01, 0.01, date, ha='left', va='bottom', alpha=0.4)

    fig.tight_layout()
    return fig

plt.close('all')

fig1 = hist_summary(brdf, summary_df, spdf, popdf, gmdf)
fig2 = hist_summary(brdf, summary_df, spdf, popdf, gmdf, maxx=0.5)
save = False
if save:
    file = 'hist_summary.pdf'
    dirname = os.path.join(paths['owncFig'], 'pythonPreview', 'baudot')
    filename = os.path.join(dirname, file)
    fig1.savefig(filename)
    file = 'hist_summary05.pdf'
    dirname = os.path.join(paths['owncFig'], 'pythonPreview', 'baudot')
    filename = os.path.join(dirname, file)
    fig2.savefig(filename)


#%% summary dot plots

plt.rcParams['axes.xmargin'] = 0.05
plt.rcParams['axes.ymargin'] = 0.05
plt.close('all')

def dotPlotLatency(df):
    fig = plt.figure(figsize=(6,8))
    ax = fig.add_subplot(111)
    # y = df.index.tolist()
    y = (bined_df.index + 0.025).tolist()
    populations = ['br_long_bar', 'br_impulse', 'bd',
                   'gm', 'cgpop', 'cgspeed']
    popSigni = ['bringuier_bars', 'bringuier_impulses', 'baudot',
              'gerardMercier', 'centrigabor_pop', 'centrigabor_speedPop']
    labels = dict(zip(populations, popSigni))
    colors = ['tab:green', 'tab:blue', 'tab:purple',
              'tab:brown', 'tab:red', 'tab:orange']
    for i, pop in enumerate(populations):
        x = (df[pop] / df[pop].sum()).tolist()
        x = list(map(lambda x: x if x>0 else np.nan, x))
        ax.plot(x, y, 'o',
                markeredgecolor='w', markerfacecolor=colors[i],
                alpha=0.6, markeredgewidth=1.5, markersize=16,
                label=labels[pop])
    ax.legend()
    ax.set_xlim(0, ax.get_xlim()[1])
    ax.set_ylim(0, ax.get_ylim()[1])

    fig.suptitle('latency (proportion of cells)')
    ax.set_ylabel('optimal speed (m/sec)')
    # ax.set_xlabel('proportion of cells')´
    ticks = np.linspace(0,1,11)
    ax.set_yticks(ticks)
    for spine in ['right', 'top']:
        ax.spines[spine].set_visible(False)
    if anot:
        date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        fig.text(0.99, 0.01, 'speedBaudot.py:dotplotLatency',
                 ha='right', va='bottom', alpha=0.4)
        fig.text(0.01, 0.01, date, ha='left', va='bottom', alpha=0.4)
    fig.tight_layout()
    return fig

fig = dotPlotLatency(bined_df)
save = False
if save:
    file = 'dotplotLatency.pdf'
    dirname = os.path.join(paths['owncFig'], 'pythonPreview', 'baudot')
    filename = os.path.join(dirname, file)
    fig.savefig(filename)



#%% fir histo
plt.close('all')

gdf = bined_df.copy()
fig, ax = plt.subplots(figsize=(7,6))
axT = ax.twinx()
txt = 'Bar n={:.0f}'.format(gdf.br_long_bar.sum())
x = gdf.index
ax.bar(x, height=gdf.br_long_bar, width=width, align=align, alpha=0.8,
           color=std_colors['blue'], edgecolor='k',
           label=txt)
txt = 'SN n={:.0f}'.format(gdf.br_impulse.sum())
ax.bar(x, height=gdf.br_impulse, bottom=gdf.br_long_bar, width=width,
    align=align, color=std_colors['green'],
    edgecolor='k', alpha=0.8, label=txt)
    # txt = 'Apparent Speed of Horizontal Propagation (ASHP) m/s'
txt = 'Propagation Speed (mm/ms)'
ax.set_xlabel(txt)
ax.set_ylabel('Nb of measures')
ax.legend()
# bottom kde
kde = stats.gaussian_kde(gdf.br_impulse)
xmin, xmax = ax.get_xlim()
x_kde = np.arange(xmin, xmax, 0.05)
axT.plot(x_kde, kde(x_kde), color=std_colors['green'],
        alpha=1, linewidth=2, linestyle='-')

fitfunc  = lambda p, x: p[0]*np.exp(-0.5*((x-p[1])/p[2])**2)+p[3]
errfunc  = lambda p, x, y: (y - fitfunc(p, x))

init  = [1.0, 0.5, 0.5, 0.5]

x_data = gdf.index.values
y_data = gdf.br_impulse

out   = leastsq( errfunc, init, args=(x_data, y_data))
c = out[0]
