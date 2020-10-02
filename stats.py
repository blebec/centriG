#!/usr/bin/env python3
# -*- coding: utf-8 -*-

""" cross time/indice """

import os
from datetime import datetime
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

import config
from load import load_data as ldat


paths = config.build_paths()
paths['save'] = os.path.join(paths['owncFig'], 'pythonPreview', 'stat')
std_colors = config.std_colors()
anot = True

plt.rcParams['axes.xmargin'] = 0.05
plt.rcParams['axes.ymargin'] = 0.05


#%% extract stats values

def build_sigpop_statdf(amp='engy'):
    """
    load the indices and extract descriptive statistics per condition
    NB sig cell = individual sig for latency OR energy
    input : amp in [gain, engy]
    output:
            pandas dataframe
            sigcells : dictionary of sigcells namesper stim condition
    """
    df = pd.DataFrame()
    sigcells = {}
    for mes in ['vm', 'spk']:
        data = ldat.load_cell_contributions(kind=mes, amp=amp, age='new')
        cols = [item for item in data.columns if not item.endswith('_sig')]
        # conditions and parameter lists
        conds = []
        for item in [st.split('_')[0] for st in cols]:
            if item not in conds:
                conds.append(item)
        params = []
        for item in [st.split('_')[1] for st in cols]:
            if item not in params:
                params.append(item)
        # build dico[cond]list of sig cells
        cells_dict = dict()
        for cond in conds:
            # select cell signicant for at least one of the param
            sig_cells = set()
            for param in params:
                col = cond + '_' + param
                sig_df = data.loc[data[col+'_sig'] > 0, [col]]
                sig_cells = sig_cells.union(sig_df.loc[sig_df[col] > 0].index)
            cells_dict[cond] = list(sig_cells)
        # extract descriptive stats
        stats= []
        for col in cols:
            cells = cells_dict[col.split('_')[0]]
            ser = data.loc[cells[:], col]# col = cols[0]
            dico = {}
            dico[mes + '_count'] = ser.count()
            dico[mes + '_mean'] = ser.mean()
            dico[mes + '_std'] = ser.std()
            dico[mes + '_sem'] = ser.sem()
            dico[mes + '_med'] = ser.median()
            dico[mes + '_mad'] = ser.mad()
            stats.append(pd.Series(dico, name=col))
        df = pd.concat([df, pd.DataFrame(stats)], axis=1)
        df = df.fillna(0)
        sigcells[mes] = cells_dict.copy()
    return df, sigcells


# stat_df = build_sigpop_statdf()

#%%%%  to build the stat representation
plt.close('all')

# check error bars

def build_stat_df(sig=False, amp='engy'):
    """
    extract a statistical description
    in this approach sig cells refers for individually sig for the given
    parameter (aka advance or energy)

    """
    df = pd.DataFrame()
    for mes in ['vm', 'spk']:
        # mes = 'spk'
        data = ldat.load_cell_contributions(kind=mes, amp=amp, age='new')
        cols = [item for item in data.columns if not item.endswith('_sig')]
        #only sig cells, independtly for each condition and each measure
        if sig:
            stats= []
            for col in cols:
                # col = cols[0]
                sig_df = data.loc[data[col+'_sig'] > 0, [col]]
                #only positive values
                sig_df = sig_df.loc[sig_df[col] > 0]
                dico = {}
                dico[mes + '_count'] = sig_df[col].count()
                dico[mes + '_mean'] = sig_df[col].mean()
                dico[mes + '_std'] = sig_df[col].std()
                dico[mes + '_sem'] = sig_df[col].sem()
                dico[mes + '_med'] = sig_df[col].median()
                dico[mes + '_mad'] = sig_df[col].mad()
                stats.append(pd.Series(dico, name=col))
            df = pd.concat([df, pd.DataFrame(stats)], axis=1)
        # all cells
        else:
            df[mes + '_count'] = data[cols].count()
            df[mes + '_mean'] = data[cols].mean()
            df[mes + '_std'] = data[cols].std()
            df[mes + '_sem'] = data[cols].sem()
            df[mes + '_med'] = data[cols].median()
            df[mes + '_mad'] = data[cols].mad()
    # replace nan by 0
    #(no sig cell or only one sig cell -> nan for all params or std)
    df = df.fillna(0)
    return df

def plot_stat(statdf, kind='mean', legend=False):
    """
    plot the stats
    input : statdf, kind in ['mean', 'med'], loc in ['50', 'peak', 'energy']
    output : matplotlib figure
    """
    if kind == 'mean':
        stat = ['_mean', '_sem']
    elif kind == 'med':
        stat = ['_med', '_mad']
    else:
        print('non valid kind argument')
        return

    colors = [std_colors['red'], std_colors['green'],
              std_colors['yellow'], std_colors['blue'],
              std_colors['dark_blue']]
    
    ref = ''
    if statdf.max().max() == 37:
        ref = 'population'
    else:
        ref = 'sub_populations'

    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(8, 8), 
                             sharex=True, sharey=True)
    axes = axes.flatten()
    title = "{}    ({} ± {}) ".format(ref, stat[0][1:], stat[1][1:])
    fig.suptitle(title)
    # plots
    for i, cond in enumerate([('vm', 'sect'), ('vm', 'full'),
                              ('spk', 'sect'), ('spk', 'full')]):
        ax = axes[i]
        rec = cond[0]
        spread = cond[1]
        ax.set_title('{} {}'.format(rec, spread))
        # select spread (sect, full)
        rows = [st for st in statdf.index.tolist() if spread in st]
        # append random full
        if spread == 'sect':
            rows.extend(
                [st for st in stat_df.index if st.startswith('rdisofull')])
        # df indexes (for x and y)
        time_rows = [st for st in rows if 'time50' in st]
        y_rows = [st for st in rows if 'engy' in st]
        cols = [col for col in statdf.columns if col.startswith(rec)]
        cols = [st for st in cols if stat[0] in st or stat[1] in st]
        #labels
        labels = [st.split('_')[0] for st in y_rows]
        # values (for x an y)
        x = statdf.loc[time_rows, cols][rec + stat[0]].values
        xerr = statdf.loc[time_rows, cols][rec + stat[1]].values
        y = statdf.loc[y_rows, cols][rec + stat[0]].values
        yerr = statdf.loc[y_rows, cols][rec + stat[1]].values
        #plot
        for xi, yi, xe, ye, ci, lbi  in zip(x, y, xerr, yerr, colors, labels):
            ax.errorbar(xi, yi, xerr=xe, yerr=ye,
                        fmt='s', color=ci, label=lbi)
        if legend:
            ax.legend()

    #adjust
    for i, ax in enumerate(axes):
        ax.axvline(0, linestyle='-', alpha=0.4)
        ax.axhline(0, linestyle='-', alpha=0.4)
        for spine in ['top', 'right']:
            ax.spines[spine].set_visible(False)
            if i % 2 == 0:
                ax.set_ylabel('energy')
            else:
                ax.spines['left'].set_visible(False)
                ax.yaxis.set_visible(False)
            if i > 1:
                ax.set_xlabel('time advance (ms)')
            else:
                ax.spines['bottom'].set_visible(False)
                ax.xaxis.set_visible(False)
    # ax = axes[2]
    # custom_ticks = np.linspace(-2, 2, 3, dtype=int)/10
    # ax.set_yticks(custom_ticks)
    # ax.set_yticklabels(custom_ticks)

    fig.tight_layout()
    fig.subplots_adjust(hspace=0.02)
    fig.subplots_adjust(wspace=0.02)
    if anot:
        date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        fig.text(0.99, 0.01, 'stat.py:plot_stat',
                 ha='right', va='bottom', alpha=0.4)
        fig.text(0.01, 0.01, date, ha='left', va='bottom', alpha=0.4)
    return fig

#%%%%%
plt.close('all')

def extract_values(df, stim='sect', param='time'):
    """ extract pop and response dico:
        input :
            dataframe
            stim in [sect, full]
            param in [timme, gain, engy]
        return:
            pop_dico -> per condition [popNb, siginNb, %]
            resp_dico -> per condition [moy, moy+sem, moy-sem]
    """
    # stim = '_' + stim_kind + '_'
    # mes = '_d' + measure + '50'
    # restrict df
    restricted_list = \
    [st for st in df.columns if stim in st and param in st]
    adf = df[restricted_list]
    #compute values
    records = [item for item in restricted_list if 'sig' not in item]
    pop_dico = {}
    resp_dico = {}
    for cond in records:
        signi = cond + '_sig'
        pop_num = len(adf)
        extract = adf.loc[adf[signi] > 0, cond].copy()
        extract = extract[extract > 0]
        signi_num = len(extract)
        percent = round((signi_num / pop_num) * 100)
        leg_cond = cond.split('_')[0]
        pop_dico[leg_cond] = [pop_num, signi_num, percent]
        # descr
        moy = extract.mean()
        sem = extract.sem()
        resp_dico[leg_cond] = [moy, moy + sem, moy - sem]
    return pop_dico, resp_dico

def autolabel(ax, rects, sup=False):
    """
    attach the text labels to the rectangles
    """
    for rect in rects:
        x = rect.get_x() + rect.get_width()/2
        height = rect.get_height()
        y = height - 1
        if y < 3 or sup:
            y = height + 1
            ax.text(x, y, '%d' % int(height) + '%',
                    ha='center', va='bottom')
        else:
            ax.text(x, y, '%d' % int(height) + '%',
                    ha='center', va='top')

def plot_cell_contribution(df, kind=''):
    """
    plot the number of significative cells contributing to the response
    """

    colors = [std_colors[item] for item in ['red', 'green', 'yellow', 'blue']]
    dark_colors = [std_colors[item] for item in \
                   ['dark_red', 'dark_green', 'dark_yellow', 'dark_blue']]


    conds = [('sect', 'time'), ('sect', 'engy'),
             ('full', 'time'), ('full', 'engy')]
    titles = {'time' : r'$\Delta$ Time (% significant cells)',
              'engy': r'Energy (% significant cells)',
              'sect': 'Sector',
              'full': 'Full'}
    fig, axes = plt.subplots(nrows=2, ncols=2, sharey=True, figsize=(8,8))
    axes = axes.flatten()
    for i, ax in enumerate(axes):
        ax.set_title(str(i))
        stim, mes = conds[i]
        ax.set_title(titles[mes], pad=0)
        pop_dico, resp_dico = extract_values(df, stim, mes)
        x = pop_dico.keys()
        heights = [pop_dico[item][-1] for item in pop_dico.keys()]
        bars = ax.bar(x, heights, color=colors, width=0.95, alpha=0.8,
                      edgecolor=colors)
        autolabel(ax, bars) # call
        ax.set_ylabel(titles[stim])
    for ax in axes:
        for spine in ['left', 'top', 'right']:
            ax.spines[spine].set_visible(False)
            ax.tick_params(axis='x', labelrotation=45)
            ax.yaxis.set_ticklabels([])
            ax.tick_params(axis='y', length=0)
    for ax in axes[:2]:
        ax.xaxis.set_visible(False)
    txt = "{} ({} cells)".format(kind, len(data))
    fig.text(0.43, 0.90, txt, ha='center', va='top', fontsize=18)
    if anot:
        date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        fig.text(0.99, 0.01, 'stat.py:plot_cell_contribution',
                 ha='right', va='bottom', alpha=0.4)
        fig.text(0.01, 0.01, date, ha='left', va='bottom', alpha=0.4)

    fig.tight_layout()
    return fig

save = False
for kind in ['vm', 'spk']:
    #load_cell_contributions(kind='vm', amp='gain', age='new'):
    data = ldat.load_cell_contributions(kind, age='new', amp='engy')
    fig = plot_cell_contribution(data, kind)
    if save:
        filename = os.path.join(paths['save'], kind + '_cell_contribution.png')
        fig.savefig(filename)

#%%
plt.close('all')

stat_df = build_stat_df()
stat_df_sig, sig_cells = build_sigpop_statdf()
fig1 = plot_stat(stat_df, kind='mean')
# fig2 = plot_stat(stat_df, kind='med')
fig3 = plot_stat(stat_df_sig, kind='mean')
# fig4 = plot_stat(stat_df_sig, kind='med')
save = False
if save:
    filename = os.path.join(paths['save'], 'meanSem.png')
    fig1.savefig(filename)
    # filename = os.path.join(paths['save'], 'medMad.png')
    # fig2.savefig(filename)
    filename = os.path.join(paths['save'], 'sig_meanSem.png')
    fig3.savefig(filename)
    # filename = os.path.join(paths['save'], 'sig_medMad.png')
    # fig4.savefig(filename)



#%% stat composite figure (top row = pop; bottom row = sig_pop)

def plot_composite_stat(statdf, statdfsig, sigcells,
                        kind='mean', amp='engy', mes='vm'):
    """
    plot the stats
    input : statdf, kind in ['mean', 'med'], loc in ['50', 'peak', 'energy']
    output : matplotlib figure
    here combined top = full pop, bottom : significative pop
    """
    if kind == 'mean':
        stat = ['_mean', '_std']
    elif kind == 'med':
        stat = ['_med', '_mad']
    else:
        print('non valid kind argument')
        return

    colors = [std_colors['red'], std_colors['green'],
              std_colors['yellow'], std_colors['blue'],
              std_colors['dark_blue']]

    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(8,8),
                             sharex=True, sharey=True)
    axes = axes.flatten()
    title = stat[0][1:] +'   (' +  stat[1][1:] + ')'
    fig.suptitle(title)
    for i, cond in enumerate([(mes, 'sect'), (mes, 'full'),
                               (mes, 'sect'), (mes, 'full')]):
        if i < 2:
            df = statdf
            pop = 'pop'
        else:
            df = statdfsig
            pop = 'sig_pop'
        ax = axes[i]
        rec = cond[0]
        spread = cond[1]
        ax_title = '{} ({} {})'.format(pop, rec, spread)
        ax.set_title(ax_title)
        # select spread (sect, full)
        rows = [st for st in df.index.tolist() if spread in st]
        # append random full if sector
        if spread == 'sect':
            rows.extend(
                [st for st in stat_df.index if st.startswith('rdisofull')])
        # df indexes (for x and y)
        time_rows = [st for st in rows if 'time50' in st]
        y_rows = [st for st in rows if amp in st]
        # y_rows = [st for st in rows if 'engy' in st]
        cols = [col for col in df.columns if col.startswith(rec)]
        cols = [st for st in cols if stat[0] in st or stat[1] in st]
        #labels
        labels = [st.split('_')[0] for st in y_rows]
        # values (for x an y)
        x = df.loc[time_rows, cols][rec + stat[0]].values
        xerr = df.loc[time_rows, cols][rec + stat[1]].values
        y = df.loc[y_rows, cols][rec + stat[0]].values
        yerr = df.loc[y_rows, cols][rec + stat[1]].values
        #plot
        for xi, yi, xe, ye, ci, lbi  in zip(x, y, xerr, yerr, colors, labels):
            ax.errorbar(xi, yi, xerr=xe, yerr=ye,
                        fmt='s', color=ci, label=lbi)
        ax.legend()

    #adjust
    for i, ax in enumerate(axes):
        ax.axvline(0, linestyle='-', alpha=0.4)
        ax.axhline(0, linestyle='-', alpha=0.4)
        for spine in ['top', 'right']:
            ax.spines[spine].set_visible(False)
            if i % 2 == 0:
                ax.set_ylabel(amp)
#                ax.set_ylabel('energy')
            else:
                ax.spines['left'].set_visible(False)
                ax.yaxis.set_visible(False)
            if i > 1:
                ax.set_xlabel('time advance (ms)')
            else:
                ax.spines['bottom'].set_visible(False)
                ax.xaxis.set_visible(False)
    fig.tight_layout()
    fig.subplots_adjust(hspace=0.02)
    fig.subplots_adjust(wspace=0.02)
    if anot:
        date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        fig.text(0.99, 0.01, 'stat.py:plot_stat',
                 ha='right', va='bottom', alpha=0.4)
        fig.text(0.01, 0.01, date, ha='left', va='bottom', alpha=0.4)
    return fig

mes = ['vm', 'spk'][0]
amp = ['gain', 'engy'][1]
kind = ['mean', 'med'][0]
stat_df = build_stat_df(amp=amp)                        # append gain to load
stat_df_sig, sig_cells = build_sigpop_statdf(amp=amp)   # append gain to load
fig1 = plot_composite_stat(stat_df, stat_df_sig, sig_cells,
                           kind=kind, amp=amp, mes=mes)
save = True
if save:
    filename = os.path.join(paths['save'], mes + amp.title() + '_composite_meanStd.png')
    fig1.savefig(filename)

#%% composite cell contribution
plt.close('all')

def plot_composite_cell_contribution(df, sigcells, kind='', amp='engy'):
    """
    cell contribution, to go to the bottom of the preceding stat description
    """

    colors = [std_colors[item] for item in ['red', 'green', 'yellow', 'blue']]
    dark_colors = [std_colors[item] for item in \
                   ['dark_red', 'dark_green', 'dark_yellow', 'dark_blue']]

    stims = ('sect', 'full')
    params = ('time', amp)
    titles = {'time' : r'$\Delta$ Time (% significant cells)',
              'engy': r'Energy (% significant cells)',
              'sect': 'Sector',
              'full': 'Full'}
    fig, axes = plt.subplots(nrows=1, ncols=2, sharey=True, figsize=(8,4))
    axes = axes.flatten()
    titles = ['sector', 'full']

    for i, ax in enumerate(axes):
        # ax.set_title(str(i))
        stim = stims[i]
        param = params[0]
        #for param in params
        ax.set_title(titles[i], pad=0)
        heights = []
        for param in params:
            pop_dico, resp_dico = extract_values(df, stim, param)
            height = [pop_dico[key][-1] for key in pop_dico]
            heights.append(height)
        # % sig cells for time and amp
        height = [round(len(sigcells[kind][st])/len(df)*100)
                  for st in list(pop_dico.keys())]
        heights.append(height)
        x = np.arange(len(pop_dico.keys()))
        width = 0.45
        # time
        bars = ax.bar(x - width/2, heights[0], color=colors, width=width, alpha=0.4,
                      edgecolor=colors)
        autolabel(ax, bars) # call
        # amp
        bars = ax.bar(x + width/2, heights[1], color=colors, width=width, alpha=0.4,
                      edgecolor=colors)
        autolabel(ax, bars) # call
        # time OR amp
        bars = ax.bar(x, heights[2], color=colors, width=0.15, alpha=0.8,
                      edgecolor=dark_colors)
        autolabel(ax, bars, sup=True) # call
        labels = list(pop_dico.keys())
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels)
    for ax in axes:
        for spine in ['left', 'top', 'right']:
            ax.spines[spine].set_visible(False)
            ax.tick_params(axis='x', labelrotation=45)
            ax.yaxis.set_ticklabels([])
            ax.tick_params(axis='y', length=0)
    # for ax in axes[:2]:
    #     ax.xaxis.set_visible(False)
    txt = "{} ({} cells)     [time | {}]".format(kind, len(data), amp)
    fig.text(0.5, 0.97, txt, ha='center', va='top', fontsize=14)
    if anot:
        date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        fig.text(0.99, 0.01, 'stat.py:plot_cell_contribution',
                 ha='right', va='bottom', alpha=0.4)
        fig.text(0.01, 0.01, date, ha='left', va='bottom', alpha=0.4)
    fig.tight_layout()
    return fig


save = False
amp = ['gain', 'engy'][0]
for kind in ['vm', 'spk']:
    #load_cell_contributions(kind='vm', amp='gain', age='new'):
    data = ldat.load_cell_contributions(kind, age='new', amp=amp)
    stat_df_sig, sig_cells = build_sigpop_statdf(amp=amp)
    fig = plot_composite_cell_contribution(data, sig_cells, kind=kind,  amp=amp)
    if save:
        filename = os.path.join(paths['save'], kind + amp.title() \
                                + '_composite_cell_contribution.png')
        fig.savefig(filename)
