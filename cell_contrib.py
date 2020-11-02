#!/usr/bin/env python3
# -*- coding: utf-8 -*-

""" statistical extraction of cell properties """

import os
from datetime import datetime
from imp import reload

import numpy as np
# import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

import config
from load import load_data as ldat

paths = config.build_paths()
paths['save'] = os.path.join(paths['owncFig'], 'pythonPreview', 'stat')
std_colors = config.std_colors()
anot = True

plt.rcParams['axes.xmargin'] = 0.05
plt.rcParams['axes.ymargin'] = 0.05

#%%%%% cell contribution

def extract_values(df, stim='sect', param='time', replaceFull=True):
    """ extract pop and response dico:
        input :
            dataframe
            stim in [sect, full]
            param in [timme, gain, engy]
        return:
            pop_dico -> per condition [popNb, siginNb, %]
            resp_dico -> per condition [moy, moy+sem, moy-sem]
    """
    adf = df.copy()
    if 'fill' in param:
        fills = [item for item in adf.columns if 'fill' in item]
        # create zero padded value columns
        while fills:
            fill = fills.pop()
            col = '_'.join(fill.split('_')[:-1])
            if col not in adf.columns:
                adf[col] = data[fill]
                adf[col] = 0
    restricted_list = \
    [st for st in adf.columns if stim in st and param in st]
    if replaceFull:
        # replace rdisosect by rdisofull
        if 'rdisosect' in set(item.split('_')[0] for item in restricted_list):
            restricted_list = [st for st in restricted_list
                               if 'rdisosect' not in st.split('_')[0]]
            restricted_list.extend(
                    [st for st in adf.columns
                     if 'rdisofull' in st and param in st])
    else:
        # append full:
        if 'rdisosect' in {item.split('_')[0] for item in restricted_list}:
            restricted_list.extend(
            [st for st in adf.columns if 'rdisofull' in st and param in st])

    adf = adf[restricted_list]
    #compute values
    # records = [item for item in restricted_list if 'sig' not in item]
    # to maintain the order
    records = [item.replace('_sig', '')
               for item in restricted_list if 'sig' in item]

    pop_dico = {}
    resp_dico = {}
    for cond in records:
        signi = cond + '_sig'
        pop_num = len(adf)
        # significant
        extract = adf.loc[adf[signi] > 0, cond].copy()
        # on ly positive measures
        extract = extract[extract >= 0]
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


#@config.profile
def plot_cell_contribution(df, kind=''):
    """
    plot the number of significative cells contributing to the response
    kind in [vm, spk]
    """

    colors = [std_colors[item] for item in ['red', 'green', 'yellow', 'blue']]

    conds = [(a, b) for a in ['sect', 'full'] for b in ['time', 'engy']]
    titles = dict(time = 'Time Advance (% significant cells)',
                  engy = r'$\Delta$ Energy (% significant cells)',
                  sect = 'Sector',
                  full = 'Full')
    fig, axes = plt.subplots(nrows=2, ncols=2, sharey=True, figsize=(8,8))
    axes = axes.flatten()
    for i, ax in enumerate(axes):
        ax.set_title(str(i))
        stim, mes = conds[i]
        ax.set_title(titles[mes], pad=0)
        pop_dico, _ = extract_values(df, stim, mes)
        x = list(pop_dico.keys())
        heights = [pop_dico[item][-1] for item in pop_dico.keys()]
        bars = ax.bar(x[:4], heights[:4], color=colors, width=0.95, alpha=0.8,
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
        fig.text(0.99, 0.01, 'cellContribution:plot_cell_contribution',
                 ha='right', va='bottom', alpha=0.4)
        fig.text(0.01, 0.01, date, ha='left', va='bottom', alpha=0.4)

    fig.tight_layout()
    return fig

plt.close('all')

save = False
for mes in ['vm', 'spk']:
    #load_cell_contributions(mes='vm', amp='gain', age='new'):
    data = ldat.load_cell_contributions(rec=mes, amp='engy', age='new')
    fig = plot_cell_contribution(data, mes)
    if save:
        filename = os.path.join(paths['save'], 'pop',
                                mes + '_cell_contribution.png')
        fig.savefig(filename)

#%% composite cell contribution

def plot_composite_sectFull_2X1(df, sigcells, kind='', amp='engy'):
    """
    cell contribution, to go to the bottom of the preceding stat description
    kind in [vm, spk]
    amp in [gain, engy]
    """

    colors = [std_colors[item] for item in
              ['red', 'green', 'yellow', 'blue']]
    dark_colors = [std_colors[item] for item in \
                   ['dark_red', 'dark_green', 'dark_yellow', 'dark_blue']]

    stims = ('sect', 'full')
    params = ('time', amp)
    titles = dict(time = r'Time Advance (% significant cells)',
                  engy = r'$\Delta$ Energy (% significant cells)',
                  sect = 'Sector',
                  full = 'Full')
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
            pop_dico, _ = extract_values(df, stim, param)
            height = [pop_dico[key][-1] for key in pop_dico]
            # remove the fifth value ie if sect = rdisofull
            heights.append(height[:4])
        # % sig cells for time and amp
        height = [round(len(sigcells[kind][st])/len(df)*100)
                  for st in list(pop_dico.keys())]
        # remove the fifth value ie if sect = rdisofull
        heights.append(height[:4])
        x = np.arange(len(pop_dico.keys()))[:4]
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
        labels = list(pop_dico.keys())[:4]
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
    txt = "{} ({} cells) [time|U|{}]".format(kind, len(data), amp)
    fig.text(0.5, 0.99, txt, ha='center', va='top', fontsize=14)
    if anot:
        date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        fig.text(1, 0.01, 'cellContribution.py:plot_composite_sectFull_2X1',
                 ha='right', va='bottom', alpha=0.4)
        fig.text(0.01, 0.01, date, ha='left', va='bottom', alpha=0.4)
    fig.tight_layout()
    return fig

plt.close('all')
save = False
amp = ['gain', 'engy'][1]
stat_df_sig, sig_cells = ldat.build_sigpop_statdf(amp=amp)
for mes in ['vm', 'spk']:
    #load_cell_contributions(mes='vm', amp='gain', age='new'):
    data = ldat.load_cell_contributions(mes, age='new', amp=amp)
    fig = plot_composite_sectFull_2X1(data, sig_cells, kind=mes,
                                               amp=amp)
    if save:
        filename = os.path.join(paths['save'], mes + amp.title() \
                                + '_composite_cell_contribution_2X1.png')
        fig.savefig(filename)

#%%

def plot_composite_1X1(df, sigcells, mes='vm', amp='engy',
                                         spread='sect'):
    """
    cell contribution, to go to the bottom of the preceding stat description
    """

    colors = [std_colors[item]
              for item in ['red', 'green', 'yellow', 'blue', 'dark_blue']]
    dark_colors = [std_colors[item] for item in \
                   ['dark_red', 'dark_green', 'dark_yellow', 'dark_blue']]

#    stims = ('sect', 'full')
    params = ('time', amp)
    titles = dict(time = r'Latency Advance (% significant cells)',
                  engy = r'$\Delta$ Energy (% significant cells)',
                  sect = 'Sector',
                  full = 'Full',
                  vm = 'Vm',
                  spk = 'Spikes')
    fig, ax = plt.subplots(nrows=1, ncols=1, sharey=True, figsize=(4,4))
    title = "{} {} ({} cells)".format(titles[mes], titles[spread],  len(data))
    fig.suptitle(title)
    # param = params[0]
    #for param in params
    # ax.set_title(titles[i], pad=0)
    heights = []
    for param in params:
        pop_dico, _ = extract_values(df, spread, param)
        height = [pop_dico[key][-1] for key in pop_dico]
        heights.append(height)
    # % sig cells for time and amp
    height = [round(len(sigcells[mes][st])/len(df)*100)
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
    for spine in ['left', 'top', 'right']:
        ax.spines[spine].set_visible(False)
        ax.tick_params(axis='x', labelrotation=45)
        ax.yaxis.set_ticklabels([])
        ax.tick_params(axis='y', length=0)
    # for ax in axes[:2]:
    #     ax.xaxis.set_visible(False)
    txt = "time|U|{}".format(amp)

    fig.text(0.5, 0.8, txt, ha='center', va='top', fontsize=14, alpha = 0.8)
    if anot:
        date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        fig.text(1, 0.01, 'cellContibution:plot_composite_1X1',
                 ha='right', va='bottom', alpha=0.4)
        fig.text(0.01, 0.01, date, ha='left', va='bottom', alpha=0.4)
    fig.tight_layout()
    return fig

plt.close('all')
save = False
amp = ['gain', 'engy'][1]
stat_df_sig, sig_cells = ldat.build_sigpop_statdf(amp=amp)
for mes in ['vm', 'spk']:
    data = ldat.load_cell_contributions(mes, age='new', amp=amp)
    for spread in ['sect', 'full']:
        fig = plot_composite_1X1(data, sig_cells,
                                  spread=spread, mes=mes, amp=amp)
        if save:
            file = 'contrib_1x1_' + mes.title() + spread.title() + '.png'
            folder = os.path.join(paths['owncFig'], 'pythonPreview',
                                  'stat', 'contrib1x1')
            filename = os.path.join(folder, file)
            fig.savefig(filename)

#%%

def plot_separate_1x3(df, sigcells, spread='sect', mes='vm', amp='engy'):
    """
    cell contribution, to go to the bottom of the preceding stat description
    """
    titles = dict(time = r'Latency Advance',
                  engy = r'$\Delta$ Energy',
                  sect = 'Sector',
                  full = 'Full',
                  vm = 'Vm',
                  spk = 'Spikes')
    colors = [std_colors[item]
              for item in ['red', 'green', 'yellow', 'blue', 'dark_blue']]
    relabel = dict(cpisosect = 'CP-ISO',
                   cfisosect = 'CF-ISO',
                   cpcxsect = 'CP-CROSS',
                   rdisosect = 'RND',
                   cpisofull = 'CP-ISO',
                   cfisofull = 'CF-ISO',
                   cpcxfull = 'CP-CROSS',
                   rdisofull = 'RND'
                   )
    #compute values ([time values], [amp values])
    params = ['time', amp]
    heights = []
    for param in params:
        pop_dico, _ = extract_values(df, spread, param)
        height = [pop_dico[key][-1] for key in pop_dico]
        heights.append(height)
    # insert union % sig cells for time and amp
    height = [round(len(sigcells[mes][st])/len(df)*100)
                  for st in list(pop_dico.keys())]
    heights.insert(1, height)

    fig, axes = plt.subplots(nrows=1, ncols=3, sharey=True, figsize=(12,4))
    axes = axes.flatten()
    titles_here = [titles['time'], 'time OR energy', titles['engy']]
    labels = [relabel[st] for st in pop_dico]

    for i, ax in enumerate(axes):
        ax.set_title(str(i))
        param = params[0]
        #for param in params
        ax.set_title(titles_here[i], pad=0)

        x = np.arange(len(heights[i]))
        width = 0.9
        if i in [0, 2]:
            bars = ax.bar(x, heights[i], color=colors, width=width, alpha=0.6,
                          edgecolor=colors)
        else:
            bars = ax.bar(x, heights[i], color=colors, width=width, alpha=0.9,
                          edgecolor='k', label=labels)
        autolabel(ax, bars) # call
        # labels = list(pop_dico.keys())
        ax.set_xticks([])
        ax.set_xticklabels([])
    for ax in axes:
        for spine in ['left', 'top', 'right']:
            ax.spines[spine].set_visible(False)
            # ax.tick_params(axis='x', labelrotation=45)
            ax.yaxis.set_ticklabels([])
            ax.tick_params(axis='y', length=0)
    # for ax in axes[:2]:
    #     ax.xaxis.set_visible(False)
    txt = "{} {} ({} cells) ".format(mes, spread, len(data))
    fig.text(0.5, 0.85, txt, ha='center', va='top', fontsize=14)
    fig.legend(handles=bars, labels=labels, loc='upper right')
    # rectangle
    box = True
    if box:
        ax = axes[1]
        x, x1 = ax.get_xlim()
        x1 -= x
        y, y1 = ax.get_ylim()
        y1 -= y
        rect = Rectangle(xy=(x, y), width=x1, height=y1,
                         fill=False , alpha=0.6, edgecolor='k', linewidth=10)
        ax.add_patch(rect)
    if anot:
        date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        fig.text(1, 0.01, 'cellContribution:plot_separate_1x3',
                 ha='right', va='bottom', alpha=0.4)
        fig.text(0.01, 0.01, date, ha='left', va='bottom', alpha=0.4)
    fig.tight_layout()
    return fig


plt.close('all')
save = False
amp = ['gain', 'engy'][1]
stat_df_sig, sig_cells = ldat.build_sigpop_statdf(amp=amp)
for mes in ['vm', 'spk']:
    data = ldat.load_cell_contributions(mes, age='new', amp=amp)
    for spread in ['sect', 'full']:
        fig = plot_separate_1x3(data, sig_cells,
                                spread=spread, mes=mes, amp=amp)
        if save:
            file = 'contrib' + mes.title() + spread.title() + 'Box.png'
            folder = os.path.join(paths['owncFig'],
                                  'pythonPreview', 'sorted', 'sorted&contrib')
            filename = os.path.join(folder, file)
            fig.savefig(filename)


#%% composite with filling in

def plot_composite_sectFull_2X1_fill(df, sigcells, kind='', amp='engy'):
    """
    cell contribution, to go to the bottom of the preceding stat description
    kind in [vm, spk]
    amp in [gain, engy]
    """

    colors = [std_colors[item] for item in ['red', 'green', 'yellow', 'blue']]
    dark_colors = [std_colors[item] for item in \
                   ['dark_red', 'dark_green', 'dark_yellow', 'dark_blue']]

    stims = ('sect', 'full')
    params = ('time', amp, 'fill')
    titles = dict(time = r'Time Advance (% significant cells)',
                  engy = r'$\Delta$ Energy (% significant cells)',
                  sect = 'Sector',
                  full = 'Full')
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
            pop_dico, _ = extract_values(df, stim, param)
            height = [pop_dico[key][-1] for key in pop_dico]
            # remove the fifth value ie if sect = rdisofull
            heights.append(height[:4])
        # % sig cells for time and amp
        height = [round(len(sigcells[kind][st])/len(df)*100)
                  for st in list(pop_dico.keys())]
        # remove the fifth value ie if sect = rdisofull
        heights.append(height[:4])
        x = np.arange(len(pop_dico.keys()))[:4]
        width = 0.3

        # union
        bars = ax.bar(x, heights[3], color=colors, width=0.9, alpha=0.2,
                      edgecolor='k')
        autolabel(ax, bars, sup=True) # call
        # time
        bars = ax.bar(x - width, heights[0], color=colors, width=width, alpha=0.4,
                      edgecolor=colors)
        autolabel(ax, bars) # call
        # amp
        bars = ax.bar(x , heights[1], color=colors, width=width, alpha=0.4,
                      edgecolor=colors)
        autolabel(ax, bars) # call
        # filling in
        bars = ax.bar(x + width, heights[2], color=colors, width=width, alpha=0.4,
                      edgecolor=colors)
        autolabel(ax, bars) # call

        labels = list(pop_dico.keys())[:4]
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
    txt = "{} ({} cells)  [time|{}|fillIn]".format(kind, len(data), amp)
    fig.text(0.5, 0.99, txt, ha='center', va='top', fontsize=14)
    if anot:
        date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        fig.text(1, 0.01, 'cellContribution.py:plot_composite_sectFull_2X1_fill',
                 ha='right', va='bottom', alpha=0.4)
        fig.text(0.01, 0.01, date, ha='left', va='bottom', alpha=0.4)
    fig.tight_layout()
    return fig

plt.close('all')
save = False
amp = ['gain', 'engy'][1]
stat_df_sig, sig_cells = ldat.build_sigpop_statdf(amp=amp, with_fill=True)
for mes in ['vm', 'spk']:
    #load_cell_contributions(mes='vm', amp='gain', age='new'):
    data = ldat.load_cell_contributions(mes, age='new', amp=amp)
    fig = plot_composite_sectFull_2X1_fill(data, sig_cells, kind=mes,
                                               amp=amp)
    if save:
        filename = os.path.join(paths['save'], mes + amp.title() \
                                + 'Fill_composite_cell_contribution_2X1.png')
        fig.savefig(filename)


#%%
plt.close('all')
save = False
amp = ['gain', 'engy'][1]
stat_df_sig, sig_cells = ldat.build_sigpop_statdf(amp=amp, with_fill=True)
