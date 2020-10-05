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
paths['save'] = os.path.join(paths['owncFig'], 'pythonPreview', 'cross')
std_colors = config.std_colors()
anot = True

plt.rcParams['axes.xmargin'] = 0.05
plt.rcParams['axes.ymargin'] = 0.05

#%% manipulate dataframes
def select_in_df(df, spread='sect', param='engy', noSig=True):
    """
    return the selected df parts for plotting
    spread in ['sect', 'full'],
    param in ['time', 'gain', 'engy']
    """
    if spread not in ['sect', 'full']:
        print("'spread' should be in ['sect', 'full']")
        return
    if param not in ['time', 'gain', 'engy']:
        print("'param' should be in ['time', 'gain', 'engy']")
        return
    # select by param (first value = control)
    col_list = [item for item in df.columns if param in item]
    # select by spread
    col_list = [item for item in col_list if spread in item]
    if noSig:
        # remove sig
        col_list = [item for item in col_list if 'sig' not in item]
    return df[col_list].copy()


def check_for_same_index(dfleft, dfright):
    """
    check the index, remove difference, return filtered dataframes

    """
    # test for same cells
    diff = (dfleft.index ^ dfright.index).tolist()
    if len(diff) > 0:
        print("left and right doesn't contain the same cells !")
        print(diff)
    for item in diff:
        if item in dfleft.index:
            dfleft = dfleft.drop(item)
        elif item in right.index:
            dfright = dfright.drop(item)
    return dfleft, dfright

#%% plot functions

def plot_sorted_responses(df_left, df_right, mes='', overlap=True,
                          left_sig=True, right_sig=True):
    """
    plot the sorted cell responses
    input = dataframes, overlap=boolean

    """
    def set_ticks_both(axis):
        """ set ticks and ticks labels on both sides """
        ticks = list(axis.majorTicks) # a copy
        ticks.extend(axis.minorTicks)
        for t in ticks:
            t.tick1line.set_visible(True)
            t.tick2line.set_visible(True)
            t.label1.set_visible(True)
            t.label2.set_visible(True)

    colors = [std_colors['red'], std_colors['green'],
              std_colors['yellow'], std_colors['blue']]
    dark_colors = [std_colors['dark_red'], std_colors['dark_green'],
                   std_colors['dark_yellow'], std_colors['dark_blue']]

    # text labels
    if 'sect' in right.columns[0].split('_')[0]:
        spread = 'sect'
    else:
        spread = 'full'
    title = 'sorted_responses' + ' (' + mes + ' ' + spread + ')'
    anotx = 'Cell rank'
    anot_left = df_left.columns[0].split('_')[1]
    anot_right = df_right.columns[0].split('_')[1]
    # anoty = ['Relative peak advance(ms)', 'Relative peak amplitude']
    #           #(fraction of Center-only response)']
    # plot
    fig = plt.figure()
    gs = fig.add_gridspec(4, 2)
    # left
    left_axes = []
    ax = fig.add_subplot(gs[0, 0])
    left_axes.append(ax)
    for i in range(1, 4):
        left_axes.append(fig.add_subplot(gs[i, 0],
                                         sharey=ax, sharex=ax))
    # right
    right_axes = []
    ax = fig.add_subplot(gs[0, 1])
    right_axes.append(ax)
    for i in range(1, 4):
        right_axes.append(fig.add_subplot(gs[i, 1],
                                          sharey=ax, sharex=ax))
    # to identify the plots (uncomment to use)
    if anot:
        fig.suptitle(title)
    x = range(1, len(df_left) + 1)
    # plot the traces
    if left_sig:
        traces = [item for item in left.columns if '_sig' not in item]
        for i, name in enumerate(traces):
            #color : white if non significant, edgecolor otherwise
            edge_color = colors[i]
            color_dic = {0 : 'w', 1 : edge_color}
            sig_name = name + '_sig'
            select = df_left[[name, sig_name]].sort_values(by=[name, sig_name],
                                                           ascending=False)
            bar_colors = [color_dic[x] for x in select[sig_name]]
            ax = left_axes[i]
            # ax.set_title(str(i))
            ax.set_title(name, alpha=0.5)
            # without significance
            #select = df_left[name].sort_values(ascending=False)
            #ax.bar(x, select, color=colors[i], edgecolor=dark_colors[i],
            #       alpha=0.8, width=0.8)
            # # with significance
            select = df_left[name].sort_values(ascending=False)

            ax.bar(x, select, color=bar_colors, edgecolor=edge_color,
                   alpha=0.8, width=0.8)
            if i == 0:
                ax.set_title(anot_left)
    else:
        for i, name in enumerate(df_left.columns):
            ax = left_axes[i]
            # ax.set_title(str(i))
            ax.set_title(name, alpha=0.5)
            # without significance
            select = df_left[name].sort_values(ascending=False)
            ax.bar(x, select, color=colors[i], edgecolor=dark_colors[i],
                   alpha=0.5, width=0.8)
            if i == 0:
                ax.set_title(anot_left)
    #right
    if right_sig:
        traces = [item for item in right.columns if '_sig' not in item]
        for i, name in enumerate(traces):
            #color : white if non significant, edgecolor otherwise
            edge_color = colors[i]
            color_dic = {0 : 'w', 1 : edge_color}
            sig_name = name + '_sig'
            select = df_right[[name, sig_name]].sort_values(by=[name, sig_name],
                                                            ascending=False)
            bar_colors = [color_dic[x] for x in select[sig_name]]
            ax = right_axes[i]
            # ax.set_title(str(i))
            ax.set_title(name, alpha=0.5)
            select = df_right[name].sort_values(ascending=False)
            ax.bar(x, select, color=bar_colors, edgecolor=edge_color,
                   alpha=0.8, width=0.8)
            if i == 0:
                ax.set_title(anot_right)
    else:
        for i, name in enumerate(df_right.columns):
            ax = right_axes[i]
            # ax.set_title(str(i))
            ax.set_title(name , alpha=0.5)
            # without significance
            select = df_right[name].sort_values(ascending=False)
            ax.bar(x, select, color=colors[i], edgecolor=dark_colors[i],
                   alpha=0.5, width=0.8)
            if i == 0:
                ax.set_title(anot_right)

    # alternate the y_axis position
    axes = fig.get_axes()
    left_axes = axes[:4]
    right_axes = axes[4:]
    for axe in [left_axes, right_axes]:
        for i, ax in enumerate(axe):
            ax.set_facecolor('None')
            # ax.set_title(i)
            for spine in ['top', 'bottom']:
                ax.spines[spine].set_visible(False)
           # ax.ticklabel_format(useOffset=True)
            # zero line
            ax.set_xlim(0, len(df_left)+1)
            lims = ax.get_xlim()
            ax.hlines(0, lims[0], lims[1], alpha=0.2)
            # ticks and ticks labels on both sides (call)
            # set_ticks_both(ax.yaxis)
            # alternate right and left
            # if overlap:
            #     #label left:
            #     if i % 2 == 0:
            #         ax.spines['right'].set_visible(False)
            #     #label right
            #     else:
            #         ax.spines['left'].set_visible(False)
            #         ax.yaxis.tick_right()
            # else:
            #     ax.spines['right'].set_visible(False)
            if i != 3:
                ax.xaxis.set_visible(False)
            else:
                ax.set_xlabel(anotx)
                ax.xaxis.set_label_coords(0.5, -0.025)
                ax.set_xticks([1, len(df_right)])
    # for ax in left_axes:
    #     custom_ticks = np.linspace(0, 0.5, 2)
    #     ax.set_yticks(custom_ticks)
    # for ax in right_axes:
    #     custom_ticks = np.linspace(0, 0.5, 2)
    #     ax.set_yticks(custom_ticks)
    no_spines = True
    if no_spines == True:
        for ax in left_axes:
            limx = ax.get_xlim()
            ax.vlines(limx[0], 0, 0.5, color='k', linewidth=2)
            # ax.vlines(limx[1], 0, -10, color='k', linewidth=2)
            for spine in ['left', 'right']:
                ax.spines[spine].set_visible(False)
        for ax in right_axes:
            limx = ax.get_xlim()
            ax.vlines(limx[0], 0, 0.5, color='k', linewidth=2)
            # ax.vlines(limx[1], 0, -0.5, color='k', linewidth=2)
            for spine in ['left', 'right']:
                ax.spines[spine].set_visible(False)
                ax.set_xlim(0, len(df_right)+1)

    # for ax in left_axes:
    #     custom_ticks = np.linspace(-10, 10, 3, dtype=int)
    #     ax.set_yticks(custom_ticks)
    # for ax in right_axes:
    #     ax.set_ylim(-0.5, 0.5)
    #     custom_ticks = np.linspace(-0.5, 0.5, 3)
    #     ax.set_yticks(custom_ticks)

    # align each row yaxis on zero between subplots
    # align_yaxis(left_axes[0], 0, right_axes[0], 0)
    # keep data range whithout distortion, preserve 0 alignment
    # change_plot_trace_amplitude(axes[1], 0.80)
    # remove the space between plots

    fig.tight_layout()
    if overlap:
        fig.subplots_adjust(hspace=-0.5, wspace=0.2)
    else:
        fig.subplots_adjust(hspace=0.05, wspace=0.2)
    if anot:
        date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        fig.text(0.99, 0.01, 'cross.py:plot_sorted_responses',
                 ha='right', va='bottom', alpha=0.4)
        fig.text(0.01, 0.01, date, ha='left', va='bottom', alpha=0.4)
    return fig


def horizontal_dot_plot(df_left, df_right, mes=''):

    colors = [std_colors['red'], std_colors['green'],
              std_colors['yellow'], std_colors['blue']]
    dark_colors = [std_colors['dark_red'], std_colors['dark_green'],
                   std_colors['dark_yellow'], std_colors['dark_blue']]
     # text labels
    if 'sect' in df_right.columns[0].split('_')[0]:
        spread = 'sect'
    else:
        spread = 'full'
    title = 'sorted_responses' + ' (' + mes + ' ' + spread + ')'
    anoty = 'Cell rank'
    # anotx = [df_left.columns[0][5:], df_right.columns[0][5:]]
    anotx = [left.columns[0].split('_')[1], right.columns[0].split('_')[1]]
    fig = plt.figure()
    fig.suptitle(title)
    # left
    ax = fig.add_subplot(121)
    df = df_left.sort_values(by=df_left.columns[0], ascending=True)
    val_cols = [item for item in df.columns if '_sig' not in item]
    sig_cols = [item for item in df.columns if '_sig' in item]
#    df = df[cols]
    #sort
    sorted_cells = df.index.copy()
    for i, col in enumerate(val_cols):
        #define colors
        # marker_edgecolor = colors[i]
        # marker_facecolor =  if stat colors[i] else 'w'
        z1 = []
        for j in np.arange(len(df)):
            z1.append(colors[i])
        #stat
        z2 = []
        if col + '_sig' in df.columns:
            alpha = 0.8
            for a, b in zip(z1, df[col + '_sig']):
                if b == 1:
                    z2.append(a)
                else:
                    z2.append('w')
        else:
            alpha = 0.5
            z2 = z1
        #plot
        yl = df.index.to_list()
        xl = df[col].tolist()
        for x, y, e, f in zip(xl, yl, z1, z2):
            ax.plot(x, y, 'o', markeredgecolor=e, markerfacecolor=f,
                    alpha=alpha, markeredgewidth=1.5, markersize=6)
  # right
    ax = fig.add_subplot(122)
    df = df_right.reindex(sorted_cells)
    val_cols = [item for item in df.columns if '_sig' not in item]
    sig_cols = [item for item in df.columns if '_sig' in item]
    for i, col in enumerate(val_cols):
        #define colors
        # marker_edgecolor = colors[i]
        # marker_facecolor =  if stat colors[i] else 'w'
        z1 = []
        for j in np.arange(len(df)):
            z1.append(colors[i])
        #stat
        z2 = []
        if col + '_sig' in df.columns:
            alpha = 0.8
            for a, b in zip(z1, df[col + '_sig']):
                if b == 1:
                    z2.append(a)
                else:
                    z2.append('w')
        else:
            alpha = 0.5
            z2 = z1
        #plot
        yl = df.index.to_list()
        xl = df[col].tolist()
        for x, y, e, f in zip(xl, yl, z1, z2):
            ax.plot(x, y, 'o', markeredgecolor=e, markerfacecolor=f,
                    alpha=alpha, markeredgewidth=1.5, markersize=6)
    for i, ax in enumerate(fig.get_axes()):
        ax.set_xlabel(anotx[i])
        for spine in ['top', 'right']:
            ax.spines[spine].set_visible(False)
        lims = ax.get_ylim()
        ax.vlines(0, lims[0], lims[1], 'k', alpha=0.3)
        ax.set_yticks([0, len(df) - 1])
        ax.set_yticklabels([1, len(df)])
    if anot:
        date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        fig.text(0.99, 0.01, 'cross.py:horizontal_dot_plot',
                 ha='right', va='bottom', alpha=0.4)
        fig.text(0.01, 0.01, date, ha='left', va='bottom', alpha=0.4)
    return fig


def scatter_lat_gain(df_left, df_right, mes=''):
    """
    build a scatter plot
        input = left, right : pandas dataframe
        mes in
    """
    colors = [std_colors['red'], std_colors['green'],
              std_colors['yellow'], std_colors['blue']]
    dark_colors = [std_colors['dark_red'], std_colors['dark_green'],
                   std_colors['dark_yellow'], std_colors['dark_blue']]
     # text labels
    if 'sect' in df_right.columns[0].split('_')[0]:
        spread = 'sect'
    else:
        spread = 'full'
    title = 'responses' + ' (' + mes + ' ' + spread + ')'
    fig = plt.figure(figsize=(8, 8))
    fig.suptitle(title)
    ax = fig.add_subplot(111)
    # remove sig columns
    cols = [item for item in df_left.columns if '_sig' not in item]
    df_left = df_left[cols]
    cols = [item for item in df_right.columns if '_sig' not in item]
    df_right = df_right[cols]
    #labels
    labels = [item.split('_')[0][:-4] for item in cols]
    for i in range(len(df_left.columns)):
        color_list = []
        for j in range(len(df_left)):
            color_list.append(colors[i])
        x = df_left[df_left.columns[i]]
        y = df_right[df_right.columns[i]]
        ax.scatter(x, y,
                   c=color_list, s=100,
                   edgecolors=dark_colors[i], alpha=0.6,
                   label = labels[i])
    ax.set_xlabel('time')
    ax.set_ylabel('gain')
    ax.set_xlabel(df_left.columns[0].split('_')[1])
    ax.set_ylabel(df_right.columns[0].split('_')[1])
    lims = ax.get_xlim()
    ax.hlines(0, lims[0], lims[1], 'k', alpha=0.3)
    lims = ax.get_ylim()
    ax.vlines(0, lims[0], lims[1], 'k', alpha=0.3)
    ax.legend()
    if anot:
        date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        fig.text(0.99, 0.01, 'cross.py:scatter_lat_gain',
                 ha='right', va='bottom', alpha=0.4)
        fig.text(0.01, 0.01, date, ha='left', va='bottom', alpha=0.4)
    return fig


#%%
def histo_lat_gain(df_left, df_right, mes=''):
    """
    histogramme des données
    """
    colors = [std_colors['red'], std_colors['green'],
              std_colors['yellow'], std_colors['blue']]
    dark_colors = [std_colors['dark_red'], std_colors['dark_green'],
                   std_colors['dark_yellow'], std_colors['dark_blue']]

    # text labels
    if 'sect' in right.columns[0].split('_')[0]:
        spread = 'sect'
    else:
        spread = 'full'
    title = 'responses' + ' (' + mes + ' ' + spread + ')'
    anotx = 'Cell rank'
    anoty = [df_left.columns[0].split('_')[1],
             df_right.columns[0].split('_')[1]]
    # anoty = ['Relative peak advance(ms)', 'Relative peak amplitude']
    #          #(fraction of Center-only response)']
    # plot
    fig = plt.figure(figsize=(8, 8))
    gs = fig.add_gridspec(4, 2)
    # remove sig columns
    cols = [item for item in df_left.columns if '_sig' not in item]
    df_left = df_left[cols]
    cols = [item for item in df_right.columns if '_sig' not in item]
    df_right = df_right[cols]
    # left
    left_axes = []
    ax = fig.add_subplot(gs[0, 0])
    left_axes.append(ax)
    for i in range(1, 4):
        left_axes.append(fig.add_subplot(gs[i, 0], sharex=ax, sharey=ax))
    # right
    right_axes = []
    ax = fig.add_subplot(gs[0, 1], sharey=ax)
    right_axes.append(ax)
    for i in range(1, 4):
        right_axes.append(fig.add_subplot(gs[i, 1], sharex=ax, sharey=ax))
    # to identify the plots (uncomment to use)
    if anot:
        fig.suptitle(title)
    # plot the traces
    # left
    for i, name in enumerate(df_left.columns):
        ax = left_axes[i]
        # ax.set_title(str(i))
        # ax.set_title(name)
        ax.hist(df_left[name], bins=15, color=colors[i],
                edgecolor=dark_colors[i], alpha=0.8)
        # ax.hist(df_left[name], width=2, color=colors[i],
        #         edgecolor=dark_colors[i], alpha=0.8)
        if i == 0:
            ax.set_title(anoty[i])
    for i, name in enumerate(df_left.columns):
        ax = left_axes[i]
        lims = ax.get_ylim()
        ax.vlines(0, lims[0], lims[1], alpha=1)
        med = df_left[name].median()
        ax.vlines(med, lims[0], lims[1], color=dark_colors[i],
                  linewidth=2, linestyle=':')
        tx = 'med=' + str(round(med, 2))
        ax.text(0.65, 0.8, tx, horizontalalignment='left',
                transform=ax.transAxes, alpha=0.5)
        # right
    for i, name in enumerate(df_right.columns):
        ax = right_axes[i]
        # ax.set_title(str(i))
        # ax.set_title(name)
        ax.hist(df_right[name], bins=15, color=colors[i],
                edgecolor=dark_colors[i], alpha=0.8)
        # ax.hist(df_right[name], width = 0.05, color=colors[i],
        #         edgecolor=dark_colors[i], alpha=0.8)
        if i == 0:
            ax.set_title(anoty[1])
    for i, name in enumerate(df_right.columns):
        ax = right_axes[i]
        lims = ax.get_ylim()
        ax.vlines(0, lims[0], lims[1], alpha=1)
        med = df_right[name].median()
        ax.vlines(med, lims[0], lims[1], color=dark_colors[i],
                  linewidth=2, linestyle=':')
        tx = 'med=' + str(round(med, 2))
        ax.text(0.65, 0.8, tx, horizontalalignment='left',
                transform=ax.transAxes, alpha=0.5)

    for ax in fig.get_axes():
        ax.set_facecolor('None')
        ax.yaxis.set_visible(False)
        for spine in ['left', 'top', 'right']:
            ax.spines[spine].set_visible(False)
    fig.tight_layout()
    if anot:
        date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        fig.text(0.99, 0.01, 'cross.py:histo_lat_gain',
                 ha='right', va='bottom', alpha=0.4)
        fig.text(0.01, 0.01, date, ha='left', va='bottom', alpha=0.4)
    return fig


#%%
plt.close('all')

for spread in ['sect', 'full']:
    for mes in ['vm', 'spk']:
      #  mes = 'vm'
        data = ldat.load_cell_contributions(rec=mes, amp='engy', age='new')
        left = select_in_df(data, spread=spread, param='time', noSig=False)
        right = select_in_df(data, spread=spread, param='engy', noSig=False)

        left, right = check_for_same_index(left, right)

        fig1 = plot_sorted_responses(left, right, mes=mes, overlap=True,
                                    left_sig=True, right_sig=True)
        fig2 = horizontal_dot_plot(left, right, mes=mes)
        fig3 = scatter_lat_gain(left, right, mes=mes)

        fig4 = histo_lat_gain(left, right, mes=mes)

        save=False
        if save:
            names = []
            for kind in ['bar', 'dot', 'scatter', 'histo']:
                name = '_'.join([kind, mes, spread])
                names.append(name)
            figs = [fig1, fig2, fig3, fig4]
            for name, fig in zip(names, figs):
                filename = os.path.join(paths['save'], name + '.png')
                fig.savefig(filename)
