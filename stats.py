#!/usr/bin/env python3
# -*- coding: utf-8 -*-

""" statistical extraction of cell properties """

import os
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config
from load import load_data as ldat

paths = config.build_paths()
paths['save'] = os.path.join(paths['owncFig'], 'pythonPreview', 'stat')
std_colors = config.std_colors()
anot = True

plt.rcParams['axes.xmargin'] = 0.05
plt.rcParams['axes.ymargin'] = 0.05

#%% ALL

def plot_stat(statdf, kind='mean', legend=False, digit=False):
    """
    plot the stats
    input :
        statdf,
        kind in ['mean', 'med'],
        legend : bool (display legend)
        digit : bool (use nb of cell as a marker)
    output : matplotlib figure
    """
    stats = dict(mean = ['_mean', '_sem'],
                med = ['_med', '_mad'])
    stat = stats.get(kind, '')
    if not stat:
        print('non valid kind argument')
        return

    colors= [std_colors[item] for item in
             ['red', 'green', 'yellow', 'blue', 'dark_blue']]
    ref = 'population' if statdf.max().max() == 37 else 'sig_pops'

    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(8, 8),
                             sharex=True, sharey=True)
    axes = axes.flatten()
    title = "{}    ({} ± {}) ".format(ref, stat[0][1:], stat[1][1:])
    fig.suptitle(title)
    # plots
    conds = conds = [(a, b) for a in ['vm', 'spk'] for b in ['sec', 'full']]
    for i, cond in enumerate(conds):
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
        # extract df indexes (for x and y)
        time_rows = [st for st in rows if 'time50' in st]
        y_rows = [st for st in rows if 'engy' in st]
        cols = [col for col in statdf.columns if col.startswith(rec)]
        cols = [st for st in cols if stat[0] in st or stat[1] in st]
        # build labels
        labels = [st.split('_')[0] for st in y_rows]
        # values (for x an y)
        x = statdf.loc[time_rows, cols][rec + stat[0]].values
        xerr = statdf.loc[time_rows, cols][rec + stat[1]].values
        y = statdf.loc[y_rows, cols][rec + stat[0]].values
        yerr = statdf.loc[y_rows, cols][rec + stat[1]].values
        #plot
#        digit = True
        if not digit:
            for xi, yi, xe, ye, ci, lbi  in zip(x, y, xerr, yerr, colors, labels):
                ax.errorbar(xi, yi, xerr=xe, yerr=ye,
                        fmt='s', color=ci, label=lbi)
        else:
            # extract nb of cells
            key = '_'.join([rec, 'count'])
            cell_nb = [int(statdf.loc[item, [key]][0]) for item in y_rows]
            for xi, yi, xe, ye, ci, lbi, nbi  \
                in zip(x, y, xerr, yerr, colors, labels, cell_nb):
                    ax.errorbar(xi, yi, xerr=xe, yerr=ye,
                                fmt='s', color=ci, label=lbi,
                                marker='s', ms=16, mec='w', mfc='w')
                        # marker='$'+ str(nbi) + '$', ms=24, mec='w', mfc=ci)
                    ax.text(xi, yi, str(nbi), color=ci, fontsize=14,
                            horizontalalignment='center',
                            verticalalignment='center')
        if legend:
            ax.legend()
    #adjust
    for i, ax in enumerate(axes):
        ax.axvline(0, linestyle='-', alpha=0.4, color='k')
        ax.axhline(0, linestyle='-', alpha=0.4, color='k')
        for spine in ['top', 'right']:
            ax.spines[spine].set_visible(False)
            if i % 2 == 0:
                ax.set_ylabel(r'$\Delta$ energy')
            else:
                ax.spines['left'].set_visible(False)
                ax.yaxis.set_visible(False)
            if i > 1:
                ax.set_xlabel('latency advance (ms)')
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


plt.close('all')

stat_df = ldat.build_pop_statdf()
stat_df_sig, sig_cells = ldat.build_sigpop_statdf()
fig1 = plot_stat(stat_df, kind='mean', digit=False)
# fig2 = plot_stat(stat_df, kind='med')
fig3 = plot_stat(stat_df_sig, kind='mean', digit=False)
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
                        kind='mean', amp='engy', mes='vm', legend=False,
                        shared=True, digit=False):
    """
    plot the stats
    input : statdf, kind in ['mean', 'med'], loc in ['50', 'peak', 'energy']
    output : matplotlib figure
    here combined top = full pop, bottom : significative pop
    """
    if kind == 'mean':
        stat = ['_mean', '_sem']
    elif kind == 'med':
        stat = ['_med', '_mad']
    else:
        print('non valid kind argument')
        return

    ylabels = dict(engy=r'$\Delta\ energy$',
                   gain=r'$\Delta\ gain$')

    colors = [std_colors['red'], std_colors['green'],
              std_colors['yellow'], std_colors['blue'],
              std_colors['dark_blue']]

    if shared:
        # share scales high and low
        fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(8,8),
                             sharex=True, sharey=True)
        axes = axes.flatten()
    else:
        fig = plt.figure(figsize=(8,8))
        ax0 = fig.add_subplot(221)
        ax1 = fig.add_subplot(222, sharex=ax0, sharey=ax0)
        ax2 = fig.add_subplot(223)
        ax3 = fig.add_subplot(224, sharex=ax2, sharey=ax2)
        axes = [ax0, ax1, ax2, ax3]
    title = stat[0][1:] +'   (' +  stat[1][1:] + ')'
    fig.suptitle(title)
    conds = [(x,y) for x in [mes, mes] for y in ['sect', 'full']]
    for i, cond in enumerate(conds):
    # for i, cond in enumerate([(mes, 'sect'), (mes, 'full'),
    #                            (mes, 'sect'), (mes, 'full')]):
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
        if not digit:
            # marker in the middle
            for xi, yi, xe, ye, ci, lbi  in zip(x, y, xerr, yerr, colors, labels):
                ax.errorbar(xi, yi, xerr=xe, yerr=ye,
                        fmt='s', color=ci, label=lbi)
        else:
            # nb of cells in the middle
            # extract nb of cells
            key = '_'.join([rec, 'count'])
            cell_nb = [int(df.loc[item, [key]][0]) for item in y_rows]
            for xi, yi, xe, ye, ci, lbi, nbi  \
                in zip(x, y, xerr, yerr, colors, labels, cell_nb):
                    ax.errorbar(xi, yi, xerr=xe, yerr=ye,
                                fmt='s', color=ci, label=lbi,
                                marker='s', ms=16, mec='w', mfc='w')
                        # marker='$'+ str(nbi) + '$', ms=24, mec='w', mfc=ci)
                    ax.text(xi, yi, str(nbi), color=ci, fontsize=14,
                            horizontalalignment='center',
                            verticalalignment='center')
        if legend:
            ax.legend()
    #adjust
    for i, ax in enumerate(axes):
        ax.axvline(0, linestyle='-', alpha=0.4, color='k')
        ax.axhline(0, linestyle='-', alpha=0.4, color='k')
        for spine in ['top', 'right']:
            ax.spines[spine].set_visible(False)
            if i % 2 == 0:
                ax.set_ylabel(ylabels.get(amp, 'not defined'))
#                ax.set_ylabel('energy')
            else:
                ax.spines['left'].set_visible(False)
                ax.yaxis.set_visible(False)
            if i > 1:
                ax.set_xlabel('latency advance (ms)')
            else:
                pass
                # ax.spines['bottom'].set_visible(False)
                # ax.xaxis.set_visible(False)
    fig.tight_layout()
    # fig.subplots_adjust(hspace=0.02)
    # fig.subplots_adjust(wspace=0.02)
    if anot:
        date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        fig.text(0.99, 0.01, 'stat.py:plot_stat',
                 ha='right', va='bottom', alpha=0.4)
        fig.text(0.01, 0.01, date, ha='left', va='bottom', alpha=0.4)
    return fig

plt.close('all')
shared=False
mes = ['vm', 'spk'][0]
amp = ['gain', 'engy'][1]
kind = ['mean', 'med'][0]
stat_df = ldat.build_pop_statdf(amp=amp)                        # append gain to load
stat_df_sig, sig_cells = ldat.build_sigpop_statdf(amp=amp)   # append gain to load
fig1 = plot_composite_stat(stat_df, stat_df_sig, sig_cells,
                           kind=kind, amp=amp, mes=mes,
                           shared=shared, digit=False)

save = False
if save:
    if shared:
        filename = os.path.join(paths['save'], mes + amp.title()
                                + '_composite_meanSem.png')
    else:
        filename = os.path.join(paths['save'], 'nshared_'
                                + mes + amp.title() + '_composite_meanSem.png')
    fig1.savefig(filename)

# for shared in [True, False]:
#     for mes in ['vm', 'spk']:
#         for amp in ['gain', 'engy']:
#             kind = ['mean', 'med'][0]


#             stat_df = ldat.build_pop_statdf(amp=amp)                        # append gain to load
#             stat_df_sig, sig_cells = ldat.build_sigpop_statdf(amp=amp)   # append gain to load
#             fig1 = plot_composite_stat(stat_df, stat_df_sig, sig_cells,
#                            kind=kind, amp=amp, mes=mes,
#                            shared=shared, digit=False)

#%%
def plot_composite_stat_1x2(statdf, statdfsig, sigcells, spread='sect',
                        kind='mean', amp='engy', mes='vm', legend=False,
                        shared=True, digit=False):
    """
    plot the stats
    input : statdf, kind in ['mean', 'med'], loc in ['50', 'peak', 'energy']
    output : matplotlib figure
    here combined top = full pop, bottom : significative pop
    """
    if kind == 'mean':
        stat = ['_mean', '_sem']
    elif kind == 'med':
        stat = ['_med', '_mad']
    else:
        print('non valid kind argument')
        return

    ylabels = dict(engy=r'$\Delta\ energy$',
                   gain=r'$\Delta\ gain$')

    titles = dict(time = r'Latency Advance (% significant cells)',
                  engy = r'$\Delta$ Energy (% significant cells)',
                  sect =  'Sector',
                  full = 'Full',
                  vm = 'Vm',
                  spk = 'Spikes')
    colors = [std_colors[color] for color in
              ['red', 'green', 'yellow', 'blue', 'dark_blue']]

    data = [statdf, statdfsig]
    pops = ['population', 'sig_pops']
    if shared:
        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(8,4),
                             sharex=True, sharey=True)
    else:
        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(8,4),
                             sharex=False, sharey=False)
    axes = axes.flatten()
    title = "{} {}   ({}±{})".format(titles[mes], titles[spread],
                                     str.title(stat[0][1:]),
                                     str.title(stat[1][1:]))
    fig.suptitle(title)

    for i, ax in enumerate(axes):
        df = data[i]
        pop = pops[i]
        rec = mes
        ax_title = '{}'.format(pop)
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
        # for xi, yi, xe, ye, ci, lbi  in zip(x, y, xerr, yerr, colors, labels):
        #     ax.errorbar(xi, yi, xerr=xe, yerr=ye,
        #                 fmt='s', color=ci, label=lbi)
        if not digit:
            # marker in the middle
            for xi, yi, xe, ye, ci, lbi  in zip(x, y, xerr, yerr, colors, labels):
                ax.errorbar(xi, yi, xerr=xe, yerr=ye,
                        fmt='s', color=ci, label=lbi)
        else:
            # nb of cells in the middle
            # extract nb of cells
            key = '_'.join([rec, 'count'])
            cell_nb = [int(df.loc[item, [key]][0]) for item in y_rows]
            for xi, yi, xe, ye, ci, lbi, nbi  \
                in zip(x, y, xerr, yerr, colors, labels, cell_nb):
                    ax.errorbar(xi, yi, xerr=xe, yerr=ye,
                                fmt='s', color=ci, label=lbi,
                                marker='s', ms=16, mec='w', mfc='w')
                        # marker='$'+ str(nbi) + '$', ms=24, mec='w', mfc=ci)
                    ax.text(xi, yi, str(nbi), color=ci, fontsize=14,
                            horizontalalignment='center',
                            verticalalignment='center')
        if legend:
            ax.legend()
    #adjust
    for i, ax in enumerate(axes):
        ax.axvline(0, linestyle='-', alpha=0.4, color='k')
        ax.axhline(0, linestyle='-', alpha=0.4, color='k')
        for spine in ['top', 'right']:
            ax.spines[spine].set_visible(False)
            ax.set_xlabel('latency advance (ms)')
            if i > 0:
                ax.spines['left'].set_visible(False)

            else:
                ax.set_ylabel(ylabels.get(amp, 'not defined'))
                # ax.spines['bottom'].set_visible(False)
                # ax.xaxis.set_visible(False)
    fig.tight_layout()
    # fig.subplots_adjust(hspace=0.02)
    # fig.subplots_adjust(wspace=0.02)
    if anot:
        date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        fig.text(0.99, 0.01, 'stat.py:plot_composite_stat_1x2',
                 ha='right', va='bottom', alpha=0.4)
        fig.text(0.01, 0.01, date, ha='left', va='bottom', alpha=0.4)
    return fig

plt.close('all')
save = False
kind = ['mean', 'med'][0]
amp = ['gain', 'engy'][1]
stat_df = ldat.build_pop_statdf(amp=amp)                     # append gain to load
stat_df_sig, sig_cells = ldat.build_sigpop_statdf(amp=amp)   # append gain to load
fig1 = plot_composite_stat_1x2(stat_df, stat_df_sig, sig_cells,
            kind=kind, amp=amp, mes='vm',
            shared=True, spread='sect',
            digit=True)
save = False
if save:
    dirname = os.path.join(paths['owncFig'],
                           'pythonPreview', 'current', 'fig')
    file = 'composite_stat_1x2_' + 'vm' + 'sect'.title() + '.png'
    fig1.savefig(os.path.join(dirname, file))
#%%
for shared in [True, False]:
    for mes in ['vm', 'spk']:
        for spread in ['sect', 'full']:
            fig1 = plot_composite_stat_1x2(stat_df, stat_df_sig, sig_cells,
                                           kind=kind, amp=amp, mes=mes,
                                           shared=shared, spread=spread,
                                           digit=False)
            if save:
                if shared:
                    filename = os.path.join(paths['save'], 'composite1x2',
                                            'composite_stat_1x2_'
                                            + mes + spread.title() + '.png')
                else:
                    filename = os.path.join(paths['save'], 'composite1x2',
                                            'ns_composite_stat_1x2_'
                                            + mes + spread.title() + '.png')
                fig1.savefig(filename)
