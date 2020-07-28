#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
plot centrigabor figures from data stored in .xlsx files
"""

import os
from importlib import reload
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# import matplotlib.gridspec as gridspec
from matplotlib.patches import Rectangle
import matplotlib.patches as patches
from matplotlib import markers
from datetime import datetime

import config
import plot_general_functions as gfuc
import load_data as ldat
import old_figs as ofig
import fig_proposal as figp

# nb description with pandas:
pd.options.display.max_columns = 30

#===========================
# global setup
# NB fig size : 8.5, 11.6 or 17.6 cm

anot = True           # to draw the date and name on the bottom of the plot
std_colors = config.std_colors()
speedColors = config.speed_colors()
plt.rcParams.update(config.rc_params())
paths = config.build_paths()
os.chdir(paths['pg'])

# energy_df = ldat.load_energy_gain_index(paths)
# latGain50_v_df = ldat.load_cell_contributions('vm')
# latGain50_s_df = ldat.load_cell_contributions('spk')


#%%
plt.close('all')

def plot_figure2(data, colsdict, fill=True, anot=False):
    """
    figure2 (individual + pop + sig)
    """
    colors = ['k', std_colors['red']]
    alphas = [0.8, 0.8]
    inv_colors = colors[::-1]
    inv_alphas = alphas[::-1]


    fig = plt.figure(figsize=(17.6, 12))
    axes = []
    for i in range(1, 7):
        axes.append(fig.add_subplot(2, 3, i))

    # axes list
    vmaxes = axes[:3]      # vm axes = top row
    spkaxes = axes[3:]     # spikes axes = bottom row
    # ____ plots individuals (first column)
    # individual vm
    cols = colsdict['indVm']
    ax = vmaxes[0]
    for i, col in enumerate(cols):
        ax.plot(data[col], color=colors[i], alpha=alphas[i],
                label=col)
    # start point
    x = 41.5
    y = data.indiVmctr.loc[x]
    ax.plot(x, y, 'o', color=std_colors['blue'])
    lims = ax.get_ylim()
    ax.vlines(x, lims[0], lims[1], linewidth=1, color=std_colors['blue'],
              linestyle=':')
    # individual spike
    cols = colsdict['indSpk']
    ax = spkaxes[0]
    for i, col in enumerate(cols[::-1]):
        ax.plot(data[col], color=colors[::-1][i],
                alpha=1, label=col, linewidth=1)
        ax.fill_between(data.index, data[col],
                        color=colors[::-1][i], alpha=0.5, label=col)
    # start point
    x = 39.8
    y = data.indiSpkCtr.loc[x]
    ax.plot(x, y, 'o', color=std_colors['blue'])
    lims = ax.get_ylim()
    ax.vlines(x, lims[0], lims[1], linewidth=1, color=std_colors['blue'],
              linestyle=':')
    # individual spike
    cols = colsdict['indSpk']
    ax = spkaxes[0]
    # ____ plots pop (column 1-3)
    df = data.loc[-30:35]       # limit xscale
    # pop vm
    cols = colsdict['popVm']
    ax = vmaxes[1]
    for i, col in enumerate(cols):
        ax.plot(df[col], color=colors[i], alpha=alphas[i],
                label=col)
    ax.annotate("n=37", xy=(0.2, 0.8),
                xycoords="axes fraction", ha='center')
    # popVmSig
    cols = colsdict['popVmSig']
    ax = vmaxes[2]
    # traces
    for i, col in enumerate(cols[:2]):
        ax.plot(df[col], color=colors[i], alpha=alphas[i],
                label=col)
        # errors : iterate on tuples
        for i, col in enumerate(cols[2:]):
            if fill:
                ax.fill_between(df.index, df[col[0]], df[col[1]],
                                color=colors[i], alpha=0.2)#alphas[i]/2)
            else:
                for i, col in enumerate(cols[2:]):
                    for j in [0, 1]:
                        ax.plot(df[col[j]], color=colors[i], alpha=alphas[i],
                                label=col, linewidth=0.5)
    # advance
    x0 = 0
    y = df.loc[x0][cols[0]]
    adf = df.loc[-20:0, [cols[1]]]
    i1 = (adf - y).abs().values.flatten().argsort()[0]
    x1 = adf.index[i1]
    ax.plot(x0, y, 'o', color=std_colors['blue'])
    ax.plot(x1, y, marker=markers.CARETLEFT, color=std_colors['blue'])
    ax.hlines(y, x1, x0, color=std_colors['blue'], linestyle=':')

    ax.annotate("n=10", xy=(0.2, 0.8),
                xycoords="axes fraction", ha='center')
    # adv = str(x0 - x1)
    # ax.annotate(r"$\Delta$=" +  adv, xy= (0.2, 0.73),
                #xycoords="axes fraction", ha='center')
    # pop spike
    cols = colsdict['popSpk']
    ax = spkaxes[1]
    for i, col in enumerate(cols[::-1]):
        ax.plot(df[col], color=colors[::-1][i],
                alpha=1, label=col)#, linewidth=1)
        # ax.fill_between(df.index, df[col],
        #                 color=colors[::-1][i], alpha=0.5, label=col)
    ax.annotate("n=20", xy=(0.2, 0.8),
                xycoords="axes fraction", ha='center')
    # popSpkSig
    cols = colsdict['popSpkSig']
    ax = spkaxes[2]
    # traces
    for i, col in enumerate(cols[:2][::-1]):
        ax.plot(df[col], color=inv_colors[i], alpha=1, label=col)
    # errors : iterate on tuples
    for i, col in enumerate(cols[2:]):
        ax.fill_between(df.index, df[col[0]], df[col[1]], color=colors[i],
                        alpha=inv_alphas[i]/2)# label=col, linewidth=0.5)
    # advance
    x0 = 0
    y = df.loc[x0][cols[0]]
    adf = df.loc[-20:0, [cols[1]]]
    i1 = (adf - y).abs().values.flatten().argsort()[0]
    x1 = adf.index[i1]
    ax.plot(x0, y, 'o', color=std_colors['blue'])
    ax.plot(x1, y, marker=markers.CARETLEFT, color=std_colors['blue'])
    ax.hlines(y, x1, x0, color=std_colors['blue'], linestyle=':')

    ax.annotate("n=5", xy=(0.2, 0.8),
                xycoords="axes fraction", ha='center')
    # #advance
    # adv = str(x0 - x1)
    # ax.annotate(r"$\Delta$=" +  adv, xy= (0.2, 0.73),
                # xycoords="axes fraction", ha='center')
    # labels
    for ax in axes:
        for spine in ['top', 'right']:
            ax.spines[spine].set_visible(False)
    ylabels = ['Membrane potential (mV)',
               'Normalized membrane potential',
               '', '', '', '']
    for i, ax in enumerate(vmaxes):
        ax.axes.get_xaxis().set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.set_ylabel(ylabels[i])
    ylabels = ['Firing rate (spikes/s)',
               'Normalized firing rate',
               '', '', '', '']
    for i, ax in enumerate(spkaxes):
        ax.set_ylabel(ylabels[i])
        ax.set_xlabel('Time (ms)')

    for ax in vmaxes[1:]:
        ax.set_ylim(-0.10, 1.2)
    for ax in spkaxes[1:]:
        ax.set_ylim(-0.10, 1.3)
        ax.set_xlabel('Relative time (ms)')

    # stimulations
    step = 28
    xlocs = np.arange(0, -150, -step)
    names = ['D0', 'D1', 'D2', 'D3', 'D4', 'D5']
    dico = dict(zip(names, xlocs))
    # lines
    for ax in [vmaxes[0], spkaxes[0]]:
        lims = ax.get_ylim()
        for dloc in xlocs:
            ax.vlines(dloc, lims[0], lims[1], linestyle=':', alpha=0.2)
    # stim location
    ax = spkaxes[0]
    for key in dico.keys():
        ax.annotate(key, xy=(dico[key]+3, -3), alpha=0.6, fontsize='x-small')
        # stim
        rect = Rectangle(xy=(dico[key], -4), width=step, height=1, fill=True,
                         alpha=0.6, edgecolor='w', facecolor=std_colors['red'])
        ax.add_patch(rect)
    # center
    rect = Rectangle(xy=(0, -5), width=step, height=1, fill=True,
                     alpha=0.6, edgecolor='w', facecolor='k')
    ax.add_patch(rect)
    # fit individual example
    vmaxes[0].set_ylim(-3.5, 12)
    spkaxes[0].set_ylim(-5.5, 18)
    # align zero between plots  NB ref = first plot
    for i in [0, 1]:
        gfuc.align_yaxis(vmaxes[i], 0, vmaxes[i+1], 0)
        gfuc.align_yaxis(spkaxes[i], 0, spkaxes[i+1], 0)
    # adjust amplitude (without moving the zero)
    for i in [1, 2]:
        gfuc.change_plot_trace_amplitude(vmaxes[i], 0.85)
        gfuc.change_plot_trace_amplitude(spkaxes[i], 0.8)
    # zerolines
    for ax in axes:
        lims = ax.get_ylim()
        ax.vlines(0, lims[0], lims[1], alpha=0.2)
        lims = ax.get_xlim()
        ax.hlines(0, lims[0], lims[1], alpha=0.2)
    # individuals
    ax = vmaxes[0]
    custom_ticks = np.linspace(-2, 10, 7, dtype=int)
    ax.set_yticks(custom_ticks)
    ax.set_yticklabels(custom_ticks)
    ax = spkaxes[0]
    custom_ticks = np.linspace(0, 15, 4, dtype=int)
    ax.set_yticks(custom_ticks)
    ax.set_yticklabels(custom_ticks)
    # pop
    for ax in vmaxes[1:]:
        custom_ticks = np.linspace(0, 1, 6)
        ax.set_yticks(custom_ticks)
    for ax in spkaxes[1:]:
        custom_ticks = np.linspace(0, 1.2, 7)
        ax.set_yticks(custom_ticks)

    fig.tight_layout()
    # remove the space between plots
    fig.subplots_adjust(hspace=0.06, wspace=0.4)
    # align ylabels
    fig.align_ylabels()

    if anot:
        date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        fig.text(0.99, 0.01, 'centrifigs.py:plot_figure2',
                 ha='right', va='bottom', alpha=0.4)
        fig.text(0.01, 0.01, date, ha='left', va='bottom', alpha=0.4)

    return fig

#data
fig2_df, fig2_cols = ldat.load2('old')
fig = plot_figure2(fig2_df, fig2_cols, anot=anot)

# =============================================================================
## other views
# #plot all
# fig = ofig.plot_2_indMoySigNsig(fig2_df, fig2_cols, std_colors, anot=anot)
# #plot ind + pop
# fig = ofig.plot_2_indMoy(fig2_df, fig2_cols, std_colors, anot)
# #sig Nsig
# fig = ofig.plot_2_sigNsig(fig2_df, fig2_cols, std_colors, anot=anot)
# =============================================================================

#%% NB the length of the sorted data are not the same compared to the other traces
#filename = 'fig2cells.xlsx'
#df = pd.read_excel(filename)

plt.close('all')

def plot_figure2B(std_colors, sig=True, anot=anot, age='new'):
    """
    plot_figure2B : sorted phase advance and delta response
    sig=boolan : true <-> shown cell signification
    """
    if age =='old':
        filename = 'data/old/fig2cells.xlsx'
        print('old file fig2cells.xlsx')
        df = pd.read_excel(filename)
        rename_dict = {'popVmscpIsolatg' : 'cpisosect_lat50',
                        'lagIndiSig' : 'cpisosect_lat50_sig',
                        'popVmscpIsoAmpg' : 'cpisosect_gain50',
                        'ampIndiSig' : 'cpisosect_gain50_sig' }
        df.rename(columns=rename_dict, inplace=True)    
    elif age == 'new':
        latGain50_v_df = ldat.load_cell_contributions('vm', amp='gain', age='new')
        cols = latGain50_v_df.columns
        df = latGain50_v_df[[item for item in cols if 'cpisosect' in item]].copy()
        df.sort_values(by=df.columns[0], ascending=False, inplace=True)
    else:
        print('fig2cells.xlsx to be updated')
        return
    #    df = pd.read_excel(filename)
    vals = [item for item in df.columns if '_sig' not in item]
    signs = [item for item in df.columns if '_sig' in item]

#    df.index += 1 # cells = 1 to 37
    color_dic = {0 :'w', 1 : std_colors['red']}

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(17.6, 4))
    for i, ax in enumerate(axes):
        colors = [color_dic[x] for x in df[signs[i]]]
        toplot = df.sort_values(by=vals[i], ascending=False)
        if sig:
            axes[i].bar(toplot.index, toplot[vals[i]], edgecolor=std_colors['red'],
                        color=colors, label=vals[i], alpha=0.8, width=0.8)
        else:
            axes[i].bar(toplot.index, toplot[vals[i]], edgecolor=std_colors['red'],
                        color=std_colors['red'], label=vals[i],
                        alpha=0.8, width=0.8)
        # zero line
        lims = ax.get_xlim()
        ax.hlines(0, lims[0], lims[1], alpha=0.2)
        # ticks
        ax.set_xlim(-1, len(df))
        ax.set_xticks([0, len(df) - 1])
        ax.set_xticklabels([1, len(df)])
        ax.set_xlabel('Cell rank')
        ax.xaxis.set_label_coords(0.5, -0.025)
        if i == 0:
            txt = r'$\Delta$ Phase (ms)'
            ylims = (-6, 29)
            ax.vlines(-1, 0, 20, linewidth=2)
            custom_yticks = np.linspace(0, 20, 3, dtype=int)
        else:
            txt = r'$\Delta$ Amplitude'
            ylims = ax.get_ylim()
            ax.vlines(-1, 0, 0.6, linewidth=2)
            custom_yticks = np.linspace(0, 0.6, 4)
        ax.set_yticks(custom_yticks)
        ax.set_ylabel(txt)
        ax.set_ylim(ylims)
        for spine in ['left', 'top', 'right', 'bottom']:
            ax.spines[spine].set_visible(False)
    # align zero between plots
    gfuc.align_yaxis(axes[0], 0, axes[1], 0)
    gfuc.change_plot_trace_amplitude(axes[1], 0.8)
    fig.tight_layout()
    # anot
    if anot:
        date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        fig.text(0.99, 0.01, 'centrifigs.py:plot_figure2B',
                 ha='right', va='bottom', alpha=0.4)
        fig.text(0.01, 0.01, date, ha='left', va='bottom', alpha=0.4)
    return fig


def sort_stat(age='new'):
    if age == 'old':
        filename = 'data/old/fig2cells.xlsx'
        print('old file fig2cells.xlsx')
        df = pd.read_excel(filename)
        rename_dict = {'popVmscpIsolatg' : 'cpisosect_lat50',
                       'lagIndiSig' : 'cpisosect_lat50_sig',
                       'popVmscpIsoAmpg' : 'cpisosect_gain50',
                       'ampIndiSig' : 'cpisosect_gain50_sig' }
        df.rename(columns=rename_dict, inplace=True)
    elif age == 'new':
        latGain50_v_df = ldat.load_cell_contributions('vm', amp='gain', age='new')
        cols = latGain50_v_df.columns
        df = latGain50_v_df[[item for item in cols if 'cpisosect' in item]]
    else:
        print('fig2cells.xlsx to be updated')
        return
    vals = [item for item in df.columns if '_sig' not in item]
    sigs = [item for item in df.columns if '_sig' in item]
 #   df.index += 1 # cells = 1 to 37
    # all cells:
    print('=== all cells ===')
    all1 = df.cpisosect_lat50
    all2 = df.cpisosect_gain50
    for item, temp in zip(['latency', 'gain'], [all1, all2]):
        print(item, len(temp), 'measures')
        print('mean= {:5.2f}'.format(temp.mean()))
        print('std= {:5.2f}'.format(temp.std()))
        print('sem= {:5.2f}'.format(temp.sem()))
    print('=== sig cells ===')
    temp1 = df.loc[df.cpisosect_lat50_sig == 1, ['cpisosect_lat50']]
    temp2 = df.loc[df.cpisosect_gain50_sig == 1, ['cpisosect_gain50']]
    for item, temp in zip(['latency', 'gain'], [temp1, temp2]):
        print(item, len(temp), 'measures')
        print('mean= {:5.2f}'.format(temp.mean()[0]))
        print('std= {:5.2f}'.format(temp.std()[0]))
        print('sem= {:5.2f}'.format(temp.sem()[0]))

plot_figure2B(std_colors, anot=anot, age='new')
sort_stat('new')
fig = figp.plot_2B_bis(std_colors, anot=anot)

#%%
plt.close('all')


def plot_figure3(std_colors, kind='sig', substract=False, anot=anot, age='new'):
    """
    plot_figure3
    input : kind in ['pop': whole population, 'sig': individually significants
    cells, 'nonsig': non significant cells]
    substract = boolan -> present as (data - centerOnly)
    """
    if age == 'old':
        filenames = {'pop' : 'data/old/fig3.xlsx',
                     'sig': 'data//old/fig3bis1.xlsx',
                     'nonsig': 'data/old/fig3bis2.xlsx'}
    else:
        print('fig3 should be updated')
        return

    titles = {'pop' : 'all cells',
              'sig': 'individually significant cells',
              'nonsig': 'individually non significants cells'}
    # samplesize
    cellnumbers = {'pop' : 37, 'sig': 10, 'nonsig': 27}
    ncells = cellnumbers[kind]
    df = pd.read_excel(filenames[kind])
    # centering
    middle = (df.index.max() - df.index.min())/2
    df.index = df.index - middle
    df.index = df.index/10
    cols = ['CENTER-ONLY', 'CP-ISO', 'CF-ISO', 'CP-CROSS', 'RND-ISO']
    df.columns = cols
    colors = ['k', std_colors['red'], std_colors['green'],
              std_colors['yellow'], std_colors['blue']]
    alphas = [0.8, 1, 0.8, 0.8, 0.8]
    if substract:
        # subtract the centerOnly response
        ref = df['CENTER-ONLY']
        df = df.subtract(ref, axis=0)

    fig = plt.figure(figsize=(6.5, 5.5))
    if anot:
        fig.suptitle(titles[kind], alpha=0.4)
    ax = fig.add_subplot(111)
    for i, col in enumerate(cols):
        ax.plot(df[col], color=colors[i], alpha=alphas[i], label=col, linewidth=2)
    ax.set_ylabel('Normalized membrane potential')
    ax.set_xlabel('Relative time (ms)')
    for ax in fig.get_axes():
        for loc in ['top', 'right']:
            ax.spines[loc].set_visible(False)
    ax.set_xlim(-15, 30)
    lims = ax.get_ylim()
    ax.vlines(0, lims[0], lims[1], alpha=0.2)
    # ax.vlines(0, -0.2, 1.1, alpha=0.2)
    lims = ax.get_xlim()
    ax.hlines(0, lims[0], lims[1], alpha=0.2)
    ax.set_ylim(-0.2, 1.1)
    custom_ticks = np.arange(0, 1.1, 0.2)
    ax.set_yticks(custom_ticks)
    custom_ticks = np.arange(-10, 31, 10)
    ax.set_xticks(custom_ticks)
    # bluePoint
    ax.plot(0, df.loc[0]['CENTER-ONLY'], 'o', color=colors[-1])
    # leg = ax.legend(loc='center right', markerscale=None, frameon=False,
    # leg = ax.legend(loc=2, markerscale=None, frameon=False,
                    #handlelength=0)
    # for line, text in zip(leg.get_lines(), leg.get_texts()):
        # text.set_color(line.get_color())
    ax.annotate('n=' + str(ncells), xy=(0.1, 0.8),
                xycoords="axes fraction", ha='center')
    if substract:
        ax.set_xlim(-45, 120)
        ax.set_ylim(-0.15, 0.4)
        custom_ticks = np.arange(-40, 110, 20)
        ax.set_xticks(custom_ticks)
        # max center only
        lims = ax.get_ylim()
        ax.vlines(21.4, lims[0], lims[1], alpha=0.5)
        # end of center only
        #(df['CENTER-ONLY'] - 0.109773).abs().sort_values().head()
        ax.vlines(88, lims[0], lims[1], alpha=0.3)
        ax.axvspan(0, 88, facecolor='k', alpha=0.2)
        ax.text(0.4, 0.9, 'center only response \n start | peak | end',
                transform=ax.transAxes, alpha=0.5)
        ax.set_ylabel('Norm vm - Norm centerOnly')
    fig.tight_layout()

    if anot:
        date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        fig.text(0.99, 0.01, 'centrifigs.py:plot_figure3(' + kind + ')',
                 ha='right', va='bottom', alpha=0.4)
        fig.text(0.01, 0.01, date, ha='left', va='bottom', alpha=0.4)
    return fig

fig = plot_figure3(std_colors, 'pop', age='old')
fig = plot_figure3(std_colors, 'sig', age='old')
#fig = plot_figure3('nonsig')
fig = plot_figure3(std_colors, 'sig', substract=True, age='old')
fig2 = plot_figure3(std_colors, 'pop', substract=True, age='old')

#pop all cells
#%% grouped sig and non sig
plt.close('all')
fig = ofig.plot_3_signonsig(std_colors, anot=anot)
fig2 = ofig.plot_3_signonsig(std_colors, substract=True, anot=anot)
#%%
plt.close('all')

def plot_figure4(substract=False):
    """ speed """
    filename = 'data/data_to_use/fig4.xlsx'
    df = pd.read_excel(filename)
    # centering
    middle = (df.index.max() - df.index.min())/2
    # OBSERVATION bottom raw 0 baseline has been decentered by police and ticks size changes
    df.index = df.index - middle
    df.index = df.index/10
    cols = ['centerOnly', '100%', '70%', '50%', '30%']
    df.columns = cols
    colors = ['k', std_colors['red'], speedColors['dark_orange'],
              speedColors['orange'], speedColors['yellow']]
    alphas = [0.8, 1, 0.8, 0.8, 1]
    if substract:
        ref = df.centerOnly
        df = df.subtract(ref, axis=0)
        # stack
        # stacks = []
        # for i, col in enumerate(df.columns[:-5:-1]):
        #     df[col] += i / 10 * 2
        #     stack.append(i / 10 * 2)
    fig = plt.figure(figsize=(7, 5.5))
   # fig.suptitle(os.path.basename(filename))
    ax = fig.add_subplot(111)
    for i, col in enumerate(cols):
        ax.plot(df[col], color=colors[i], alpha=alphas[i],
                label=col)
    ax.set_ylabel('Normalized membrane potential')
    # fontname = 'Arial', fontsize = 14)
    ax.set_xlabel('Relative time (ms)')
    for ax in fig.get_axes():
        for loc in ['top', 'right']:
            ax.spines[loc].set_visible(False)
    fig.tight_layout()
    # fig.legend()
    ax.set_xlim(-40, 45)
    ax.set_ylim(-0.1, 1.1)
    lims = ax.get_ylim()
    ax.vlines(0, lims[0], lims[1], alpha=0.2)
    lims = ax.get_xlim()
    ax.hlines(0, lims[0], lims[1], alpha=0.2)
    custom_ticks = np.linspace(-40, 40, 5)
    ax.set_xticks(custom_ticks)

    # leg = ax.legend(loc='upper left', markerscale=None, frameon=False,
    #                handlelength=0)
    # for line, text in zip(leg.get_lines(), leg.get_texts()):
    #    text.set_color(line.get_color())
    ax.annotate("n=12", xy=(0.1, 0.8),    #xy=(0.2,0.8)
                xycoords="axes fraction", ha='center')
    # bluePoint
    ax.plot(0, df.loc[0]['centerOnly'], 'o', color=std_colors['blue'])
    if substract:
        ax.set_ylim(-0.05, 0.4)
        custom_ticks = np.linspace(0, 0.3, 4)
        ax.set_yticks(custom_ticks)
        ax.set_xlim(-80, 60)
        lims = ax.get_xlim()
        # for i, col in enumerate(df.columns)):
        #     ax.hline()
    if anot:
        date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        fig.text(0.99, 0.01, 'centrifigs.py:plot_figure4',
                 ha='right', va='bottom', alpha=0.4)
        fig.text(0.01, 0.01, date, ha='left', va='bottom', alpha=0.4)

    return fig

fig = plot_figure4()
fig = plot_figure4(substract=True)
## alpha=0.8, figsize = 8,6, ha = 'left'
#%% fig 5 <-> sup 7

plt.close('all')

def plot_fig5():
    """
    plot supplementary figure 2: Vm all conditions of FULL stimulation
    input : kind in ['pop': whole population, 'sig': individually significants
    cells, 'nonsig': non significant cells]
    """
#    filenames = ['data/figSup7a.xlsx', 'data/figSup5bis.xlsx']#'data/figSup7b.xlsx']
    filenames = ['data/data_to_use/highspeed.xlsx', 
                 'data/data_to_use/lowspeed.xlsx']
    titles = ['High speed', 'Low speed']

    filename = filenames[0]
    df = pd.read_excel(filename)
    # centering
    middle = (df.index.max() - df.index.min())/2
    df.index = df.index - middle
    df.index = df.index/10
    # reduce the time range
    df = df.loc[-100:300]
    # remove the negative values
    for col in df.columns:
        df[col].loc[df[col] < 0] = 0
    cols = ['scp-Iso-Stc-HighSpeed', 'scp-Cross-Stc-HighSpeed']#,
           # 'scp-Cross-Stc-LowSpeed', 'scp-Iso-Stc-LowSpeed']
    df.columns = cols
    colors = [std_colors['red'], std_colors['yellow']]
    darkcolors = [std_colors['dark_red'], std_colors['dark_yellow']]
    alphas = [0.7, 0.7]

    fig = plt.figure(figsize=(4, 7))
    ax1 = fig.add_subplot(211)
    for i, col in enumerate(cols[:2]):
        ax1.fill_between(df.index, df[col], facecolor=colors[i],
                         edgecolor='black', alpha=1, linewidth=1)
    ax1.axes.get_xaxis().set_visible(False)
    ax1.spines['bottom'].set_visible(False)
    ax1.set_ylim(0, 5.5)

    filename = filenames[1]
    df = pd.read_excel(filename)
    # centering
    middle = (df.index.max() - df.index.min())/2
    df.index = df.index - middle
    df.index = df.index/10
    # reduce the time range
    df = df.loc[-100:300]
    # remove the negative values
    for col in df.columns:
        df[col].loc[df[col] < 0] = 0

    cols = ['scp-Cross-Stc-LowSpeed', 'scp-Iso-Stc-LowSpeed']
    df.columns = cols
    colors = colors[::-1]
    darkcolors = darkcolors[::-1]
    ax2 = fig.add_subplot(212)
    for i, col in enumerate(cols[:2]):
        ax2.fill_between(df.index, df[col], facecolor=colors[i],
                         edgecolor='black', alpha=1, linewidth=1)
    ax2.axes.get_xaxis().set_visible(True)
    ax2.spines['bottom'].set_visible(True)
    ax2.set_ylim(0, 11.5)
    ax2.set_xlabel('Time (ms)')

    ax1.annotate('100°/s', xy=(0.2, 0.95),
                 xycoords="axes fraction", ha='center')

    ax2.annotate('5°/s', xy=(0.2, 0.95),
                 xycoords="axes fraction", ha='center')

    for ax in fig.get_axes():
        lims = ax.get_ylim()
        ax.vlines(0, lims[0], lims[1], alpha=0.1)
        lims = ax.get_xlim()
        ax.hlines(0, lims[0], lims[1], alpha=0.1)
        for loc in ['top', 'right']:
            ax.spines[loc].set_visible(False)

    fig.tight_layout()
    fig.text(0.02, 0.5, 'Firing rate (spk/s)',
             va='center', rotation='vertical')
    # remove the space between plots
    fig.subplots_adjust(hspace=0.1)

    if anot:
        date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        fig.text(0.99, 0.01, 'centrifigs.py:plot_fig5',
                 ha='right', va='bottom', alpha=0.4)
        fig.text(0.01, 0.01, date, ha='left', va='bottom', alpha=0.4)
    return fig

fig = plot_fig5()
#%%

plt.close('all')

def plot_figure6(std_colors):
    """
    plot_figure6
    """
    filename = 'data/fig5.xlsx'
    df = pd.read_excel(filename)
    # centering
    middle = (df.index.max() - df.index.min())/2
    df.index = df.index - middle
    df.index = df.index/10
    # rename columns
    cols = df.columns
    cols = ['Center-Only', 'Surround-then-Center', 'Surround-Only',
            'Static linear prediction']
    dico = dict(zip(df.columns, cols))
    df.rename(columns=dico, inplace=True)
    # color parameters
    colors = ['k', std_colors['red'], std_colors['dark_green'], std_colors['dark_green']]
    alphas = [0.6, 0.8, 0.8, 0.8]
    # plotting
    fig = plt.figure(figsize=(8.5, 8))
    # fig.suptitle(os.path.basename(filename))
    ax1 = fig.add_subplot(211)
    for i, col in enumerate(cols[:2]):
        ax1.plot(df.loc[-120:200, [col]], color=colors[i], alpha=alphas[i],
                 label=col)
    ax1.spines['bottom'].set_visible(False)
    ax1.axes.get_xaxis().set_visible(False)

    ax2 = fig.add_subplot(212, sharex=ax1, sharey=ax1)
    for i, col in enumerate(cols):
        if i == 3:
            ax2.plot(df.loc[-120:200, [col]], color=colors[i], alpha=alphas[i],
                     label=col, linestyle='--', linewidth=1.5)
        else:
            ax2.plot(df.loc[-120:200, [col]], color=colors[i], alpha=alphas[i],
                     label=col)
    ax2.set_xlabel('Time (ms)')
    # stims
    step = 21
    hlocs = np.arange(0, -110, -step)
    names = ['D0', 'D1', 'D2', 'D3', 'D4', 'D5']
    # vlocs = np.linspace(-0.7, -1.7, 4)
    vlocs = np.linspace(-1.4, -2.4, 4)
    dico = dict(zip(names, hlocs))

    # ax1
    for key in dico.keys():
        # name
        ax1.annotate(key, xy=(dico[key]+6, vlocs[0]), alpha=0.6,
                     annotation_clip=False, fontsize='small')
        # stim1
        rect = Rectangle(xy=(dico[key], vlocs[1]), width=step, height=0.3,
                         fill=True, alpha=0.6, edgecolor='w',
                         facecolor=colors[1])
        ax1.add_patch(rect)
    # center
    rect = Rectangle(xy=(0, vlocs[2]), width=step, height=0.3, fill=True,
                     alpha=0.6, edgecolor='w', facecolor=colors[0])#'k')
    ax1.add_patch(rect)

    st = 'Surround-then-Center'
    ax1.annotate(st, xy=(30, vlocs[1]), color=colors[1], alpha=1,
                 annotation_clip=False, fontsize='small')
    st = 'Center-Only'
    ax1.annotate(st, xy=(30, vlocs[2]), color=colors[0], alpha=1,
                 annotation_clip=False, fontsize='small')
        # see annotation_clip=False
    ax1.set_ylim(-2.5, 4.5)

    # ax2
    for key in dico.keys():
        # names
        ax2.annotate(key, xy=(dico[key]+6, vlocs[0]), alpha=0.6,
                     annotation_clip=False, fontsize='small')
        # stim1
        rect = Rectangle(xy=(dico[key], vlocs[1]), width=step, height=0.3,
                         fill=True, alpha=1, edgecolor='w',
                         facecolor=colors[2])
        if key == 'D0':
            rect = Rectangle(xy=(dico[key], vlocs[1]), width=step, height=0.3,
                             fill=True, alpha=1, edgecolor=colors[2],
                             facecolor='w')
        ax2.add_patch(rect)
        # stim2
        rect = Rectangle(xy=(dico[key], vlocs[2]), width=step, height=0.3,
                         fill=True, alpha=0.6, edgecolor='w',
                         facecolor=colors[1])
        ax2.add_patch(rect)
    #center
    rect = Rectangle(xy=(0, vlocs[3]), width=step, height=0.3, fill=True,
                     alpha=0.6, edgecolor='w', facecolor=colors[0])
    ax2.add_patch(rect)
    for i, st in enumerate(['Surround-Only', 'Surround-then-Center', 'Center-Only']):
        ax2.annotate(st, xy=(30, vlocs[i+1]), color=colors[2-i],
                     annotation_clip=False, fontsize='small')
    for ax in fig.get_axes():
        # leg = ax.legend(loc='upper right', markerscale=None, frameon=False,
        #                handlelength=0)
        # colored text
        # for line, text in zip(leg.get_lines(), leg.get_texts()):
            # text.set_color(line.get_color())
        ax.set_ylabel('Membrane potential (mV)')
        for loc in ['top', 'right']:
            ax.spines[loc].set_visible(False)
        lims = ax.get_xlim()
        ax.hlines(0, lims[0], lims[1], alpha=0.2)
        lims = ax.get_ylim()
        ax.vlines(0, lims[0], lims[1], alpha=0.2)
        # response start
        x = 41
        y = df['Center-Only'].loc[x]
        ax.plot(x, y, 'o', color=std_colors['blue'])
        ax.vlines(x, -0.5, lims[1], color=std_colors['blue'],
                  linestyle=':', alpha=0.8)
        for dloc in hlocs:
            ax.vlines(dloc, lims[0], lims[1], linestyle=':', alpha=0.2)
        #ticks
        custom_ticks = np.linspace(0, 4, 5, dtype=int)
        ax.set_yticks(custom_ticks)
        ax.set_yticklabels(custom_ticks)
    fig.tight_layout()

    if anot:
        date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        fig.text(0.99, 0.01, 'centrifigs.py:plot_figure6',
                 ha='right', va='bottom', alpha=0.4)
        fig.text(0.01, 0.01, date, ha='left', va='bottom', alpha=0.4)

    return fig

fig = plot_figure6(std_colors)

#%%
plt.close('all')

fig = figp.plot_figure6_bis(std_colors)
# fig = plot_figure6_bis(substract=True)
fig = figp.plot_figure6_bis(std_colors, linear=False, substract=True)


#%%
plt.close('all')


def plot_figure7(std_colors):
    """
    plot_figure7
    """
    filename = 'data/fig6.xlsx'
    df = pd.read_excel(filename)
    #centering
    middle = (df.index.max() - df.index.min())/2
    df.index = df.index - middle
    df.index = df.index/10
    #limit the date time range
    df = df.loc[-150:150]
    cols = ['centerOnly', 'surroundThenCenter', 'surroundOnly'
            'sdUp', 'sdDown', 'linearPrediction']
    dico = dict(zip(df.columns, cols))
    df.rename(columns=dico, inplace=True)
    cols = df.columns
    colors = ['k', 'r', 'b', 'g', 'b', 'b']
    colors = ['k', std_colors['red'], std_colors['dark_green'],
              std_colors['blue_violet'], std_colors['blue_violet'],
              std_colors['blue_violet']]
    alphas = [0.5, 0.7, 0.7, 0.6, 0.6, 0.6]

    fig = plt.figure(figsize=(11.6, 5))
   # fig.suptitle(os.path.basename(filename))
    ax1 = fig.add_subplot(121)
    for i, col in enumerate(cols[:3]):
        if i == 2:
            ax1.plot(df[col], color=colors[i], alpha=alphas[i],
                     linewidth=2, label=col)
        else:
            ax1.plot(df[col], color=colors[i], alpha=alphas[i],
                     label=col)
    x = 0
    y = df.centerOnly.loc[0]
    ax1.plot(x, y, 'o', color=std_colors['blue'])
    # ax1.hlines(y, -150, 10, colors=std_colors['blue'], alpha=0.5)
    ax1.set_ylim(-0.2, 1)
    ax2 = fig.add_subplot(122, sharex=ax1)
    for i in [2, 5]:
        print('i=', i, colors[i])
        ax2.plot(df[df.columns[i]], color=colors[i], alpha=alphas[i],
                 label=df.columns[i])
    ax2.fill_between(df.index, df[df.columns[3]], df[df.columns[4]],
                     color=colors[2], alpha=0.2)
    # ax2.set_ylim(-0.2, 0.3)
    # set fontname and fontsize for y label
    ax1.set_ylabel('Normalized membrane potential')
    ax1.annotate("n=12", xy=(0.1, 0.8),
                 xycoords="axes fraction", ha='center')
    for ax in fig.get_axes():
        # leg = ax.legend(loc='upper left', markerscale=None, frameon=False,
        #                handlelength=0)
        # colored text
        # for line, text in zip(leg.get_lines(), leg.get_texts()):
        #    text.set_color(line.get_color())
        # ax.set_xlim(-150, 150)
        # set fontname and fontsize for x label
        ax.set_xlabel('Relative time (ms)')
        #, fontname = 'Arial', fontsize = 14)
        for loc in ['top', 'right']:
            ax.spines[loc].set_visible(False)
        lims = ax.get_xlim()
        ax.hlines(0, lims[0], lims[1], alpha=0.2)
        lims = ax.get_ylim()
        ax.vlines(0, lims[0], lims[1], alpha=0.2)
    # align zero between subplots
    gfuc.align_yaxis(ax1, 0, ax2, 0)
    gfuc.change_plot_trace_amplitude(ax2, 0.9)
    fig.tight_layout()
    # add ref
    ref = (0, df.loc[0, ['centerOnly']])
    custom_ticks = np.arange(0, 1.1, 0.2)
    ax1.set_yticks(custom_ticks)

    if anot:
        date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        fig.text(0.99, 0.01, 'centrifigs.py:plot_figure7',
                 ha='right', va='bottom', alpha=0.4)
        fig.text(0.01, 0.01, date, ha='left', va='bottom', alpha=0.4)

    return fig

fig = plot_figure7(std_colors)
fig2 = figp.plot_figure7_bis(std_colors)


#%% fig 9
plt.close('all')

def plot_figure9CD(data, colsdict):
    """
    plot_figure9CD
    """
    colors = ['k', std_colors['red']]
    alphas = [0.8, 0.8]
    # no individual : focus on initial response
    df = data.loc[-30:35]

    fig = plt.figure(figsize=(11.6, 5)) #fig = plt.figure(figsize=(8, 8))
    ax0 = fig.add_subplot(1, 2, 1)
    cols = colsdict['popVmSig']
    # ax.set_title('significative population')
    # traces
    for i, col in enumerate(cols[:2]):
        ax0.plot(df[col], color=colors[i], alpha=alphas[i],
                 label=col)
        # errors : iterate on tuples
    for i, col in enumerate(cols[2:]):
        ax0.fill_between(df.index, df[col[0]], df[col[1]], color=colors[i],
                         alpha=alphas[i]/2)
    # advance
    x0 = 0
    y = df.loc[x0][cols[0]]
    adf = df.loc[-20:0, [cols[1]]]
    i1 = (adf - y).abs().values.flatten().argsort()[0]
    x1 = adf.index[i1]
    # ax.plot(x0, y, 'o', color=std_colors['blue'])
    # ax.plot(x1, y, '|', color=std_colors['blue'])
    # ax.hlines(y, x1, x0, color=std_colors['blue'])
    # ax.annotate("n=10", xy=(0.2, 0.8),
    #               xycoords="axes fraction", ha='center')
    ylabel = 'Normalized membrane potential'
    ax0.set_ylabel(ylabel)
    ax0.set_ylim(-0.10, 1.2)
    ax0.set_xlabel('Relative time (ms)')
    lims = ax0.get_ylim()
    ax0.vlines(0, lims[0], lims[1], alpha=0.2)
    lims = ax0.get_xlim()
    ax0.hlines(0, lims[0], lims[1], alpha=0.2)
    # lims = ax1.get_ylim()
    custom_ticks = np.arange(0, 1.1, 0.2)
    ax0.set_yticks(custom_ticks)

    # hist
    ax1 = fig.add_subplot(1, 2, 2)
    filename = 'data/fig2cells.xlsx'
    df = pd.read_excel(filename)
    # cols = df.columns[:2]
    # signs = df.columns[2:]
    df.index += 1 # cells = 1 to 37

    nsig = df.loc[df.lagIndiSig == 0].popVmscpIsolatg.tolist()
    sig = df.loc[df.lagIndiSig == 1].popVmscpIsolatg.tolist()

    bins = np.arange(-5, 36, 5) - 2.5
    ax1.hist([sig, nsig], bins=bins, stacked=True,
             color=[std_colors['red'], 'None'], edgecolor='k', linewidth=1)
    ax1.set_xlim(-10, 35)
    # adjust nb of ticks
    lims = ax1.get_ylim()
    custom_ticks = np.linspace(lims[0], 18, 7, dtype=int)
    # custom_ticks = np.arange(0, 13, 4)
    ax1.set_yticks(custom_ticks)
    ax1.set_yticklabels(custom_ticks)
    ax1.vlines(0, lims[0], lims[1], linestyle='--')
    ax1.set_ylabel('Number of cells')
    ax1.set_xlabel(r'$\Delta$ Phase (ms)')

    for ax in fig.get_axes():
        for spine in ['top', 'right']:
            ax.spines[spine].set_visible(False)
    fig.tight_layout()
    # remove the space between plots
    # fig.subplots_adjust(hspace=0.06) #fig.subplots_adjust(hspace=0.02)
    if anot:
        date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        fig.text(0.99, 0.01, 'centrifigs.py:plot_figure9CD',
                 ha='right', va='bottom', alpha=0.4)
        fig.text(0.01, 0.01, date, ha='left', va='bottom', alpha=0.4)
    return fig

# fig2_df, fig2_cols = load2()
plot_figure9CD(fig2_df, fig2_cols)

#%% plot latency (left) and gain (right)

plt.close('all')

def plot_sorted_responses_sup1(overlap=True, sort_all=True, key=0):
    """
    plot the sorted cell responses
    input = conditions parameters
    overlap : boolean, overlap the different rows to superpose the plots
    sort_all : if false, only the 'key' trace is sorted
    key : number to choose the trace to be sorted
    output : matplotlib plot
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

    # parameter
    colors = [std_colors['red'], std_colors['red'],
              std_colors['green'], std_colors['green'],
              std_colors['yellow'], std_colors['yellow'],
              std_colors['blue'], std_colors['blue'],
              std_colors['blue'], std_colors['blue']]
    # data (call)
    # nb builded on data/figSup34Vm.xlsx
    df = ldat.load_cell_contributions('vm')
    # extract list of traces : sector vs full
#    traces = [item for item in df.columns if 's_' in item[:7]]
    traces = [item for item in df.columns if 'sect' in item]
    # append full random
#    f_rnd = [item for item in df.columns if 'vm_f_rnd' in item]
    rdfull = [item for item in df.columns if 'rdisofull' in item]
    for item in rdfull:
        traces.append(item)
    # filter -> only significative cells
    traces = [item for item in traces if 'sig' not in item]
    # text labels
    title = 'Vm (sector)'
    anotx = 'Cell rank'
    anoty = [r'$\Delta$ Phase (ms)', r'$\Delta$ Amplitude']
             #(fraction of Center-only response)']
    # plot
    fig, axes = plt.subplots(5, 2, figsize=(12, 16), sharex=True,
                             sharey='col', squeeze=False)#•sharey=True,
    if anot:
        fig.suptitle(title)
    axes = axes.flatten()
    x = range(1, len(df)+1)
    # use cpisotime for ref
    name = traces[0]
    name = traces[key]
    sig_name = name + '_sig'
    df = df.sort_values(by=[name, sig_name], ascending=False)
    # plot all traces
    for i, name in enumerate(traces):
        sig_name = name + '_sig'
        # color : white if non significant, edgecolor otherwise
        edge_color = colors[i]
        color_dic = {0 : 'w', 1 : edge_color}
        if sort_all:
            select = df[[name, sig_name]].sort_values(by=[name, sig_name],
                                                      ascending=False)
        else:
            select = df[[name, sig_name]]
        bar_colors = [color_dic[x] for x in select[sig_name]]
        ax = axes[i]
        # ax.set_title(str(i))
        ax.bar(x, select[name], color=bar_colors, edgecolor=edge_color,
               alpha=0.8, width=0.8)
        if i in [0, 1]:
            ax.set_title(anoty[i])
    # alternate the y_axis position
    axes = fig.get_axes()
    left_axes = axes[::2]
    right_axes = axes[1::2]
    for axe in [left_axes, right_axes]:
        for i, ax in enumerate(axe):
            ax.set_facecolor('None')
            # ax.set_title(i)
            ax.spines['top'].set_visible(False)
            ax.ticklabel_format(useOffset=True)
            ax.spines['bottom'].set_visible(False)
            # zero line
            lims = ax.get_xlim()
            ax.hlines(0, lims[0], lims[1], alpha=0.2)
            if i != 4:
                ax.xaxis.set_visible(False)
            else:
                ax.set_xlabel(anotx)
                ax.xaxis.set_label_coords(0.5, -0.025)
                ax.set_xticks([1, len(df)])
                ax.set_xlim(0, len(df)+1)
    for ax in left_axes:
        custom_ticks = np.linspace(0, 10, 2, dtype=int)
        ax.set_yticks(custom_ticks)
    for ax in right_axes:
        custom_ticks = np.linspace(0, 0.5, 2)
        ax.set_yticks(custom_ticks)
    no_spines = True
    if no_spines == True:
        for ax in left_axes:
            limx = ax.get_xlim()
            ax.vlines(limx[0], 0, 10, color='k', linewidth=2)
            # ax.vlines(limx[1], 0, -10, color='k', linewidth=2)
            for spine in ['left', 'right']:
                ax.spines[spine].set_visible(False)
        for ax in right_axes:
            limx = ax.get_xlim()
            ax.vlines(limx[0], 0, 0.5, color='k', linewidth=2)
            # ax.vlines(limx[1], 0, -0.5, color='k', linewidth=2)
            for spine in ['left', 'right']:
                ax.spines[spine].set_visible(False)

    # align each row yaxis on zero between subplots
    gfuc.align_yaxis(axes[0], 0, axes[1], 0)
    # keep data range whithout distortion, preserve 0 alignment
    gfuc.change_plot_trace_amplitude(axes[1], 0.80)
    # remove the space between plots
    fig.tight_layout()
    if overlap:
        fig.subplots_adjust(hspace=-0.5, wspace=0.2)
    else:
        fig.subplots_adjust(hspace=0.05, wspace=0.2)
    if anot:
        date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        fig.text(0.99, 0.01, 'centrifigs.py:plot_sorted_responses_sup1',
                 ha='right', va='bottom', alpha=0.4)
        fig.text(0.01, 0.01, date, ha='left', va='bottom', alpha=0.4)
    return fig

fig = plot_sorted_responses_sup1(overlap=True)
fig = plot_sorted_responses_sup1(overlap=True, sort_all=False)

# =============================================================================
# savePath = os.path.join(paths['cgFig'], 'pythonPreview', 'sorted', 'testAllSortingKeys')
# for key in range(10):
#     fig = plot_sorted_responses_sup1(overlap=True, sort_all=False, key=key)
#     filename = os.path.join(savePath, str(key) + '.png')
#     fig.savefig(filename, format='png')

# =============================================================================

#%%
plt.close('all')

def plot_figSup2B(kind='pop'):
    """
    plot supplementary figure 1 : Vm with random Sector control
    input : kind in ['pop': whole population, 'sig': individually significants
    cells, 'nonsig': non significant cells]
    """
    filenames = {'pop' : 'data/figSup3.xlsx',
                 'sig': 'data/figSup3bis.xlsx',
                 'nonsig': 'data/figSup3bis2.xlsx'}
    titles = {'pop' : 'all cells',
              'sig': 'individually significant cells',
              'nonsig': 'individually non significants cells'}
    # samplesize
    cellnumbers = {'pop' : 37, 'sig': 10, 'nonsig': 27}
    ncells = cellnumbers[kind]
    df = pd.read_excel(filenames[kind])
    # centering
    middle = (df.index.max() - df.index.min())/2
    df.index = df.index - middle
    df.index = df.index/10
    cols = ['CENTER-ONLY', 'CP-ISO', 'CF-ISO', 'CP-CROSS',
            'RND-ISO SECTOR', 'RND-ISO-FULL']
    df.columns = cols
    colors = ['k', std_colors['red'], std_colors['green'],
              std_colors['yellow'], std_colors['blue'], std_colors['blue']]
    # alphas = [0.5, 0.8, 0.5, 1, 0.6]
    alphas = [0.8, 0.8, 0.8, 0.8, 0.8, 0.8]
    fig = plt.figure(figsize=(8, 7))
    # SUGGESTION: make y dimension much larger to see maximize visual difference between traces
    if anot:
        fig.suptitle('Vm ' + titles[kind])
    ax = fig.add_subplot(111)
    for i, col in enumerate(cols):
        if i == 4:
            ax.plot(df[col], linestyle='dotted', color=colors[i],
                    alpha=alphas[i], label=col, linewidth=2)
        elif i != 4:
            ax.plot(df[col], color=colors[i], alpha=alphas[i],
                    label=col, linewidth=2)
    ax.set_ylabel('Normalized membrane potential')
    ax.set_xlabel('Relative time (ms)')
    for ax in fig.get_axes():
        for loc in ['top', 'right']:
            ax.spines[loc].set_visible(False)
    ax.set_xlim(-15, 30)
    lims = ax.get_ylim()
    ax.vlines(0, lims[0], lims[1], alpha=0.2)
    # ax.vlines(0, -0.2, 1.1, alpha=0.2)
    lims = ax.get_xlim()
    ax.hlines(0, lims[0], lims[1], alpha=0.2)
    ax.hlines(0, lims[0], lims[1], alpha=0.2)
    ax.set_ylim(-0.1, 1)
    custom_ticks = np.arange(0, 1.1, 0.2)
    ax.set_yticks(custom_ticks)
    custom_ticks = np.arange(-10, 31, 10)
    ax.set_xticks(custom_ticks)
    # blue point
    ax.plot(0, df.loc[0]['CENTER-ONLY'], 'o', color=std_colors['blue'])

    # leg = ax.legend(loc='center right', markerscale=None, frameon=False,
    leg = ax.legend(loc=2, markerscale=None, frameon=False,
                    handlelength=0)
    for line, text in zip(leg.get_lines(), leg.get_texts()):
        text.set_color(line.get_color())
    # ax.annotate('n=' + str(ncells), xy=(0.1, 0.8),
    #            xycoords="axes fraction", ha='center')
    fig.tight_layout()

    if anot:
        date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        fig.text(0.99, 0.01, 'centrifigs.py:plot_figSup2B',
                 ha='right', va='bottom', alpha=0.4)
        fig.text(0.01, 0.01, date, ha='left', va='bottom', alpha=0.4)

    return fig


fig = plot_figSup2B('pop')
#fig = plot_figSup2B('sig')
#fig = plot_figSup2B('nonsig')



#%%
plt.close('all')


def plot_figSup4(kind, overlap=True):
    """
    plot supplementary figure 3:
        Vm all conditions of surround-only stimulation CP-ISO sig
    input : kind in ['minus'', 'plus]
        'minus': Surround-then-center - Center Only Vs Surround-Only,
        'plus': Surround-Only + Center only Vs Surround-then-center]
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

    filenames = {'minus': 'data/figSup2m.xlsx',
                 'plus': 'data/figSup2p.xlsx'}

    titles = {'minus': 'Surround-then-center minus center only',
              'plus': 'Surround-only plus center-only'}

    yliminf = {'minus': -0.15, 'plus': -0.08}
    ylimsup = {'minus': 0.4, 'plus' : 1.14}

    # samplesize
    # cellnumbers = {'minus' : 12, 'plus': 12}
    # ncells = cellnumbers[kind]
    # ylimtinf = yliminf[kind]
    # ylimtsup = ylimsup[kind]
    df = pd.read_excel(filenames[kind])
    # centering
    middle = (df.index.max() - df.index.min())/2
    df.index = df.index - middle
    # adjust time
    df.index = df.index/10
    df = df.loc[-150:150]
    # rename
    cols = ['CP_Iso_Stc', 'CP_Iso_Stc_SeUp',
            'CP_Iso_Stc_SeDw', 'CP_Iso_Stc_Dlp',
            'CF_Iso_Stc', 'CF_Iso_Stc_SeUp',
            'CF_Iso_Stc_SeDw', 'CF_Iso_Stc_Dlp',
            'CP_Cross_Stc', 'CP_Cross_Stc_SeUp',
            'CP_Cross_Stc_SeDw', 'CP_Cross_Stc_Dlp',
            'RND_Iso_Stc_Sec', 'RND_Iso_Stc_SeUp_Sec',
            'RND_Iso_Stc_SeDw_Sec', 'RND_Iso_Stc_Dlp_Sec',
            'RND_Iso_Stc_Full', 'RND_Iso_Stc_SeUp_Full',
            'RND_Iso_Stc_SeDw_Full', 'RND_Iso_Stc_Dlp_Full']
    df.columns = cols
    # colors
    light_colors = [std_colors['red'], std_colors['green'],
                    std_colors['yellow'], std_colors['blue'],
                    std_colors['blue']]
    dark_colors = [std_colors['dark_red'], std_colors['dark_green'],
                   std_colors['dark_yellow'], std_colors['dark_blue'],
                   std_colors['dark_blue'], std_colors['dark_blue']]
    alphas = [0.7, 0.2] # front, fillbetween
    # traces -> lists of 4 columns ie each condition (val, up, down, sum)
    col_seg = [cols[i:i+4] for i in np.arange(0, 17, 4)]

    fig = plt.figure(figsize=(4, 8))
    for i in range(5):
        if i == 0:
            ax = fig.add_subplot(5, 1, i+1)
        else:
            ax = fig.add_subplot(5, 1, i+1, sharex=ax, sharey=ax)
        toPlot = col_seg[i]
        col = toPlot[0]
        # print(col)
        ax.plot(df[col], color=light_colors[i], alpha=alphas[0],
                label=col, linewidth=1.5)
        ax.fill_between(df.index, df[toPlot[1]], df[toPlot[2]],
                        color=light_colors[i], alpha=alphas[1])
        col = toPlot[-1]
        ax.plot(df[col], color=dark_colors[i], alpha=alphas[0],
                label=col, linewidth=1.5)
        # ax.axvspan(xmin=0, xmax=50, ymin=0.27, ymax=0.96,
        #            color='grey', alpha=alphas[1])

        ax.spines['top'].set_visible(False)
        ax.set_facecolor('None')
        # axis on both sides
        set_ticks_both(ax.yaxis)
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
        if i != 4:
            ax.xaxis.set_visible(False)
            ax.spines['bottom'].set_visible(False)
        else:
            ax.set_xlabel('Relative time (ms)')
    for i, ax in enumerate(fig.get_axes()):
        # lims = ax.get_ylim()
        # ax.vlines(0, lims[0], lims[1], alpha=0.3)
        lims = ax.get_xlim()
        ax.hlines(0, lims[0], lims[1], alpha=0.3)
        custom_ticks = np.arange(-0.1, 0.3, 0.1)
        ax.set_yticks(custom_ticks)

    for ax in fig.get_axes():
        lims = ax.get_ylim()
        print(lims)
        r1 = patches.Rectangle((0, 0), 50, 0.4, color='grey',#ax.get_ylim()[1]
                               alpha=0.1)
        ax.add_patch(r1)

    fig.tight_layout()
    if overlap:
        fig.subplots_adjust(hspace=-0.3, wspace=0.2)

    fig.text(0.01, 0.5, 'Normalized membrane potential',
             va='center', rotation='vertical')

    if anot:
        date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        fig.text(0.99, 0.01, 'centrifigs.py:plot_figSup4',
                 ha='right', va='bottom', alpha=0.4)
        fig.text(0.01, 0.01, date, ha='left', va='bottom', alpha=0.4)
    return fig

fig = plot_figSup4('minus', overlap=True)
# fig = plot_figSup4('plus')
# pop all cells
#%%
plt.close('all')


def plot_figSup3B(kind, stimmode):
    """
    plot supplementary figure 5: All conditions spiking responses of Sector and Full stimulations
    input : kind in ['pop': whole population, 'sig': individually significants
    cells, 'nonsig': non significant cells]
    """
    filenames = {'pop' : 'data/figSup5Spk.xlsx'}#,
                 #'sig': 'data/figSup1bis.xlsx',
                 #'nonsig': 'data/figSup1bis2.xlsx'}
    titles = {'pop' : 'all cells'}#,
              #'sig': 'individually significant cells',
              #'nonsig': 'individually non significants cells'}
    #samplesize
    cellnumbers = {'pop' : 20}#, 'sig': x1, 'nonsig': x2}
    ncells = cellnumbers[kind]
    df = pd.read_excel(filenames[kind])
    #centering
    middle = (df.index.max() - df.index.min())/2
    df.index = df.index - middle
    df.index = df.index/10
    cols = ['CENTER-ONLY-SEC', 'CP-ISO-SEC', 'CF-ISO-SEC', 'CP-CROSS-SEC', 'RND-ISO-SEC',
            'CENTER-ONLY-FULL', 'CP-ISO-FULL', 'CF-ISO-FULL', 'CP-CROSS-FULL', 'RND-ISO-FULL']
    df.columns = cols
    colors = ['k', std_colors['red'], std_colors['green'],
              std_colors['yellow'], std_colors['blue']]
    #alphas = [0.5, 0.8, 0.5, 1, 0.6]
    alphas = [0.8, 0.8, 0.8, 0.8, 0.8]
    fig = plt.figure(figsize=(6.5, 5.5))
    # SUGGESTION: make y dimension much larger to see maximize visual difference
    # between traces
    if anot:
        fig.suptitle(titles[kind] + ' spikes')
    ax = fig.add_subplot(111)
    if stimmode == 'sec':
        for i, col in enumerate(cols[:5]):
            ax.plot(df[col], color=colors[i], alpha=alphas[i],
                    label=col, linewidth=2)
    else:
        if stimmode == 'ful':
            for i, col in enumerate(cols[5:]):
                ax.plot(df[col], color=colors[i], alpha=alphas[i],
                        label=col, linewidth=2)

    ax.set_ylabel('Normalized firing rate')
    ax.set_xlabel('Relative time (ms)')
    for ax in fig.get_axes():
        for loc in ['top', 'right']:
            ax.spines[loc].set_visible(False)
    ax.set_xlim(-15, 30)
    custom_ticks = np.arange(-10, 31, 10)
    ax.set_xticks(custom_ticks)
    lims = ax.get_xlim()
    ax.hlines(0, lims[0], lims[1], alpha=0.2)
    ax.set_ylim(-0.2, 1.1)
    lims = ax.get_ylim()
    ax.vlines(0, lims[0], lims[1], alpha=0.2)
    custom_ticks = np.arange(0, 1.1, 0.2)
    ax.set_yticks(custom_ticks)
    # bluePoint
    ax.plot(0, df.loc[0]['CENTER-ONLY-FULL'], 'o', color=colors[-1])

    # leg = ax.legend(loc='center right', markerscale=None, frameon=False,
    leg = ax.legend(loc=2, markerscale=None, frameon=False,
                    handlelength=0)
    for line, text in zip(leg.get_lines(), leg.get_texts()):
        text.set_color(line.get_color())
    # ax.annotate('n=' + str(ncells), xy=(0.1, 0.8),
    #             xycoords="axes fraction", ha='center')
    fig.tight_layout()
    if anot:
        date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        fig.text(0.99, 0.01, 'centrifigs.py:plot_figSup3B',
                 ha='right', va='bottom', alpha=0.4)
        fig.text(0.01, 0.01, date, ha='left', va='bottom', alpha=0.4)

    return fig


# fig = plot_figSup3B('pop', 'sec')
fig = plot_figSup3B('pop', 'ful')

# fig = plot_figSup1('sig')
# fig = plot_figSup1('nonsig')
#%%
plt.close('all')

def plot_figSup6(kind):
    """
    plot supplementary figure 2: Vm all conditions of FULL stimulation
    input : kind in ['pop': whole population,
            'sig': individually significants cells,
            'nonsig': non significant cells]
    """
    filenames = {'pop' : 'data/figSup6.xlsx'}#,
                 # 'sig': 'data/figSup1bis.xlsx',
                 # 'nonsig': 'data/figSup1bis2.xlsx'}
    titles = {'pop' : 'all cells'}#,
              # 'sig': 'individually significant cells',
              # 'nonsig': 'individually non significants cells'}
    # samplesize
    cellnumbers = {'pop' : 37} #, 'sig': 10, 'nonsig': 27}
    ncells = cellnumbers[kind]
    df = pd.read_excel(filenames[kind])
    # centering
    middle = (df.index.max() - df.index.min())/2
    df.index = df.index - middle
    df.index = df.index/10
    cols = ['CENTER-ONLY', 'CP-ISO', 'CF-ISO', 'CP-CROSS', 'RND-ISO']
    df.columns = cols
    colors = ['k', std_colors['red'], std_colors['green'],
              std_colors['yellow'], std_colors['blue']]
    alphas = [0.8, 0.8, 0.8, 0.8, 0.8]

    fig = plt.figure(figsize=(6, 10))
    # fig.suptitle(titles[kind])
    # fig, ax = plt.subplots(nrows = 2, ncols = 1, sharex = True, sharey = True, figsize = (8,7))
    ax1 = fig.add_subplot(211)
    for i, col in enumerate(cols):
        if i in (0, 1, 4):
            ax1.plot(df[col], color=colors[i], alpha=alphas[i], label=col, linewidth=2)
    ax1.axes.get_xaxis().set_visible(False)
    ax1.spines['bottom'].set_visible(False)
    # ax1.set_ylabel('Normalized membrane potential')
    ax1.set_ylim(-0.2, 1.1)

    ax2 = fig.add_subplot(212, sharex=ax1, sharey=ax1)
    for i, col in enumerate(cols):
        if i in (0, 1, 3):
            ax2.plot(df[col], color=colors[i], alpha=alphas[i], label=col, linewidth=2)

    ax2.axes.get_xaxis().set_visible(True)
    ax2.spines['bottom'].set_visible(True)
    # ax2.set_ylabel('Normalized membrane potential')
    ax2.set_ylim(-0.2, 1.1)
    ax2.set_xlabel('Relative time (ms)')

    # axes = list(fig.get_axes())
    # leg = ax.legend(loc='center right', markerscale=None, frameon=False,
    #     leg = ax.legend(loc=2, markerscale=None, frameon=False,
    #                     handlelength=0)
    #     for line, text in zip(leg.get_lines(), leg.get_texts()):
    #         text.set_color(line.get_color())
    # ax.annotate('n=' + str(ncells), xy=(0.1, 0.8),
    #             xycoords="axes fraction", ha='center')

    for ax in fig.get_axes():
        ax.set_xlim(-15, 30)
        ax.set_ylim(-0.2, 1.1)
        lims = ax.get_ylim()
        ax.vlines(0, lims[0], lims[1], alpha=0.1)
        lims = ax.get_xlim()
        ax.hlines(0, lims[0], lims[1], alpha=0.1)
        for loc in ['top', 'right']:
            ax.spines[loc].set_visible(False)

    fig.tight_layout()
    fig.text(-0.04, 0.6, 'Normalized membrane potential', fontsize=16,
             va='center', rotation='vertical')
    # remove the space between plots
    fig.subplots_adjust(hspace=0.1)

    if anot:
        date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        fig.text(0.99, 0.01, 'centrifigs.py:plot_figSup6',
                 ha='right', va='bottom', alpha=0.4)
        fig.text(0.01, 0.01, date, ha='left', va='bottom', alpha=0.4)
    return fig

fig = plot_figSup6('pop')


#%%
plt.close("all")


def extract_values(df, stim='sect', mes='time'):
    """ extract pop and response dico:
        input : dataframe, stim kind (s or f) and mesaure kind (lat or gain)
    """
    # stim = '_' + stim_kind + '_'
    # mes = '_d' + measure + '50'
    # restrict df
    cols = df.columns
    restricted_list = [item for item in cols if stim in item and mes in item]
    adf = df[restricted_list]
    #compute values
    records = [item for item in restricted_list if 'sig' not in item]
    pop_dico = {}
    resp_dico = {}
    for cond in records:
        signi = cond + '_sig'
        pop_num = len(adf)
        signi_num = len(adf.loc[adf[signi] > 0, cond])
        percent = round((signi_num / pop_num) * 100)
        # leg_cond = cond.split('_')[2] + '-' + cond.split('_')[3]
        # leg_cond = leg_cond.upper()
        leg_cond = cond.split('_')[0]
        pop_dico[leg_cond] = [pop_num, signi_num, percent]
        # descr
        moy = adf.loc[adf[signi] > 0, cond].mean()
        stdm = adf.loc[adf[signi] > 0, cond].sem()
        resp_dico[leg_cond] = [moy, moy + stdm, moy - stdm]
    return pop_dico, resp_dico


def autolabel(ax, rects):
    # attach some text labels
    for rect in rects:
        x = rect.get_x() + rect.get_width()/2
        height = rect.get_height()
        y = height - 1
        if y < 3:
            y = height + 1
            ax.text(x, y, '%d' % int(height) + '%',
                    ha='center', va='bottom')
        else:
            ax.text(x, y, '%d' % int(height) + '%',
                    ha='center', va='top')
        #print(y)


def plot_cell_contribution(df, kind=''):
    "sup 2A"
    colors = [std_colors['red'], std_colors['green'],
              std_colors['yellow'], std_colors['blue']]
    dark_colors = [std_colors['dark_red'], std_colors['dark_green'],
                   std_colors['dark_yellow'], std_colors['dark_blue']]
    fig = plt.figure(figsize=(8, 8))
    # if anot:
    #     fig.suptitle('vm')
    # sector phase
    ax = fig.add_subplot(221)
    ax.set_title(r'$\Delta$ Phase (% significant cells)', pad=0)
#    stim = 's'
    stim = 'sect'
#    mes = 'lat'
    mes = 'time50'
    pop_dico, resp_dico = extract_values(df, stim, mes)
    x = pop_dico.keys()
    heights = [pop_dico[item][-1] for item in pop_dico.keys()]
    bars = ax.bar(x, heights, color=colors, width=0.95, alpha=0.8,
                  edgecolor=dark_colors)
    autolabel(ax, bars) # call
    ax.set_ylabel('SECTOR')
    ax.xaxis.set_visible(False)
    # sector amplitude
    ax = fig.add_subplot(222, sharey=ax)
    ax.set_title(r'$\Delta$ Amplitude (% significant cells)', pad=0)
    stim = 'sect'
    mes = 'gain'
    pop_dico, resp_dico = extract_values(df, stim, mes)
    x = pop_dico.keys()
    height = [pop_dico[item][-1] for item in pop_dico.keys()]
    bars = ax.bar(x, height, color=colors, width=0.95, alpha=0.8,
                  edgecolor=dark_colors)
    autolabel(ax, bars)
    ax.xaxis.set_visible(False)
    # full phase
    ax = fig.add_subplot(223, sharey=ax)
    stim = 'full'
    mes = 'time'
    pop_dico, resp_dico = extract_values(df, stim, mes)
    x = pop_dico.keys()
    height = [pop_dico[item][-1] for item in pop_dico.keys()]
    colors = colors
    bars = ax.bar(x, height, color=colors, width=0.95, alpha=0.8,
                  edgecolor=dark_colors)
    autolabel(ax, bars)
    ax.set_ylabel('FULL')

    # full amplitude
    ax = fig.add_subplot(224, sharey=ax)
    stim = 'full'
    mes = 'gain'
    pop_dico, resp_dico = extract_values(df, stim, mes)
    x = pop_dico.keys()
    height = [pop_dico[item][-1] for item in pop_dico.keys()]
    colors = colors
    bars = ax.bar(x, height, color=colors, width=0.95, alpha=0.8,
                  edgecolor=dark_colors)
    autolabel(ax, bars)

    for ax in fig.get_axes():
        for spine in ['left', 'top', 'right']:
            ax.spines[spine].set_visible(False)
            ax.tick_params(axis='x', labelrotation=45)
            ax.yaxis.set_ticklabels([])
            ax.tick_params(axis='y', length=0)

    fig.text(0.5, 1.01, kind, ha='center', va='top', fontsize=18)
    if anot:
        date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        fig.text(0.99, 0.01, 'centrifigs.py:plot_figSup2A',
                 ha='right', va='bottom', alpha=0.4)
        fig.text(0.01, 0.01, date, ha='left', va='bottom', alpha=0.4)

    fig.tight_layout()

kind = 'vm'
df = ldat.load_cell_contributions(kind)
plot_cell_contribution(df, kind)


#%%
plt.close('all')

# TODO: in first figure, 1st condition latency advance of CP-ISO
# plot and fill the actual 10 and 11th df.index significant cell row
# before the actual actual 9th
def plot_sorted_responses(dico):
    """
    plot the sorted cell responses
    input = conditions parameters

    """
    # parameter
    colors = [std_colors['red'], std_colors['red'],
              std_colors['green'], std_colors['green'],
              std_colors['yellow'], std_colors['yellow'],
              std_colors['blue'], std_colors['blue']]
    # data (call)
    df = ldat.load_cell_contributions(dico['kind'])
    # extract list of traces : sector vs full
    traces = [item for item in df.columns if dico['spread'] in item]
    # filter -> only significative cells
    traces = [item for item in traces if 'sig' not in item]
    # text labels
    title = dico['kind'] + ' (' + dico['spread']+ ')'
    # title = title_dico[dico['kind']]
    anotx = 'Cell rank'
    anoty = [r'$\Delta$ phase (ms)', r'$\Delta$ amplitude']
             # (fraction of Center-only response)']
    # plot
    fig, axes = plt.subplots(4, 2, figsize=(12, 16), sharex=True,
                             sharey='col', squeeze=False)#•sharey=True,
    fig.suptitle(title)
    axes = axes.flatten()
    x = range(1, len(df)+1)
    # plot all traces
    for i, name in enumerate(traces):
        sig_name = name + '_sig'
        # color : white if non significant, edgecolor otherwise
        edge_color = colors[i]
        color_dic = {0 : 'w', 1 : edge_color}
        select = df[[name, sig_name]].sort_values(by=[name, sig_name],
                                                  ascending=False)
        bar_colors = [color_dic[x] for x in select[sig_name]]
        ax = axes[i]
        ax.set_title(name)
        ax.bar(x, select[name], color=bar_colors, edgecolor=edge_color,
               alpha=0.8, width=0.8)
        if i in [0, 1]:
            ax.set_title(anoty[i])
    for i, ax in enumerate(axes):
        ax.ticklabel_format(useOffset=True)
        lims = ax.get_xlim()
        ax.hlines(0, lims[0], lims[1], alpha=0.3, linestyle=':')
        for spine in ['left', 'top', 'right', 'bottom']:
            ax.spines[spine].set_visible(False)
        if i in range(6):
            ax.xaxis.set_visible(False)
        else:
            ax.set_xlabel(anotx)
            ax.xaxis.set_label_coords(0.5, -0.025)
            ax.set_xticks([1, len(df)])
            ax.set_xlim(0, len(df)+2)
    # left
    for ax in axes[::2]:
        ax.vlines(0, 0, 20, color='k', linewidth=2)
        custom_ticks = np.linspace(0, 20, 2, dtype=int)
        ax.set_yticks(custom_ticks)
    # right
    for ax in axes[1::2]:
        ax.vlines(0, 0, 1, color='k', linewidth=2)
        custom_ticks = np.linspace(0, 1, 2, dtype=int)
        ax.set_yticks(custom_ticks)
    # align each row yaxis on zero between subplots
    gfuc.align_yaxis(axes[0], 0, axes[1], 0)
    # keep data range whithout distortion, preserve 0 alignment
    gfuc.change_plot_trace_amplitude(axes[1], 0.80)
    # remove the space between plots
    fig.subplots_adjust(hspace=0.00, wspace=0.00)
    if anot:
        date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        fig.text(0.99, 0.01, 'centrifigs.py:plot_sorted_responses',
                 ha='right', va='bottom', alpha=0.4)
        fig.text(0.01, 0.01, date, ha='left', va='bottom', alpha=0.4)
    fig.tight_layout()
    return fig

parameter_dico = {
        'kind' : 'vm',
        'spread' : 'sect',
        'position' : 'cp',
        'theta' : 'cross',
        'extra' : 'stc'
        }

fig = plot_sorted_responses(parameter_dico)

#%%
plt.close('all')

# iterate through conditions for plotting
for kind in ['vm', 'spk']:
    parameter_dico['kind'] = kind
    for spread in ['sect', 'full']:
        parameter_dico['spread'] = spread
        fig = plot_sorted_responses(parameter_dico)

#%% opt
colors = ['k', std_colors['red'], speedColors['dark_orange'],
          speedColors['orange'], speedColors['yellow']]
alphas = [0.8, 1, 0.8, 0.8, 1]

df = pd.read_excel('data/figOpt.xlsx')
df.set_index('time', inplace=True)


def plot_speed_multigraph():
    """
    plot the speed effect of centrigabor protocol
    """
    fig = plt.figure(figsize=(12, 8))
    fig.suptitle('Aligned on Center-Only stimulus onset (t=0 ms)')
    # build grid
    gs = fig.add_gridspec(5, 2)
    left_axes = []
    left_axes.append(fig.add_subplot(gs[4, 0]))
    for i in range(4):
        left_axes.append(fig.add_subplot(gs[i, 0]))
    right_ax = fig.add_subplot(gs[:, 1])
    # to identify the plots (uncomment to use)
    for i, ax in enumerate(left_axes):
        st = str('ax {}'.format(i))
        ax.annotate(st, (0.5, 0.5))
        # ax.set_xtickslabels('', minor=False)
    # (manipulate the left_axes list to reorder the plots if required)
    # axes.set_xticklabels(labels, fontdict=None, minor=False)
    # plot left
    # axes = axes[1:].append(axes[0])   # ctrl at the bottom
    cols = df.columns
    for i, ax in enumerate(left_axes):
        ax.plot(df.loc[-140:40, [cols[i]]], color='black', scalex=False,
                scaley=False, label=cols[i])
        ax.fill_between(df.index, df[cols[i]], color=colors[i])
        ax.yaxis.set_ticks(np.arange(-0.15, 0.25, 0.1))
        ax.set_xlim(-140, 40)
        ax.set_ylim(-0.15, 0.25)
    # add labels
    left_axes[3].set_ylabel('Normalized Membrane potential')
    left_axes[0].set_xlabel('Relative time to center-only onset (ms)')
    left_axes[0].xaxis.set_ticks(np.arange(-140, 41, 40))
    ticks = np.arange(-140, 41, 20)
    for i, ax in enumerate(left_axes[1:]):
        ax.set_xticks(ticks, minor=False)
        ax.tick_params(axis='x', labelsize=0)

    # plot right
    for i, col in enumerate(df.columns):
        right_ax.plot(df.loc[40:100, [col]], color=colors[i],
                      label=col, alpha=alphas[i])
        maxi = float(df.loc[30:200, [col]].max())
        right_ax.hlines(maxi, 40, 50, color=colors[i])
    right_ax.set_xlabel('Relative time to center-only onset (ms)')
    # adjust
    for ax in fig.get_axes():
        for spine in ['top', 'right']:
            ax.spines[spine].set_visible(False)
    for ax in left_axes:
        lims = ax.get_ylim()
        ax.vlines(0, lims[0], lims[1], alpha=0.5)
    # adjust spacing
    gs.update(wspace=0.2, hspace=0.05)
    # add ticks to the top
    right_ax.tick_params(axis='x', bottom=True, top=True)
    # legend
    # leg = right_ax.legend(loc='lower right', markerscale=None,
    #                       handlelength=0, framealpha=1)
    # for line, text in zip(leg.get_lines(), leg.get_texts()):
    #     text.set_color(line.get_color())
    fig.tight_layout()
    if anot:
        date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        fig.text(0.99, 0.01, 'centrifigs.py:plot_speed_multigraph',
                 ha='right', va='bottom', alpha=0.4)
        fig.text(0.01, 0.01, date, ha='left', va='bottom', alpha=0.4)
    return fig

fig = plot_speed_multigraph()

#%% test to analyse with x(t) = x(t) - x(t-1)

def plotSpeeddiff():
    colors = ['k', std_colors['red'], speedColors['dark_orange'],
              speedColors['orange'], speedColors['yellow']]
    alphas = [0.5, 1, 0.8, 0.8, 1]

    df = pd.read_excel('data/figOpt.xlsx')
    df.set_index('time', inplace=True)
    # perform shift (x(t) <- x[t) - x(t-1]
    for col in df.columns:
        df[col] = df[col] - df[col].shift(1)

    fig = plt.figure()
    title = 'speed, y(t) <- y(t) - y(t-1), only positives values'
    fig.suptitle(title)
    ax = fig.add_subplot(111)
    cols = df.columns.to_list()
    cols = cols[::-1]
    for j, col in enumerate(cols):
        i = len(cols) - j -1
        print(i, j)
        xvals = df.loc[-140:100].index
        yvals = df.loc[-140:100, [cols[i]]].values[:, 0]
        # replace negative values <-> negative slope by 0
        yvals = yvals.clip(0)
        ax.fill_between(xvals, yvals + i/400, i/400, color=colors[j],
                        label=cols[i], alpha=alphas[j])
    lims = ax.get_ylim()
    ax.vlines(0, lims[0], lims[1], alpha=0.2)
    for spine in ['left', 'top', 'right']:
        ax.spines[spine].set_visible(False)
    ax.yaxis.set_visible(False)
    fig.legend()
    fig.tight_layout()

    if anot:
        date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        fig.text(0.99, 0.01, 'centrifigs.py:plotSpeeddiff',
                 ha='right', va='bottom', alpha=0.4)
        fig.text(0.01, 0.01, date, ha='left', va='bottom', alpha=0.4)
    return fig

fig = plotSpeeddiff()
