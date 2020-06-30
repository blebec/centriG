

"""
plot centrigabor figures from data stored in .xlsx files
"""

import platform
import os
import getpass
import inspect
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# import matplotlib.gridspec as gridspec
from matplotlib.patches import Rectangle
# from matplotlib.ticker import MaxNLocator
import matplotlib.patches as patches
from matplotlib import markers
from matplotlib.ticker import StrMethodFormatter
from datetime import datetime

import plot_general_functions as gf
import load_data as ld

# nb description with pandas:
pd.options.display.max_columns = 30

#===========================
# global setup
font_size = 'medium'  # large, medium
anot = True           # to draw the date and name on the bottom of the plot
#============================

def build_paths():
    paths = {}
    osname = platform.system()
    username = getpass.getuser()
    if osname == 'Windows'and username == 'Benoit':
        paths['pg'] = r'D:\\travail\sourcecode\developing\paper\centriG'
    elif osname == 'Linux' and username == 'benoit':
        paths['pg'] = r'/media/benoit/data/travail/sourcecode/developing/paper/centriG'
    elif osname == 'Windows'and username == 'marc':
        paths['pg'] = r'H:/pg/centriG'
    elif osname == 'Darwin' and username == 'cdesbois':
        paths['pg'] = os.path.expanduser('~/pg/chrisPg/centriG')
        paths['owncFig'] = os.path.expanduser('~/ownCloud/cgFigures')
    return paths

paths = build_paths()
os.chdir(paths['pg'])

# colors
stdColors = {'rouge' : [x/256 for x in [229, 51, 51]],
             'vert' : [x/256 for x in [127, 204, 56]],
             'bleu' :	[x/256 for x in [0, 125, 218]],
             'jaune' :	[x/256 for x in [238, 181, 0]],
             'violet' : [x/256 for x in [255, 0, 255]],
             'vertSombre': [x/256 for x in [0, 150, 68]],
             'orangeFonce' : [x/256 for x in [237, 73, 59]],
             'bleuViolet': [x/256 for x in [138, 43, 226]],
             'dark_rouge': [x/256 for x in [115, 0, 34]],
             'dark_vert': [x/256 for x in [10, 146, 13]],
             'dark_jaune': [x/256 for x in [163, 133, 16]],
             'dark_bleu': [x/256 for x in [14, 73, 118]]}
speedColors = {'orangeFonce' : [x/256 for x in [237, 73, 59]],
               'orange' : [x/256 for x in [245, 124, 67]],
               'jaune' : [x/256 for x in [253, 174, 74]]}

############################
# NB fig size : 8.5, 11.6 or 17.6 cm
params = {'font.sans-serif': ['Arial'],
          'font.size': 14,
          'legend.fontsize': font_size,
          'figure.figsize': (11.6, 5),
          'figure.dpi'    : 100,
          'axes.labelsize': font_size,
          'axes.titlesize': font_size,
          'xtick.labelsize': font_size,
          'ytick.labelsize': font_size,
          'axes.xmargin': 0}
plt.rcParams.update(params)
plt.rcParams['axes.xmargin'] = 0            # no gap between axes and traces



energy_df = ld.load_energy_gain_index(paths)
data50_v_df = ld.load_50vals('vm')
data50_s_df = ld.load_50vals('spk')

#%%
plt.close('all')


def load2():
    """
    import the datafile
    return a pandasDataframe and a dictionary of contents
    """
    #____data
    filename = 'data/fig2traces.xlsx'
    df = pd.read_excel(filename)
    #centering
    middle = (df.index.max() - df.index.min())/2
    df.index = (df.index - middle)/10
    df = df.loc[-200:150]
    # nb dico : key + [values] or key + [values, (stdUp, stdDown)]
    colsdict = {
        'indVm': ['indiVmctr', 'indiVmscpIsoStc'],
        'indSpk': ['indiSpkCtr', 'indiSpkscpIsoStc'],
        'popVm': ['popVmCtr', 'popVmscpIsoStc'],
        'popSpk': ['popSpkCtr', 'popSpkscpIsoStc'],
        'popVmSig': ['popVmCtrSig', 'popVmscpIsoStcSig',
                     ('popVmCtrSeUpSig', 'popVmCtrSeDwSig'),
                     ('popVmscpIsoStcSeUpSig', 'popVmscpIsoStcSeDwSig')],
        'popSpkSig': ['popSpkCtrSig', 'popSpkscpIsoStcSig',
                      ('popSpkCtrSeUpSig', 'popSpkCtrSeDwSig'),
                      ('popSpkscpIsoStcSeUpSig', 'popSpkscpIsoStcSeDwSig')],
        'popVmNsig': ['popVmCtrNSig', 'popVmscpIsoStcNSig',
                      ('popVmCtrSeUpNSig', 'popVmCtrSeDwNSig'),
                      ('popVmscpIsoStcSeUpNSig', 'popVmscpIsoStcSeDwNSig')],
        'popSpkNsig': ['popSpkCtrNSig', 'popSpkscpIsoStcNSig',
                       ('popSpkCtrSeUpNSig', 'popSpkCtrSeDwNSig'),
                       ('popSpkscpIsoStcSeUpNSig', 'popSpkscpIsoStcSeDwNSig')],
        'sort': ['popVmscpIsolatg', 'popVmscpIsoAmpg',
                 'lagIndiSig', 'ampIndiSig']
                }
    return df, colsdict


def plot_figure2(data, colsdict, fill=True):
    """
    plot_figure2 (individual + moy + sig + nonsig)
    """
    colors = ['k', stdColors['rouge']]
    alphas = [0.8, 0.8]

    #fig = plt.figure(figsize=(inchtocm(17.6),inchtocm(12)))
    fig = plt.figure(figsize=(16.6, 12))
    #build axes with sharex and sharey
    axes = []
    axL = fig.add_subplot(2, 4, 1)
    axes.append(axL)
    ax = fig.add_subplot(2, 4, 2)
    axes.append(ax)
    axes.append(fig.add_subplot(2, 4, 3, sharex=ax, sharey=ax))
    axes.append(fig.add_subplot(2, 4, 4, sharex=ax, sharey=ax))
    axes.append(fig.add_subplot(2, 4, 5, sharex=axL))
    ax = fig.add_subplot(2, 4, 6)
    axes.append(ax)
    axes.append(fig.add_subplot(2, 4, 7, sharex=ax, sharey=ax))
    axes.append(fig.add_subplot(2, 4, 8, sharex=ax, sharey=ax))
    # axes list
    vmaxes = axes[:4]      # vm axes = top row
    spkaxes = axes[4:]     # spikes axes = bottom row
    #____ plots individuals (first column)
    # individual vm
    cols = colsdict['indVm']
    ax = vmaxes[0]
    for i, col in enumerate(cols):
        ax.plot(data[col], color=colors[i], alpha=alphas[i],
                label=col)
    # start point
    x = 41.5
    y = data.indiVmctr.loc[x]
    ax.plot(x, y, 'o', color=stdColors['bleu'])
    lims = ax.get_ylim()
    ax.vlines(x, lims[0], lims[1], linewidth=1, color=stdColors['bleu'])
    # invert the plot order for spikes
    inv_colors = colors[::-1]
    inv_alphas = alphas[::-1]
    # individual spike
    cols = colsdict['indSpk']
    ax = spkaxes[0]
    for i, col in enumerate(cols[::-1]):
        ax.fill_between(data.index, data[col],
                        color=colors[::-1][i], alpha=0.5, label=col)
        ax.plot(data[col], color=inv_colors[i],
                alpha=1, label=col)  #, linewidth=1)
#    ax.plot(39.8, 0.1523, 'o', color= stdColors['bleu'])
    x = 39.8
    y = data.indiSpkCtr.loc[x]
    ax.plot(x, y, 'o', color=stdColors['bleu'])
    lims = ax.get_ylim()
    ax.vlines(x, lims[0], lims[1], linewidth=1, color=stdColors['bleu'])
    # plots pop (column 1-3)
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
        #errors : iterate on tuples
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
    ax.plot(x0, y, 'o', color=stdColors['bleu'])
    ax.plot(x1, y, '|', color=stdColors['bleu'])
    ax.hlines(y, x1, x0, color=stdColors['bleu'])
    ax.annotate("n=10", xy=(0.2, 0.8),
                xycoords="axes fraction", ha='center')
    # popVmNsig
    cols = colsdict['popVmNsig']
    ax = vmaxes[3]
    # traces
    for i, col in enumerate(cols[:2]):
        ax.plot(df[col], color=colors[i], alpha=alphas[i],
                label=col)
        #errors : iterate on tuples
        for i, col in enumerate(cols[2:]):
            if fill:
                ax.fill_between(df.index, df[col[0]], df[col[1]],
                                color=colors[i], alpha=0.2)#alphas[i]/2)
            else:
                for i, col in enumerate(cols[2:]):
                    for j in [0, 1]:
                        ax.plot(df[col[j]], color=colors[i], alpha=alphas[i],
                                label=col, linewidth=0.5)
    ax.annotate("n=27", xy=(0.2, 0.8),
                xycoords="axes fraction", ha='center')
    # pop spike
    cols = colsdict['popSpk']
    ax = spkaxes[1]
    for i, col in enumerate(cols[::-1]):
        ax.plot(df[col], color=inv_colors[i],
                alpha=1, label=col)#, linewidth=1)
        # ax.fill_between(df.index, df[col],
        #                 color=inv_colors[i], alpha=0.5, label=col)
    ax.annotate("n=20", xy=(0.2, 0.8),
                xycoords="axes fraction", ha='center')

# TODO define a Spk plotmode[lines, allhist, sdFill] for popSpkSig and popSpkNsig
    #popSpkSig
    cols = colsdict['popSpkSig']
    ax = spkaxes[2]
    # traces
    for i, col in enumerate(cols[:2][::-1]):
        # ax.fill_between(df.index, df[col], color=inv_colors[i],
        #                 alpha=inv_alphas[i]/2)
        ax.plot(df[col], color=inv_colors[i], alpha=inv_alphas[i],
                label=col, linewidth=2)
    # errors : iterate on tuples
    for i, col in enumerate(cols[2:]):
        ax.fill_between(df.index, df[col[0]], df[col[1]], color=colors[i],
                        alpha=inv_alphas[i]/2)# label=col, linewidth=0.5)
        # for j in [0, 1]:
        #     ax.plot(df[col[j]], color=colors[i],
        #             alpha=1, label=col, linewidth=0.5)
    # advance
    x0 = 0
    y = df.loc[x0][cols[0]]
    adf = df.loc[-20:0, [cols[1]]]
    i1 = (adf - y).abs().values.flatten().argsort()[0]
    x1 = adf.index[i1]
    ax.plot(x0, y, 'o', color=stdColors['bleu'])
    ax.plot(x1, y, '|', color=stdColors['bleu'])
    ax.hlines(y, x1, x0, color=stdColors['bleu'])
    ax.annotate("n=5", xy=(0.2, 0.8),
                xycoords="axes fraction", ha='center')
    # popSpkNsig
    cols = colsdict['popSpkNsig']
    ax = spkaxes[3]
    # traces
    for i, col in enumerate(cols[:2][::-1]):
        # ax.fill_between(df.index, df[col[0]], df[col[1]], color=inv_colors[i],
        #                 alpha=inv_alphas[i]/2)
        ax.plot(df[col], color=inv_colors[i], alpha=inv_alphas[i],
                label=col, linewidth=2)
    # errors : iterate on tuples
    for i, col in enumerate(cols[2:]):
        ax.fill_between(df.index, df[col[0]], df[col[1]], color=colors[i],
                        alpha=alphas[i]/2)#, label=col, linewidth=0.5)
        # for j in [0, 1]:
        #     ax.plot(df[col[j]], color=colors[i],
        #             alpha=1, label=col, linewidth=0.5)
    ax.annotate("n=15", xy=(0.2, 0.8),
                xycoords="axes fraction", ha='center')

    # labels
    for ax in axes:
        for spine in ['top', 'right']:
            ax.spines[spine].set_visible(False)
    ylabels = ['Membrane potential (mV)',
               'Normalized membrane potential',
               '', '']
    for i, ax in enumerate(vmaxes):
        ax.axes.get_xaxis().set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.set_ylabel(ylabels[i])
    ylabels = ['Firing rate (spikes/s)',
               'Normalized firing rate',
               '', '']
    for i, ax in enumerate(spkaxes):
        ax.set_ylabel(ylabels[i])
        ax.set_xlabel('time (ms)')

    for ax in vmaxes[1:]:
        ax.set_ylim(-0.10, 1.2)
    for ax in spkaxes[1:]:
        ax.set_ylim(-0.10, 1.3)
        ax.set_xlabel('relative time (ms)')

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
                         alpha=0.6, edgecolor='w', facecolor=stdColors['rouge'])
        ax.add_patch(rect)
        # center
    rect = Rectangle(xy=(0, -5), width=step, height=1, fill=True,
                     alpha=0.6, edgecolor='w', facecolor='k')
    ax.add_patch(rect)
    # fit individual example
    vmaxes[0].set_ylim(-4, 13)
    spkaxes[0].set_ylim(-5.5, 20)
    # align zero between plots  NB ref = first plot
    gf.align_yaxis(vmaxes[0], 0, vmaxes[1], 0)
    gf.align_yaxis(spkaxes[0], 0, spkaxes[1], 0)
    # adjust amplitude (without moving the zero)
    gf.change_plot_trace_amplitude(vmaxes[1], 0.85)
    gf.change_plot_trace_amplitude(spkaxes[1], 0.8)
    # zerolines
    for ax in axes:
        lims = ax.get_ylim()
        ax.vlines(0, lims[0], lims[1], alpha=0.2)
        lims = ax.get_xlim()
        ax.hlines(0, lims[0], lims[1], alpha=0.2)
        ax.yaxis.set_major_formatter(StrMethodFormatter('{x:,.1f}'))
    # rebuild tick to one decimal
    # individuals
    ax = vmaxes[0]
    custom_ticks = np.linspace(-2, 12, 8, dtype=int)
    ax.set_yticks(custom_ticks)
    ax.set_yticklabels(custom_ticks)
    ax = spkaxes[0]
    custom_ticks = np.linspace(0, 15, 4, dtype=int)
    ax.set_yticks(custom_ticks)
    ax.set_yticklabels(custom_ticks)
    # pop
    ax = vmaxes[1]
    custom_ticks = np.linspace(-0.2, 1, 7)
    ax.set_yticks(custom_ticks)
    ax = spkaxes[1]
    custom_ticks = np.linspace(0, 1, 6)
    ax.set_yticks(custom_ticks)

    fig.tight_layout()

    # remove the space between plots
    fig.subplots_adjust(hspace=0.06) #fig.subplots_adjust(hspace=0.02)
    if anot:
        date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        fig.text(0.99, 0.01, 'centrifigs.py:plot_figure2',
                 ha='right', va='bottom', alpha=0.4)
        fig.text(0.01, 0.01, date, ha='left', va='bottom', alpha=0.4)
    return fig

data_df, cols_dict = load2()
fig = plot_figure2(data_df, cols_dict)


#%%
plt.close('all')


def plot_half_figure2(data, colsdict):
    """
    plot_figure2 indiv + pop
    """
    colors = ['k', stdColors['rouge']]
    alphas = [0.8, 0.8]

    fig = plt.figure(figsize=(8.5, 8))
    # build axes with sharex and sharey
    axes = []
    for i in range(4):
        axes.append(fig.add_subplot(2, 2, i+1))
    # axes list
    vmaxes = axes[:2]      # vm axes = top row
    spkaxes = axes[2:]     # spikes axes = bottom row
    # plots individuals (first column)
    # individual vm
    cols = colsdict['indVm']
    ax = vmaxes[0]
    for i, col in enumerate(cols):
        ax.plot(data[col], color=colors[i], alpha=alphas[i],
                label=col)
    # individual spike
    cols = colsdict['indSpk']
    ax = spkaxes[0]
    for i, col in enumerate(cols[::-1]):
        ax.plot(data[col], color=colors[::-1][i],
                alpha=1, label=col, linewidth=1)
        ax.fill_between(data.index, data[col],
                        color=colors[::-1][i], alpha=0.5, label=col)
    # plots pop (column 1-3)
    df = data.loc[-30:35]       # limit xscale
    # pop vm
    cols = colsdict['popVm']
    ax = vmaxes[1]
    for i, col in enumerate(cols):
        ax.plot(df[col], color=colors[i], alpha=alphas[i],
                label=col)
    ax.annotate("n=37", xy=(0.2, 0.8),
                xycoords="axes fraction", ha='center')
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
    # labels
    for ax in axes:
        for spine in ['top', 'right']:
            ax.spines[spine].set_visible(False)
    ylabels = ['Membrane potential (mV)',
               'Normalized membrane potential',
               '', '']
    for i, ax in enumerate(vmaxes):
        ax.axes.get_xaxis().set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.set_ylabel(ylabels[i])
    ylabels = ['Firing rate (spikes/s)',
               'Normalized firing rate',
               '', '']
    for i, ax in enumerate(spkaxes):
        ax.set_ylabel(ylabels[i])
        ax.set_xlabel('Time (ms)')

    for ax in vmaxes[1:]:
        ax.set_ylim(-0.10, 1.1)
    for ax in spkaxes[1:]:
        ax.set_ylim(-0.10, 1.1)
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
                         alpha=0.6, edgecolor='w', facecolor=stdColors['rouge'])
        ax.add_patch(rect)
        # center
    rect = Rectangle(xy=(0, -5), width=step, height=1, fill=True,
                     alpha=0.6, edgecolor='w', facecolor='k')
    ax.add_patch(rect)
    # fit individual example
    vmaxes[0].set_ylim(-4, 13)
    spkaxes[0].set_ylim(-5.5, 20)
    # align zero between plots  NB ref = first plot
    gf.align_yaxis(vmaxes[0], 0, vmaxes[1], 0)
    gf.align_yaxis(spkaxes[0], 0, spkaxes[1], 0)
    # adjust amplitude (without moving the zero)
    gf.change_plot_trace_amplitude(vmaxes[1], 0.85)
    gf.change_plot_trace_amplitude(spkaxes[1], 0.8)
    # adjust ticks
    # individuals
    ax = vmaxes[0]
    custom_ticks = np.linspace(-2, 12, 8, dtype=int)
    ax.set_yticks(custom_ticks)
    ax.set_yticklabels(custom_ticks)
    ax = spkaxes[0]
    custom_ticks = np.linspace(0, 15, 4, dtype=int)
    ax.set_yticks(custom_ticks)
    ax.set_yticklabels(custom_ticks)
    # pop
    ax = vmaxes[1]
    custom_ticks = np.arange(-0.2, 1.2, 0.2)
    ax.set_yticks(custom_ticks)
    ax = spkaxes[1]
    custom_ticks = np.arange(0, 1.1, 0.2)
    ax.set_yticks(custom_ticks)
    # zerolines
    for ax in axes:
        lims = ax.get_ylim()
        ax.vlines(0, lims[0], lims[1], alpha=0.2)
        lims = ax.get_xlim()
        ax.hlines(0, lims[0], lims[1], alpha=0.2)
    fig.tight_layout()
    # remove the space between plots
    fig.subplots_adjust(hspace=0.06) #fig.subplots_adjust(hspace=0.02)

    if anot:
        date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        fig.text(0.99, 0.01, 'centrifigs.py:plot_half_figure2',
                 ha='right', va='bottom', alpha=0.4)
        fig.text(0.01, 0.01, date, ha='left', va='bottom', alpha=0.4)

    return fig

fig = plot_half_figure2(data_df, cols_dict)

#%%
plt.close('all')

#>> that's the figure2

def plot_3quarter_figure2(data, colsdict, fill=True):
    """
    figure2 (individual + pop + sig)
    """
    colors = ['k', stdColors['rouge']]
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
    ax.plot(x, y, 'o', color=stdColors['bleu'])
    lims = ax.get_ylim()
    ax.vlines(x, lims[0], lims[1], linewidth=1, color=stdColors['bleu'],
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
    ax.plot(x, y, 'o', color=stdColors['bleu'])
    lims = ax.get_ylim()
    ax.vlines(x, lims[0], lims[1], linewidth=1, color=stdColors['bleu'],
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
    ax.plot(x0, y, 'o', color=stdColors['bleu'])
    ax.plot(x1, y, marker=markers.CARETLEFT, color=stdColors['bleu'])
    ax.hlines(y, x1, x0, color=stdColors['bleu'], linestyle=':')

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
    ax.plot(x0, y, 'o', color=stdColors['bleu'])
    ax.plot(x1, y, marker=markers.CARETLEFT, color=stdColors['bleu'])
    ax.hlines(y, x1, x0, color=stdColors['bleu'], linestyle=':')

    ax.annotate("n=5", xy=(0.2, 0.8),
                xycoords="axes fraction", ha='center')
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
                         alpha=0.6, edgecolor='w', facecolor=stdColors['rouge'])
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
        gf.align_yaxis(vmaxes[i], 0, vmaxes[i+1], 0)
        gf.align_yaxis(spkaxes[i], 0, spkaxes[i+1], 0)
    # adjust amplitude (without moving the zero)
    for i in [1, 2]:
        gf.change_plot_trace_amplitude(vmaxes[i], 0.85)
        gf.change_plot_trace_amplitude(spkaxes[i], 0.8)
    # zerolines
    for ax in axes:
        lims = ax.get_ylim()
        ax.vlines(0, lims[0], lims[1], alpha=0.2)
        lims = ax.get_xlim()
        ax.hlines(0, lims[0], lims[1], alpha=0.2)
    # # adjust ticks
    # ax = vmaxes[0]
    # custom_ticks = np.linspace(-2, 10, 7, dtype=int)
    # ax.set_yticks(custom_ticks)
    # for ax in vmaxes[1:]:
    #     custom_ticks = np.arange(-0.2, 1.2, 0.2)
    #     ax.set_yticks(custom_ticks)
    # for ax in spkaxes[1:]:
    #     custom_ticks = np.arange(-0.3, 1.3, 0.3)
    #     ax.set_yticks(custom_ticks)
    # rebuild tick to one decimal
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
        fig.text(0.99, 0.01, 'centrifigs.py:plot_3quarter_figure2',
                 ha='right', va='bottom', alpha=0.4)
        fig.text(0.01, 0.01, date, ha='left', va='bottom', alpha=0.4)

    return fig

fig = plot_3quarter_figure2(data_df, cols_dict)
#%% sigNonsig


def plot_signonsig_figure2(data, colsdict, fill=True, fillground=True):
    """
    plot_figure2
    """
    colors = ['k', stdColors['rouge']]
    alphas = [0.8, 0.8]
    # no individual : focus on initial response
    df = data.loc[-30:35]

    fig = plt.figure(figsize=(8.5, 8)) #fig = plt.figure(figsize=(8, 8))
    # build axes with sharex and sharey
    axes = []
    for i in range(4):
        axes.append(fig.add_subplot(2, 2, i+1))
    # axes list
    vmaxes = axes[:2]      # vm axes = top row
    spkaxes = axes[2:]     # spikes axes = bottom row
    # ____ plots individuals (first column)
#    # individual vm
#    cols = colsdict['indVm']
#    ax = vmaxes[0]
#    for i, col in enumerate(cols):
#        ax.plot(data[col], color=colors[i], alpha=alphas[i],
#                label=col)
#    #individual spike
#    cols = colsdict['indSpk']
#    ax = spkaxes[0]
#    for i, col in enumerate(cols[::-1]):
#        ax.plot(data[col], color=colors[::-1][i],
#                alpha=1, label=col, linewidth=1)
#        ax.fill_between(data.index, data[col],
#                        color=colors[::-1][i], alpha=0.5, label=col)
#    #____ plots pop (column 1-3)
#   df = data.loc[-30:35]       # limit xscale
#    # pop vm
#    cols = colsdict['popVm']
#    ax = vmaxes[1]
#    for i, col in enumerate(cols):
#        ax.plot(df[col], color=colors[i], alpha=alphas[i],
#                label=col)
#    ax.annotate("n=37", xy=(0.2, 0.8),
#                xycoords="axes fraction", ha='center')
    # popVmSig
    cols = colsdict['popVmSig']
    ax = vmaxes[-2]
    ax.set_title('significative population')
    # traces
    for i, col in enumerate(cols[:2]):
        ax.plot(df[col], color=colors[i], alpha=alphas[i],
                label=col)
        #errors : iterate on tuples
    for i, col in enumerate(cols[2:]):
        if fill:
            ax.fill_between(df.index, df[col[0]], df[col[1]], color=colors[i],
                            alpha=alphas[i]/2)
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
    ax.plot(x0, y, 'o', color=stdColors['bleu'])
    ax.plot(x1, y, '|', color=stdColors['bleu'])
    ax.hlines(y, x1, x0, color=stdColors['bleu'])
    ax.annotate("n=10", xy=(0.2, 0.8),
                xycoords="axes fraction", ha='center')
    # popVmNsig
    cols = colsdict['popVmNsig']
    ax = vmaxes[-1]
    ax.set_title('non significative population')
    # traces
    for i, col in enumerate(cols[:2]):
        ax.plot(df[col], color=colors[i], alpha=alphas[i],
                label=col)
    # errors : iterate on tuples
    for i, col in enumerate(cols[2:]):
        if fill:
            ax.fill_between(df.index, df[col[0]], df[col[1]], color=colors[i],
                            alpha=alphas[i]/2)
        else:
            for i, col in enumerate(cols[2:]):
                for j in [0, 1]:
                    ax.plot(df[col[j]], color=colors[i], alpha=alphas[i],
                            label=col, linewidth=0.5)
    ax.annotate("n=27", xy=(0.2, 0.8),
                xycoords="axes fraction", ha='center')
    # NB for spiking response, the control should be plotted last,
    # ie revert the order
    inv_colors = colors[::-1]
    inv_alphas = alphas[::-1]
    # popSpkSig
    cols = colsdict['popSpkSig']
    ax = spkaxes[-2]
    # traces
    for i, col in enumerate(cols[:2][::-1]):
        # ax.fill_between(df.index, df[col], color=inv_colors[i],
        #                 alpha=inv_alphas[i]/2)
        ax.plot(df[col], color=inv_colors[i], alpha=inv_alphas[i],
                label=col, linewidth=2)
    # errors : iterate on tuples
    for i, col in enumerate(cols[2:]):
        ax.fill_between(df.index, df[col[0]], df[col[1]], color=colors[i],
                        alpha=alphas[i]/2, label=col)
    # advance
    x0 = 0
    y = df.loc[x0][cols[0]]
    adf = df.loc[-20:0, [cols[1]]]
    i1 = (adf - y).abs().values.flatten().argsort()[0]
    x1 = adf.index[i1]
    ax.plot(x0, y, 'o', color=stdColors['bleu'])
    ax.plot(x1, y, '|', color=stdColors['bleu'])
    ax.hlines(y, x1, x0, color=stdColors['bleu'])
    ax.annotate("n=5", xy=(0.2, 0.8),
                xycoords="axes fraction", ha='center')
    # popSpkNsig
    cols = colsdict['popSpkNsig']
    ax = spkaxes[-1]
    # traces
    for i, col in enumerate(cols[:2][::-1]):
        ax.plot(df[col], color=inv_colors[i], alpha=inv_alphas[i],
                label=col, linewidth=2)
        # ax.fill_between(df.index, df[col], color=inv_colors[i],
        #                 alpha=inv_alphas[i]/2)
    # errors : iterate on tuples
    for i, col in enumerate(cols[2:]):
        ax.fill_between(df.index, df[col[0]], df[col[1]], color=colors[i],
                        alpha=alphas[i]/2)
    ax.annotate("n=15", xy=(0.2, 0.8),
                xycoords="axes fraction", ha='center')
    # labels
    for ax in axes:
        for spine in ['top', 'right']:
            ax.spines[spine].set_visible(False)
    ylabels = ['membrane potential (mV)',
               'normalized membrane potential',
               '', '']
    ylabels = ylabels[1:]   # no individual exemple
    for i, ax in enumerate(vmaxes):
        ax.axes.get_xaxis().set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.set_ylabel(ylabels[i])
    ylabels = ['firing rate (spikes/s)',
               'normalized firing rate',
               '', '']
    ylabels = ylabels[1:]   # no individual exemple
    for i, ax in enumerate(spkaxes):
        ax.set_ylabel(ylabels[i])
        ax.set_xlabel('time (ms)')

    for ax in vmaxes:
        ax.set_ylim(-0.10, 1.2)
    for ax in spkaxes:
        ax.set_ylim(-0.10, 1.3)
        ax.set_xlabel('relative time (ms)')
    # zerolines
    for ax in axes:
        lims = ax.get_ylim()
        ax.vlines(0, lims[0], lims[1], alpha=0.2)
        lims = ax.get_xlim()
        ax.hlines(0, lims[0], lims[1], alpha=0.2)
    fig.tight_layout()
    # remove the space between plots
    fig.subplots_adjust(hspace=0.06) #fig.subplots_adjust(hspace=0.02)

    if anot:
        date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        fig.text(0.99, 0.01, 'centrifigs.py:plot_signonsig_figure2',
                 ha='right', va='bottom', alpha=0.4)
        fig.text(0.01, 0.01, date, ha='left', va='bottom', alpha=0.4)

    return fig

plot_signonsig_figure2(data_df, cols_dict)

#%% NB the length of the sorted data are not the same compared to the other traces
#filename = 'fig2cells.xlsx'
#df = pd.read_excel(filename)
plt.close('all')

def plot_figure2B(sig=True):
    """
    plot_figure2B : sorted phase advance and delta response
    sig=boolan : true <-> shown cell signification
    """
    filename = 'data/fig2cells.xlsx'
    df = pd.read_excel(filename)
    cols = df.columns[:2]
    signs = df.columns[2:]
    df.index += 1 # cells = 1 to 37
    color_dic = {0 :'w', 1 : stdColors['rouge']}

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(17.6, 4))
    for i, ax in enumerate(axes):
        colors = [color_dic[x] for x in df[signs[i]]]
        if sig:
            axes[i].bar(df.index, df[cols[i]], edgecolor=stdColors['rouge'],
                        color=colors, label=cols[i], alpha=0.8, width=0.8)
        else:
            axes[i].bar(df.index, df[cols[i]], edgecolor=stdColors['rouge'],
                        color=stdColors['rouge'], label=cols[i],
                        alpha=0.8, width=0.8)
        # zero line
        lims = ax.get_xlim()
        ax.hlines(0, lims[0], lims[1], alpha=0.2)
        # ticks
        ax.set_xlim(0, 38)
        ax.set_xticks([df.index.min(), df.index.max()])
        ax.set_xlabel('Cell rank')
        ax.xaxis.set_label_coords(0.5, -0.025)
        if i == 0:
            txt = r'$\Delta$ Phase (ms)'
            ylims = (-6, 29)
            ax.vlines(0, 0, 20, linewidth=2)
            custom_yticks = np.linspace(0, 20, 3, dtype=int)
        else:
            txt = r'$\Delta$ Amplitude'
            ylims = ax.get_ylim()
            ax.vlines(0, 0, 0.6, linewidth=2)
            custom_yticks = np.linspace(0, 0.6, 4)
        ax.set_yticks(custom_yticks)
        ax.set_ylabel(txt)
        ax.set_ylim(ylims)
        for spine in ['left', 'top', 'right', 'bottom']:
            ax.spines[spine].set_visible(False)
    # align zero between plots
    gf.align_yaxis(axes[0], 0, axes[1], 0)
    gf.change_plot_trace_amplitude(axes[1], 0.8)
    fig.tight_layout()
    # anot
    if anot:
        date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        fig.text(0.99, 0.01, 'centrifigs.py:plot_figure2B',
                 ha='right', va='bottom', alpha=0.4)
        fig.text(0.01, 0.01, date, ha='left', va='bottom', alpha=0.4)
    return fig

def plot_2B_bis():
    """
    plot_figure2B alternative : sorted phase advance and delta response
    response are sorted only by phase
    """
    df = ld.load_cell_contributions('vm')
    alist = [item for item in df.columns if 'vm_s_cp_iso_' in item]
    df = df[alist].sort_values(by=alist[0], ascending=False)
    cols = df.columns[::2]
    sigs = df.columns[1::2]
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(17.6, 4))
    color_dic = {0 :'w', 1 : stdColors['rouge']}
    for i, ax in enumerate(axes):
        colors = [color_dic[x] for x in df[sigs[i]]]
        axes[i].bar(df.index, df[cols[i]], edgecolor=stdColors['rouge'],
                    color=colors, label=cols[i], alpha=0.8, width=0.8)
        lims = ax.get_xlim()
        ax.hlines(0, lims[0], lims[1], alpha=0.3)
        ax.set_xticks([0, len(df)-1])
        ax.set_xticklabels([1, len(df)])
        ax.set_xlim(-1, len(df)+0.5)
        if i == 0:
            ax.vlines(-1, 0, 20, linewidth=2)
            custom_yticks = np.linspace(0, 20, 3, dtype=int)
            ylabel = r'$\Delta$ Phase (ms)'
            ax.set_ylim(-6, 29)
            x_label = 'Cell rank'
        else:
            ax.vlines(-1, 0, 0.6, linewidth=2)
            custom_yticks = np.linspace(0, 0.6, 4)
            x_label = 'Ranked cells'
            ylabel = r'$\Delta$ Amplitude'
        ax.set_xlabel(x_label)
        ax.xaxis.set_label_coords(0.5, -0.025)
        ax.set_yticks(custom_yticks)
        ax.set_ylabel(ylabel)
        for spine in ['bottom', 'left', 'top', 'right']:
            ax.spines[spine].set_visible(False)

    gf.align_yaxis(axes[0], 0, axes[1], 0)
    gf.change_plot_trace_amplitude(axes[1], 0.8)
    fig.tight_layout()

    if anot:
        date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        fig.text(0.99, 0.01, 'centrifigs.py:plot_figure2B_bis',
                 ha='right', va='bottom', alpha=0.4)
        fig.text(0.01, 0.01, date, ha='left', va='bottom', alpha=0.4)
    return fig


def sort_stat():
    filename = 'data/fig2cells.xlsx'
    df = pd.read_excel(filename)
    cols = df.columns[:2]
    signs = df.columns[2:]
    df.index += 1 # cells = 1 to 37

    temp1 = df.loc[df.lagIndiSig == 1, ['popVmscpIsolatg']]
    temp2 = df.loc[df.ampIndiSig == 1, ['popVmscpIsoAmpg']]
    for item, temp in zip(['latency', 'gain'], [temp1, temp2]):
        print(item, len(temp), 'measures')
        print('mean= {:5.2f}'.format(temp.mean()[0]))
        print('std= {:5.2f}'.format(temp.std()[0]))
        print('sem= {:5.2f}'.format(temp.sem()[0]))

fig = plot_2B_bis()
plot_figure2B('horizontal')
#plot_figure2B('vertical')
sort_stat()

#%%
# plt.close('all')


def plot_figure3(kind='sig', substract=False):
    """
    plot_figure3
    input : kind in ['pop': whole population, 'sig': individually significants
    cells, 'nonsig': non significant cells]
    """
    filenames = {'pop' : 'data/fig3.xlsx',
                 'sig': 'data/fig3bis1.xlsx',
                 'nonsig': 'data/fig3bis2.xlsx'}
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
    colors = ['k', stdColors['rouge'], stdColors['vert'],
              stdColors['jaune'], stdColors['bleu']]
    # alphas = [0.5, 0.8, 0.5, 1, 0.6]
    alphas = [0.8, 0.8, 0.8, 0.8, 0.8]

    if substract:
        ref = df['CENTER-ONLY']
        df = df.subtract(ref, axis=0)

    fig = plt.figure(figsize=(6.5, 5.5))
    #  #SUGGESTION: make y dimension much larger to see maximize visual difference between traces
    # fig.suptitle(titles[kind])
    ax = fig.add_subplot(111)
    for i, col in enumerate(cols):
        #if (i == 0) or (i == 4):
            #ax.plot(df[col], color=colors[i], alpha=alphas[i], label=col, linewidth=2)
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
        ax.vlines(21.4, lims[0], lims[1])
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

fig = plot_figure3('pop')
fig = plot_figure3('sig')
#fig = plot_figure3('nonsig')
fig = plot_figure3('sig', substract=True)
fig2 = plot_figure3('pop', substract=True)

#pop all cells
#%% grouped sig and non sig
plt.close('all')


def plot_figure3_signonsig(substract=False):
    """
    plot_figure3
    with individually significants and non significant cells
    """
    filenames = {'pop' : 'data/fig3.xlsx',
                 'sig': 'data/fig3bis1.xlsx',
                 'nonsig': 'data/fig3bis2.xlsx'}
    titles = {'pop' : 'recorded cells',
              'sig': 'individually significant cells',
              'nonsig': 'individually non significants cells'}
    # samplesize
    cellnumbers = {'pop' : 37, 'sig': 10, 'nonsig': 27}
    colors = ['k', stdColors['rouge'], stdColors['vert'],
              stdColors['jaune'], stdColors['bleu']]
    alphas = [0.6, 0.8, 0.5, 1, 0.6]

    fig = plt.figure(figsize=(11.6, 6))
    axes = []
    for i in range(2):
        axes.append(fig.add_subplot(1, 2, i+1))
    for i, kind in enumerate(['sig', 'nonsig']):
        ncells = cellnumbers[kind]
        df = pd.read_excel(filenames[kind])
        # centering
        middle = (df.index.max() - df.index.min())/2
        df.index = (df.index - middle)/10
        df = df.loc[-45:120]
        cols = ['CNT-ONLY', 'CP-ISO', 'CF-ISO', 'CP-CROSS', 'RND-ISO']
        df.columns = cols
        if substract:
            ref = df['CNT-ONLY']
            df = df.subtract(ref, axis=0)
        ax = axes[i]
        ax.set_title(titles[kind])
        ncells = cellnumbers[kind]
        for j, col in enumerate(cols):
            ax.plot(df[col], color=colors[j], alpha=alphas[j],
                    label=col, linewidth=2)
            ax.annotate('n=' + str(ncells), xy=(0.1, 0.8),
                        xycoords="axes fraction", ha='center')
        if substract:
            leg = ax.legend(loc='upper right', markerscale=None, frameon=False,
                            handlelength=0)
        else:
            leg = ax.legend(loc='lower right', markerscale=None, frameon=False,
                            handlelength=0)
        for line, text in zip(leg.get_lines(), leg.get_texts()):
            text.set_color(line.get_color())
        # bleu point
        ax.plot(0, df.loc[0]['CNT-ONLY'], 'o', color=stdColors['bleu'])

    axes[0].set_ylabel('normalized membrane potential')
    for ax in axes:
        ax.set_xlim(-15, 30)
        ax.set_ylim(-0.1, 1.1)
        lims = ax.get_ylim()
        ax.vlines(0, lims[0], lims[1], alpha=0.2)
        lims = ax.get_xlim()
        ax.hlines(0, lims[0], lims[1], alpha=0.2)
        ax.set_xlabel('relative time (ms)')
        for loc in ['top', 'right']:
            ax.spines[loc].set_visible(False)

    if substract:
        for ax in axes:
            ax.set_xlim(-45, 120)
            ax.set_ylim(-0.15, 0.4)
            custom_ticks = np.arange(-40, 110, 20)
            ax.set_xticks(custom_ticks)
            # max center only
            lims = ax.get_ylim()
            ax.vlines(21.4, lims[0], lims[1])
            # end of center only
            #(df['CENTER-ONLY'] - 0.109773).abs().sort_values().head()
            ax.vlines(88, lims[0], lims[1], alpha=0.3)
            ax.axvspan(0, 88, facecolor='k', alpha=0.1)
            ax.text(0.41, 0.6, 'center only response \n start | peak | end',
                    transform=ax.transAxes, alpha=0.5)
        axes[0].set_ylabel('Norm vm - Norm centerOnly')
        # axes[1].yaxis.set_visible(False)
        # axes[1].spines['left'].set_visible(False)

    fig.tight_layout()

    if anot:
        date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        fig.text(0.99, 0.01, 'centrifigs.py:plot_figure3_signonsig',
                 ha='right', va='bottom', alpha=0.4)
        fig.text(0.01, 0.01, date, ha='left', va='bottom', alpha=0.4)

    return fig

plot_figure3_signonsig()
fig = plot_figure3_signonsig(substract=True)
#%%
plt.close('all')

def plot_figure4(substract=False):
    """ speed """
    filename = 'data/fig4.xlsx'
    df = pd.read_excel(filename)
    # centering
    middle = (df.index.max() - df.index.min())/2
    # OBSERVATION bottom raw 0 baseline has been decentered by police and ticks size changes
    df.index = df.index - middle
    df.index = df.index/10
    cols = ['centerOnly', '100%', '70%', '50%', '30%']
    df.columns = cols
    colors = ['k', stdColors['rouge'], speedColors['orangeFonce'],
              speedColors['orange'], speedColors['jaune']]
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
    ax.plot(0, df.loc[0]['centerOnly'], 'o', color=stdColors['bleu'])
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
    filenames = ['data/figSup7a.xlsx', 'data/figSup5bis.xlsx']#'data/figSup7b.xlsx']
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
    colors = [stdColors['rouge'], stdColors['jaune']]
    darkcolors = [stdColors['dark_rouge'], stdColors['dark_jaune']]
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

def plot_figure6():
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
    colors = ['k', stdColors['rouge'], stdColors['vertSombre'], stdColors['vertSombre']]
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
        ax.plot(x, y, 'o', color=stdColors['bleu'])
        ax.vlines(x, -0.5, lims[1], color=stdColors['bleu'],
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

fig = plot_figure6()

#%%
plt.close('all')

def align_center(adf, showPlot=False):
    df = adf.copy()
    ref = df['center_only'].copy()        
    cp = df.surround_then_center.copy()  
    ref50_y = (ref.loc[30:80].max() - ref.loc[30:80].min()) / 2
    ref50_x = (ref.loc[30:80] - ref50_y).abs().sort_values().index[0]
    cp50_y = ref50_y
    cp50_x = ((cp.loc[30:70] - cp50_y)).abs().sort_values().index[0]
    
    if showPlot:
        fig =  plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(ref, '-k', alpha = 0.5, label='ref')
        ax.plot(cp, '-r', alpha = 0.6, label='cp')
        ax.set_xlim(0, 100)
        lims = ax.get_xlim()
        ax.hlines(0, lims[0], lims[1], alpha=0.3)
        lims = ax.get_ylim()
        ax.vlines(0, lims[0], lims[1], alpha=0.2)   
        limx = ax.get_xlim()
        limy = ax.get_ylim()
        ax.hlines(ref50_y, limx[0], limx[1], alpha=0.3)
        ax.vlines(ref50_x, limy[0], limy[1], alpha=0.3)
        ax.vlines(cp50_x, limy[0], limy[1], 'r', alpha=0.4)
        adv = cp50_x - ref50_x
        print('adv=', adv)
        ax.plot(ref.shift(int(10*adv)), ':k', alpha=0.5, label='ref_corr',
                linewidth=2)
        ref_corr = ref.shift(int(10*adv))
    
        ax.plot(cp - ref, '-b', alpha=0.5, label='cp - ref')
        ax.plot(cp - ref_corr, ':b', alpha=0.5, label='cp - ref_corr',
                linewidth=2)
        ax.legend()
        fig.tight_layout()
        for spine in ['top', 'right']:
            ax.spines[spine].set_visible(False)
    return ref_corr    


def plot_figure6_bis(linear=True, substract=False):
    """
    plot_figure6 minus center
    """
    filename = 'data/fig5.xlsx'
    df = pd.read_excel(filename)
    # centering
    middle = (df.index.max() - df.index.min())/2
    df.index = df.index - middle
    df.index = df.index/10
    # rename columns
    cols = df.columns
    cols = ['center_only', 'surround_then_center', 'surround_only',
            'static_linear_prediction']
    dico = dict(zip(df.columns, cols))
    df.rename(columns=dico, inplace=True)
    # color parameters
    colors = ['k', stdColors['rouge'], stdColors['vertSombre'], stdColors['vertSombre']]
    alphas = [0.6, 0.8, 0.8, 0.8]
    # substract
    # build a time shifted reference (centerOnly) to perfome the substraction
    ref_shift = align_center(df, showPlot=True)
    df['surround_then_center'] = df['surround_then_center'] - ref_shift
   # df['Center_Only'] -= ref
    # plotting
    fig = plt.figure(figsize=(8.5, 4))
#    fig.suptitle(os.path.basename(filename))
    ax = fig.add_subplot(111)
    for i, col in enumerate(cols): # omit centrerOnly
        if i == 0:
            pass
        # suround the center minus center
        elif i == 1:
            if substract:
                ax.plot(df.loc[-120:200, [col]], color=colors[i], alpha=alphas[i],
                        label=col, linestyle='--', linewidth=1.5)
        # linear predictor
        elif i == 3:
            if linear:
                ax.plot(df.loc[-120:200, [col]], color=colors[i], alpha=alphas[i],
                        label=col, linestyle='--', linewidth=1.5)
        else:
            ax.plot(df.loc[-120:200, [col]], color=colors[i], alpha=1,
                    label=col)
    ax.set_xlabel('Time (ms)')
    # stims
    step = 21
    hlocs = np.arange(0, -110, -step)
    names = ['D0', 'D1', 'D2', 'D3', 'D4', 'D5']
    # vlocs = np.linspace(-0.7, -1.7, 4)
    # vlocs = np.linspace(-1.4, -2.4, 4)
    vlocs = np.linspace(-0.8, -1.5, 4)
    dico = dict(zip(names, hlocs))

    # ax
    for key in dico.keys():
        # names
        ax.annotate(key, xy=(dico[key]+6, vlocs[0]), alpha=0.6,
                    annotation_clip=False, fontsize='small')
        #stim1
        rect = Rectangle(xy=(dico[key], vlocs[1]), width=step, height=0.2,
                         fill=True, alpha=1, edgecolor='w',
                         facecolor=colors[2])
        if key == 'D0':
            rect = Rectangle(xy=(dico[key], vlocs[1]), width=step, height=0.2,
                             fill=True, alpha=1, edgecolor=colors[2],
                             facecolor='w')
        ax.add_patch(rect)
        # #stim2
        # rect = Rectangle(xy=(dico[key], vlocs[2]), width=step, height=0.3,
        #                  fill=True, alpha=0.6, edgecolor='w',
        #                  facecolor=colors[1])
        # if key == 'D0':
        #     rect = Rectangle(xy=(dico[key], vlocs[2]), width=step, height=0.3,
        #                  fill=True, alpha=0.6, edgecolor='k',
        #                  facecolor='y')
        # ax.add_patch(rect)
    # #center
    # rect = Rectangle(xy=(0, vlocs[3]), width=step, height=0.3, fill=True,
    #                  alpha=0.6, edgecolor='w', facecolor=colors[0])
    # ax.add_patch(rect)
#    for i, st in enumerate(['Surround-Only', 'Surround-then-Center minus Center']):
    for i, st in enumerate(['surround-only']):
        ax.annotate(st, xy=(30, vlocs[i+1]), color=colors[2-i],
                    annotation_clip=False, fontsize='small')

    ax.set_ylabel('Membrane potential (mV)')
    for loc in ['top', 'right']:
        ax.spines[loc].set_visible(False)
    lims = ax.get_xlim()
    ax.hlines(0, lims[0], lims[1], alpha=0.2)
    lims = ax.get_ylim()
    ax.vlines(0, lims[0], lims[1], alpha=0.2)
    # response start
    x = 41
    y = df['center_only'].loc[x]
    ax.plot(x, y, 'o', color=stdColors['bleu'])
    ax.vlines(x, -0.5, lims[1], color=stdColors['bleu'],
              linestyle=':', alpha=0.8)
    for dloc in hlocs:
        ax.vlines(dloc, lims[0], lims[1], linestyle=':', alpha=0.2)
    # end
    x = 150.1
    ax.vlines(x, -0.5, lims[1], color=stdColors['bleu'],
              linestyle=':', alpha=0.8)
    # peak
    x = 63.9
    ax.vlines(x, -0.5, lims[1], 'k', alpha=0.5)
    ax.axvspan(41, 150.1, ymin=0.28, color='k', alpha=0.1)
    # ticks
    custom_ticks = np.linspace(0, 1, 2, dtype=int)
    ax.set_yticks(custom_ticks)
    ax.set_yticklabels(custom_ticks)

    fig.tight_layout()

    ax.text(0.55, 0.17, 'center only response \n start | peak | end',
            transform=ax.transAxes, alpha=0.5)
    if anot:
        date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        fig.text(0.99, 0.01, 'centrifigs.py:plot_figure6_bis',
                 ha='right', va='bottom', alpha=0.4)
        fig.text(0.01, 0.01, date, ha='left', va='bottom', alpha=0.4)

    return fig

fig = plot_figure6_bis()
# fig = plot_figure6_bis(substract=True)
fig = plot_figure6_bis(linear=False, substract=True)


#%%
plt.close('all')


def plot_figure7():
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
    colors = ['k', stdColors['rouge'], stdColors['vertSombre'],
              stdColors['bleuViolet'], stdColors['bleuViolet'],
              stdColors['bleuViolet']]
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
    ax1.plot(x, y, 'o', color=stdColors['bleu'])
    # ax1.hlines(y, -150, 10, colors=stdColors['bleu'], alpha=0.5)
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
    gf.align_yaxis(ax1, 0, ax2, 0)
    gf.change_plot_trace_amplitude(ax2, 0.9)
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

fig = plot_figure7()

#%%
plt.close('all')

def plot_figure7_bis():
    """
    plot_figure7 minus center
    """
    filename = 'data/fig6.xlsx'
    df = pd.read_excel(filename)
    # centering
    middle = (df.index.max() - df.index.min())/2
    df.index = df.index - middle
    df.index = df.index/10
    # limit the date time range
    df = df.loc[-150:160]
    cols = ['centerOnly', 'surroundThenCenter', 'surroundOnly'
            'sdUp', 'sdDown', 'linearPrediction']
    dico = dict(zip(df.columns, cols))
    df.rename(columns=dico, inplace=True)
    cols = df.columns
    # if substract:
    #     ref = df.centerOnly
    #     df.surroundThenCenter = df.surroundThenCenter - ref
    #     df.centerOnly = df.centerOnly - ref
    colors = ['k', 'r', 'b', 'g', 'b', 'b']
    colors = ['k', stdColors['rouge'], stdColors['vertSombre'],
              stdColors['bleuViolet'], stdColors['bleuViolet'],
              stdColors['bleuViolet']]
    colors = ['k', stdColors['rouge'], stdColors['vertSombre'],
              stdColors['bleuViolet'], stdColors['rouge'],
              stdColors['bleuViolet']]
    alphas = [0.5, 0.7, 0.7, 0.6, 0.6, 0.6]

    # plotting
    fig = plt.figure(figsize=(8.5, 4))
    ax = fig.add_subplot(111)
    ax.plot(df.popfillVmscpIsoDlp, ':r', alpha=1, linewidth=2,
            label='sThenCent - cent')
    # surroundOnly
    ax.plot(df.surroundOnlysdUp, color=stdColors['vertSombre'], alpha=0.7,
            label='surroundOnly')
    ax.fill_between(df.index, df[df.columns[3]], df[df.columns[4]],
                    color=stdColors['vertSombre'], alpha=0.2)

    ax.set_ylabel('Normalized membrane potential')
    ax.annotate("n=12", xy=(0.1, 0.8),
                xycoords="axes fraction", ha='center')
    ax.set_xlabel('Relative time (ms)')
    for loc in ['top', 'right']:
        ax.spines[loc].set_visible(False)
    lims = ax.get_xlim()
    ax.hlines(0, lims[0], lims[1], alpha=0.3)
    lims = ax.get_ylim()
    ax.vlines(0, lims[0], lims[1], alpha=0.3)
    # response start
    x0 = 0
    y = df['centerOnly'].loc[x0]
    ax.plot(x0, y, 'o', color=stdColors['bleu'])
    ax.vlines(x0, lims[0], lims[1], color=stdColors['bleu'],
              linestyle=':', alpha=0.8)
    # end
    x2 = 124.6
    y = df['centerOnly'].loc[x2]
    ax.vlines(x2, lims[0], lims[1], color='k',
              linestyle='-', alpha=0.2)
    # ax.plot(x2, y, 'o', color=stdColors['bleu'])
    # peak
    # df.centerOnly.idxmax()
    x1 = 26.1
    ax.vlines(x1, 0, lims[1], 'k', alpha=0.3)
    ax.axvspan(x0, x2, color='k', alpha=0.1)
    #ticks
    custom_ticks = np.linspace(0, 0.4, 3)
    ax.set_yticks(custom_ticks)
    ax.set_yticklabels(custom_ticks)

    fig.tight_layout()

    ax.text(0.5, 0.10, 'center only response \n start | peak | end',
            transform=ax.transAxes, alpha=0.5)
    if anot:
        date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        fig.text(0.99, 0.01, 'centrifigs.py:plot_figure7_bis',
                 ha='right', va='bottom', alpha=0.4)
        fig.text(0.01, 0.01, date, ha='left', va='bottom', alpha=0.4)

    return fig

fig = plot_figure7_bis()


#%% fig 9
plt.close('all')

def plot_figure9CD(data, colsdict):
    """
    plot_figure9CD
    """
    colors = ['k', stdColors['rouge']]
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
    # ax.plot(x0, y, 'o', color=stdColors['bleu'])
    # ax.plot(x1, y, '|', color=stdColors['bleu'])
    # ax.hlines(y, x1, x0, color=stdColors['bleu'])
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
             color=[stdColors['rouge'], 'None'], edgecolor='k', linewidth=1)
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

# data_df, cols_dict = load2()
plot_figure9CD(data_df, cols_dict)

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
    colors = [stdColors['rouge'], stdColors['rouge'],
              stdColors['vert'], stdColors['vert'],
              stdColors['jaune'], stdColors['jaune'],
              stdColors['bleu'], stdColors['bleu'],
              stdColors['bleu'], stdColors['bleu']]
    # data (call)
    df = ld.load_cell_contributions('vm')
    # extract list of traces : sector vs full
    traces = [item for item in df.columns if 's_' in item[:7]]
    # append full random
    f_rnd = [item for item in df.columns if 'vm_f_rnd' in item]
    for item in f_rnd:
        traces.append(item)
    # filter -> only significative cells
    traces = [item for item in traces if 'indisig' not in item]
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
    sig_name = name + '_indisig'
    df = df.sort_values(by=[name, sig_name], ascending=False)
    # plot all traces
    for i, name in enumerate(traces):
        sig_name = name + '_indisig'
        # color : white if non significant, edgecolor otherwise
        edgeColor = colors[i]
        color_dic = {0 : 'w', 1 : edgeColor}
        if sort_all:
            select = df[[name, sig_name]].sort_values(by=[name, sig_name],
                                                      ascending=False)
        else:
            select = df[[name, sig_name]]
        barColors = [color_dic[x] for x in select[sig_name]]
        ax = axes[i]
        # ax.set_title(str(i))
        ax.bar(x, select[name], color=barColors, edgecolor=edgeColor,
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
    gf.align_yaxis(axes[0], 0, axes[1], 0)
    # keep data range whithout distortion, preserve 0 alignment
    gf.change_plot_trace_amplitude(axes[1], 0.80)
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
    colors = ['k', stdColors['rouge'], stdColors['vert'],
              stdColors['jaune'], stdColors['bleu'], stdColors['bleu']]
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
    ax.plot(0, df.loc[0]['CENTER-ONLY'], 'o', color=stdColors['bleu'])

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
    light_colors = [stdColors['rouge'], stdColors['vert'],
                    stdColors['jaune'], stdColors['bleu'],
                    stdColors['bleu']]
    dark_colors = [stdColors['dark_rouge'], stdColors['dark_vert'],
                   stdColors['dark_jaune'], stdColors['dark_bleu'],
                   stdColors['dark_bleu'], stdColors['dark_bleu']]
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
    colors = ['k', stdColors['rouge'], stdColors['vert'],
              stdColors['jaune'], stdColors['bleu']]
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
    colors = ['k', stdColors['rouge'], stdColors['vert'],
              stdColors['jaune'], stdColors['bleu']]
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

#%% sup7 = fig5B

#%% fig supp3 bars

def print_keys(alist):
    """ build a list of the keys defining a stimulation """
    keys = []
    for item in alist:
        for key in item.split('_'):
            if key not in keys:
                keys.append(key)
    print(keys)

def build_keys_list(alist):
    """
    build a list to use the name of the file:
       [[vm, spk], [s, f], [], [] ...]
    """
    keys = []
    for i in range(7):
        keys.append([])
    for item in alist:
        for i, key in enumerate(item.split('_')):
            if key not in keys[i]:
                keys[i].append(key)
    return keys


filename = 'data/figSup34Spk.xlsx'
filename = 'data/figSup34Vm.xlsx'

df = pd.read_excel(filename)
df.set_index('Neuron', inplace=True)
# rename using snake_case
cols = ld.new_columns_names(df.columns)
df.columns = cols
# check stimulations
print_keys(cols)
# build key listing
# ex [['vm'],['s', 'f'],['cp', 'cf', 'rnd'],['iso', 'cross'],['stc'],
# ['dlat50', 'dgain50'],['indisig']]
keys = build_keys_list(cols)

#%%
# #latency advance
# sec_lat = [item for item in cols if '_s_' in item and '_dlat50' in item]
# adf = df[sec_lat]

# # retriving the numbers:
# # latency cpiso
# cond = 'vm_s_cp_iso_stc_dlat50'
# signi = cond + '_indisig'
# mean = adf.loc[adf[signi] > 0, cond].mean()
# std = adf.loc[adf[signi] > 0, cond].std()
# print(cond, 'mean= %2.2f, std: %2.2f'% (mean, std))
# # !!! doesnt suit with the figure !!!
#%%
plt.close("all")


def extract_values(df, stim_kind='s', measure='lat'):
    """ extract pop and response dico:
        input : dataframe, stim kind (s or f) and mesaure kind (lat or gain)
    """
    stim = '_' + stim_kind + '_'
    mes = '_d' + measure + '50'
    # restrict df
    restricted_list = [item for item in cols if stim in item and mes in item]
    adf = df[restricted_list]
    #compute values
    records = [item for item in restricted_list if 'indisig' not in item]
    pop_dico = {}
    resp_dico = {}
    for cond in records:
        signi = cond + '_indisig'
        pop_num = len(adf)
        signi_num = len(adf.loc[adf[signi] > 0, cond])
        percent = round((signi_num / pop_num) * 100)
        leg_cond = cond.split('_')[2] + '-' + cond.split('_')[3]
        leg_cond = leg_cond.upper()
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


def plot_cell_contribution(df):
    "sup 2A"
    colors = [stdColors['rouge'], stdColors['vert'],
              stdColors['jaune'], stdColors['bleu']]
    dark_colors = [stdColors['dark_rouge'], stdColors['dark_vert'],
                   stdColors['dark_jaune'], stdColors['dark_bleu']]
    fig = plt.figure(figsize=(8, 8))
    if anot:
        fig.suptitle('vm')
    # sector phase
    ax = fig.add_subplot(221)
    ax.set_title(r'$\Delta$ Phase (% significant cells)', pad=0)
    stim = 's'
    mes = 'lat'
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
    stim = 's'
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
    stim = 'f'
    mes = 'lat'
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
    stim = 'f'
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

    if filename == 'data/figSup34Spk.xlsx':
        fig.text(0.5, 1.01, 'Spikes',
                 ha='center', va='top',
                 fontsize=18)
    if anot:
        date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        fig.text(0.99, 0.01, 'centrifigs.py:plot_figSup2A',
                 ha='right', va='bottom', alpha=0.4)
        fig.text(0.01, 0.01, date, ha='left', va='bottom', alpha=0.4)

    fig.tight_layout()

plot_cell_contribution(df)


#%%
plt.close('all')

# plot latency (left) and gain (right



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
    colors = [stdColors['rouge'], stdColors['rouge'],
              stdColors['vert'], stdColors['vert'],
              stdColors['jaune'], stdColors['jaune'],
              stdColors['bleu'], stdColors['bleu']]
    # data (call)
    df = ld.load_cell_contributions(dico['kind'])
    # extract list of traces : sector vs full
    traces = [item for item in df.columns if dico['spread']+'_' in item[:7]]
    # filter -> only significative cells
    traces = [item for item in traces if 'indisig' not in item]
    # text labels
    title_dico = {'spk' : 'Spikes',
                  'vm' : 'Vm',
                  'f' : 'Full',
                  's' : 'Sector'
                  }
    title = title_dico[dico['kind']] + ' (' + title_dico[dico['spread']] + ')'
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
        sig_name = name + '_indisig'
        # color : white if non significant, edgecolor otherwise
        edgeColor = colors[i]
        color_dic = {0 : 'w', 1 : edgeColor}
        select = df[[name, sig_name]].sort_values(by=[name, sig_name],
                                                  ascending=False)
        barColors = [color_dic[x] for x in select[sig_name]]
        ax = axes[i]
        ax.set_title(name)
        ax.bar(x, select[name], color=barColors, edgecolor=edgeColor,
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
    gf.align_yaxis(axes[0], 0, axes[1], 0)
    # keep data range whithout distortion, preserve 0 alignment
    gf.change_plot_trace_amplitude(axes[1], 0.80)
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
        'spread' : 's',
        'position' : 'cp',
        'theta' : 'cross',
        'extra' : 'stc'
        }

fig = plot_sorted_responses(parameter_dico)
# iterate through conditions for plotting
for kind in ['vm', 'spk']:
    parameter_dico['kind'] = kind
    for spread in ['s', 'f']:
        parameter_dico['spread'] = spread
        fig = plot_sorted_responses(parameter_dico)

#%% opt
colors = ['k', stdColors['rouge'], speedColors['orangeFonce'],
          speedColors['orange'], speedColors['jaune']]
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
    colors = ['k', stdColors['rouge'], speedColors['orangeFonce'],
              speedColors['orange'], speedColors['jaune']]
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

#%% bar plot peaks
plt.close('all')


def load_peakdata(name):
    'load the excel file'
    df = pd.read_excel(name)
    # replace 'sec' by 'sect' for homogeneity
    new_list = []
    for item in df.columns:
        new_list.append(item.replace('sec', 'sect'))
    df.columns = new_list
    # adapt column names
    new_list = []
    for item in df.iloc[0].tolist():
        if 'value' in str(item):
            new_list.append('_gain')
        elif 'time' in str(item):
            new_list.append('_time')
        else:
            new_list.append('')
    cols = [item.split('.')[0] for item in df.columns]
    cols = [a + b for a, b in zip(cols, new_list)]
    df.columns = cols
    # remove first line
    df = df.drop(df.index[0])
    # remove empty column
    df = df.drop('Unnamed: 10', axis=1)
    df = df.set_index('Neuron')
    df = df.astype('float')
    return df


def normalize_peakdata_and_select(df, spread='sect', param='gain'):
    """
    return the normalized and selected df parts for plotting
    spread in ['sect', 'full'],
    param in ['time', 'gain']
    """
    if spread not in ['sect', 'full']:
        print("'spread' should be in ['sect', 'full']")
        return
    elif param not in ['time', 'gain']:
        print("'param' should be in ['time', 'gain']")
        return
    # select by param (first value = control)
    col_list = [item for item in df.columns if param in item]
    # normalization with center only (Y - Yref)/Yref
    ctrl = df[col_list[0]]
    for item in col_list[1:]:
        print(df[item].mean())
        df[item] = (df[item] - ctrl) / ctrl
        print(df[item].mean())
    # select by spread
    col_list = [item for item in col_list if spread in item]
    return df[col_list]


def plot_sorted_responses(df_left, df_right, mes='', overlap=True):
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

    colors = [stdColors['rouge'], stdColors['vert'],
              stdColors['jaune'], stdColors['bleu']]
    dark_colors = [stdColors['dark_rouge'], stdColors['dark_vert'],
                   stdColors['dark_jaune'], stdColors['dark_bleu']]

    # text labels
    if 'sect' in right.columns[0].split('_')[0]:
        spread = 'sect'
    else:
        spread = 'full'
    title = 'sorted_responses' + ' (' + mes + ' ' + spread + ')'
    anotx = 'Cell rank'
    anoty = [df_left.columns[0][5:], df_right.columns[0][5:]]
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
    left_sig = True
    if left_sig:
        traces = [item for item in left.columns if '_sig' not in item]
        for i, name in enumerate(traces):
            #color : white if non significant, edgecolor otherwise
            edgeColor = colors[i]
            color_dic = {0 : 'w', 1 : edgeColor}
            sig_name = name + '_sig'
            select = df_left[[name, sig_name]].sort_values(by=[name, sig_name],
                                                       ascending=False)
            barColors = [color_dic[x] for x in select[sig_name]]
            ax = left_axes[i]
            # ax.set_title(str(i))
            ax.set_title(name)
            # without significance
            #select = df_left[name].sort_values(ascending=False)
            #ax.bar(x, select, color=colors[i], edgecolor=dark_colors[i],
            #       alpha=0.8, width=0.8)
            # # with significance
            select = df_left[name].sort_values(ascending=False)
            
            ax.bar(x, select, color=barColors, edgecolor=edgeColor,
                    alpha=0.8, width=0.8)
            if i == 0:
                ax.set_title(anoty[i])
    else:
        for i, name in enumerate(df_left.columns):
            ax = left_axes[i]
            # ax.set_title(str(i))
            ax.set_title(name)
            # without significance
            select = df_left[name].sort_values(ascending=False)
            ax.bar(x, select, color=colors[i], edgecolor=dark_colors[i],
                   alpha=0.8, width=0.8)
            if i == 0:
                ax.set_title(anoty[i])
    #right
    right_sig = True
    if right_sig:
        traces = [item for item in right.columns if '_sig' not in item]
        for i, name in enumerate(traces):
            #color : white if non significant, edgecolor otherwise
            edgeColor = colors[i]
            color_dic = {0 : 'w', 1 : edgeColor}
            sig_name = name + '_sig'
            select = df_right[[name, sig_name]].sort_values(by=[name, sig_name],
                                                       ascending=False)
            barColors = [color_dic[x] for x in select[sig_name]]
            ax = right_axes[i]
            # ax.set_title(str(i))
            ax.set_title(name)
            select = df_right[name].sort_values(ascending=False)
            ax.bar(x, select, color=barColors, edgecolor=edgeColor,
                    alpha=0.8, width=0.8)
            if i == 0:
                ax.set_title(anoty[i])
    else:
        # left
        for i, name in enumerate(df_right.columns):
            ax = right_axes[i]
            # ax.set_title(str(i))
            ax.set_title(name)
            # without significance
            select = df_right[name].sort_values(ascending=False)
            ax.bar(x, select, color=colors[i], edgecolor=dark_colors[i],
                   alpha=0.8, width=0.8)
            if i == 0:
                ax.set_title(anoty[i])

    # alternate the y_axis position
    axes = fig.get_axes()
    left_axes = axes[:4]
    right_axes = axes[4:]
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
    for ax in left_axes:
        custom_ticks = np.linspace(0, 0.5, 2)
        ax.set_yticks(custom_ticks)
    for ax in right_axes:
        custom_ticks = np.linspace(0, 0.5, 2)
        ax.set_yticks(custom_ticks)
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
        fig.text(0.99, 0.01, 'centrifigs.py:plot_sorted_responses',
                 ha='right', va='bottom', alpha=0.4)
        fig.text(0.01, 0.01, date, ha='left', va='bottom', alpha=0.4)
    return fig



def select_50(df, spread='sect', param='gain', noSig=True):
    """
    return the selected df parts for plotting
    spread in ['sect', 'full'],
    param in ['time', 'gain']
    """
    if spread not in ['sect', 'full']:
        print("'spread' should be in ['sect', 'full']")
        return
    elif param not in ['time', 'gain']:
        print("'param' should be in ['time', 'gain']")
        return
    # select by param (first value = control)
    col_list = [item for item in df.columns if param in item]
    # select by spread
    col_list = [item for item in col_list if spread in item]
    if noSig:
        # remove sig
        col_list = [item for item in col_list if 'sig' not in item]
    return df[col_list]

def horizontal_dot_plot(df_left, df_right, mes=''):

    colors = [stdColors['rouge'], stdColors['vert'],
              stdColors['jaune'], stdColors['bleu']]
    dark_colors = [stdColors['dark_rouge'], stdColors['dark_vert'],
                   stdColors['dark_jaune'], stdColors['dark_bleu']]
     # text labels
    if 'sect' in df_right.columns[0].split('_')[0]:
        spread = 'sect'
    else:
        spread = 'full'
    title = 'sorted_peak_responses' + ' (' + mes + ' ' + spread + ')'
    anoty = 'Cell rank'
    anotx = [df_left.columns[0][5:], df_right.columns[0][5:]]

    fig = plt.figure()
    fig.suptitle(title)
    # left
    ax = fig.add_subplot(121)
    df = df_left.sort_values(by=df_left.columns[0], ascending=True)
    sorted_cells = df.index.copy()
    df = df.reset_index(drop=True)
    df.index += 1 #'cell name from 0'
    df *= -1 # time
    for i, col in enumerate(df.columns):
        ax.plot(df[col], df.index, 'o', color=colors[i], alpha=0.6)
    ax.set_yticks([1, len(df)])
    ax.set_ylim(-1, len(df)+1)
    ax.set_xlabel('time')
    # right
    ax = fig.add_subplot(122)
    # df = df_right.sort_values(by=df_right.columns[0], ascending=True)
    df = df_right.reindex(sorted_cells)
    df = df.reset_index(drop=True)
    df.index += 1
    for i, col in enumerate(df.columns):
        ax.plot(df[col], df.index, 'o', color=colors[i], alpha=0.6)
    ax.set_yticks([1, len(df)])
    ax.set_ylim(-1, len(df)+1)
    ax.set_xlabel('gain')
    for ax in fig.get_axes():
        for spine in ['top', 'right']:
            ax.spines[spine].set_visible(False)
        lims = ax.get_ylim()
        ax.vlines(0, lims[0], lims[1], 'k', alpha=0.3)
    if anot:
        date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        fig.text(0.99, 0.01, 'centrifigs.py:horizontal_dot_plot',
                 ha='right', va='bottom', alpha=0.4)
        fig.text(0.01, 0.01, date, ha='left', va='bottom', alpha=0.4)
    return fig

def scatter_lat_gain(df_left, df_right, mes=''):
    colors = [stdColors['rouge'], stdColors['vert'],
              stdColors['jaune'], stdColors['bleu']]
    dark_colors = [stdColors['dark_rouge'], stdColors['dark_vert'],
                   stdColors['dark_jaune'], stdColors['dark_bleu']]
     # text labels
    if 'sect' in df_right.columns[0].split('_')[0]:
        spread = 'sect'
    else:
        spread = 'full'
    title = 'responses' + ' (' + mes + ' ' + spread + ')'
    fig = plt.figure(figsize=(8, 8))
    fig.suptitle(title)
    ax = fig.add_subplot(111)
    for i in range(len(df_left.columns)):
        color_list = []
        for j in range(len(df_left)):
            color_list.append(colors[i])
        ax.scatter(left[df_left.columns[i]], df_right[right.columns[i]],
                   c=color_list,
                   edgecolors=dark_colors[i], alpha=0.6)
    ax.set_xlabel('time')
    ax.set_ylabel('gain')
    lims = ax.get_xlim()
    ax.hlines(0, lims[0], lims[1], 'k', alpha=0.3)
    lims = ax.get_ylim()
    ax.vlines(0, lims[0], lims[1], 'k', alpha=0.3)

    if anot:
        date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        fig.text(0.99, 0.01, 'centrifigs.py:horizontal_dot_plot',
                 ha='right', va='bottom', alpha=0.4)
        fig.text(0.01, 0.01, date, ha='left', va='bottom', alpha=0.4)
    return fig

def histo_lat_gain(df_left, df_right, mes=''):
    """
    histogramme des données
    """
    colors = [stdColors['rouge'], stdColors['vert'],
              stdColors['jaune'], stdColors['bleu']]
    dark_colors = [stdColors['dark_rouge'], stdColors['dark_vert'],
                   stdColors['dark_jaune'], stdColors['dark_bleu']]

    # text labels
    if 'sect' in right.columns[0].split('_')[0]:
        spread = 'sect'
    else:
        spread = 'full'
    title = 'peak_responses' + ' (' + mes + ' ' + spread + ')'
    anotx = 'Cell rank'
    anoty = [df_left.columns[0][5:], df_right.columns[0][5:]]
    # anoty = ['Relative peak advance(ms)', 'Relative peak amplitude']
    #          #(fraction of Center-only response)']
    # plot
    fig = plt.figure(figsize=(8, 8))
    gs = fig.add_gridspec(4, 2)
    # left
    left_axes = []
    ax = fig.add_subplot(gs[0, 0])
    left_axes.append(ax)
    for i in range(1, 4):
        left_axes.append(fig.add_subplot(gs[i, 0], sharex=ax))
    # right
    right_axes = []
    ax = fig.add_subplot(gs[0, 1])
    right_axes.append(ax)
    for i in range(1, 4):
        right_axes.append(fig.add_subplot(gs[i, 1], sharex=ax))
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
        fig.text(0.99, 0.01, 'centrifigs.py:histo_lat_gain',
                 ha='right', va='bottom', alpha=0.4)
        fig.text(0.01, 0.01, date, ha='left', va='bottom', alpha=0.4)
    return fig

def plot_glob_varia(left, right, mes=''):
    pass


def extract_stat(onlySig=False):
    # output
    desc_df = pd.DataFrame()
    # vm
    mes = 'vm'
    filename = 'data/cg_peakValueTime_vm.xlsx'
    data50 = ld.load_50vals(mes)
    if onlySig:
        data50 = data50.loc[data50.cpisosect_time50_sig > 0]
    print(len(data50), ' cells')
    # remove the significance
    data50 = data50[[item for item in data50.columns if '_sig' not in item]]
    sigCells = data50.index.to_list()
    # vm_time50
    times = [item for item in data50.columns if 'time' in item]
    times = [item for item in times if 'sect_' in item] \
            + [item for item in times if 'full_' in item]
    advance_df = data50[times].copy()
    advance_df.columns = [item.split('_')[0] for item in advance_df.columns]
    advance_df.rename(columns={'rndsect':'rndisosect', 'rndfull':'rndisofull'}, inplace=True)
    desc_df['time50_vm_mean'] = advance_df.mean()
    desc_df['time50_vm_std'] = advance_df.std()
    desc_df['time50_vm_med'] = advance_df.median()
    desc_df['time50_vm_mad'] = advance_df.mad()
    # vm_gain50
    gains = [item for item in data50.columns if 'gain' in item]
    gains = [item for item in gains if 'sect_' in item] \
            + [item for item in gains if 'full_' in item]
    gain_df = data50[gains].copy()
    gain_df.columns = [item.split('_')[0] for item in gain_df.columns]
    gain_df.rename(columns={'rndsect':'rndisosect', 'rndfull':'rndisofull'},
                   inplace=True)
    desc_df['gain50_vm_mean'] = gain_df.mean()
    desc_df['gain50_vm_std'] = gain_df.std()
    desc_df['gain50_vm_med'] = gain_df.median()
    desc_df['gain50_vm_mad'] = gain_df.mad()

    # spike
    mes = 'spk'
    filename = 'data/cg_peakValueTime_spk.xlsx'
    data50 = ld.load_50vals(mes)
    if onlySig:
        data50 = data50.loc[data50.cpisosect_time50_sig > 0]
    # remove the significance
    data50 = data50[[item for item in data50.columns if '_sig' not in item]]
    # spk_time50
    times = [item for item in data50.columns if 'time' in item]
    times = [item for item in times if 'sect_' in item] \
            + [item for item in times if 'full_' in item]
    time_df = data50[times].copy()
    time_df.columns = [item.split('_')[0] for item in time_df.columns]
    desc_df['time50_spk_mean'] = time_df.mean()
    desc_df['time50_spk_std'] = time_df.std()
    desc_df['time50_spk_med'] = time_df.median()
    desc_df['time50_spk_mad'] = time_df.mad()
    # spk_gain50
    gains = [item for item in data50.columns if 'gain' in item]
    gains = [item for item in gains if 'sect_' in item] \
            + [item for item in gains if 'full_' in item]
    gain_df = data50[gains].copy()
    gain_df.columns = [item.split('_')[0] for item in gain_df.columns]
    desc_df['gain50_spk_mean'] = gain_df.mean()
    desc_df['gain50_spk_std'] = gain_df.std()
    desc_df['gain50_spk_med'] = gain_df.median()
    desc_df['gain50_spk_mad'] = gain_df.mad()

    # peak (non normalized data)
    # vm
    filename = 'data/cg_peakValueTime_vm.xlsx'
    data = load_peakdata(filename)
    gains = [item for item in data.columns if 'gain' in item]
    times = [item for item in data.columns if 'time' in item]
    # normalise
    for item in gains[1:]:
        data[item] = data[item] / data[gains[0]]
    # for item in times[1:]:
    #     data[item] = data[item] / data[times[0]]
    # select
    gain_df = data[gains[1:]].copy()
    time_df = data[times[1:]].copy()
    #stat
    gain_df.columns = [item.split('_')[0] for item in gain_df.columns]
    gain_df.rename(columns={'rndsect':'rndisosect', 'cpisosfull':'cpisofull',
                            'rndfull':'rndisofull'}, inplace=True)
    desc_df['gainP_vm_mean'] = gain_df.mean()
    desc_df['gainP_vm_std'] = gain_df.std()
    desc_df['gainP_vm_med'] = gain_df.median()
    desc_df['gainP_vm_mad'] = gain_df.mad()

    time_df.columns = [item.split('_')[0] for item in time_df.columns]
    time_df.rename(columns={'rndsect':'rndisosect', 'cpisosfull':'cpisofull',
                            'rndfull':'rndisofull'}, inplace=True)
    desc_df['timeP_vm_mean'] = time_df.mean()
    desc_df['timeP_vm_std'] = time_df.std()
    desc_df['timeP_vm_med'] = time_df.median()
    desc_df['timeP_vm_mad'] = time_df.mad()

    # spk_time
    filename = 'data/cg_peakValueTime_spk.xlsx'
    data = load_peakdata(filename)
    gains = [item for item in data.columns if 'gain' in item]
    times = [item for item in data.columns if 'time' in item]
    # normalise
    for item in gains[1:]:
        data[item] = data[item] / data[gains[0]]
    # for item in times[1:]:
    #     data[item] = data[item] / data[times[0]]
    # select
    gain_df = data[gains[1:]].copy()
    time_df = data[times[1:]].copy()
    # stat
    gain_df.columns = [item.split('_')[0] for item in gain_df.columns]
    gain_df.rename(columns={'rndsect':'rndisosect', 'cpisosfull':'cpisofull',
                            'rndfull':'rndisofull'}, inplace=True)
    desc_df['gainP_spk_mean'] = gain_df.mean()
    desc_df['gainP_spk_std'] = gain_df.std()
    desc_df['gainP_spk_med'] = gain_df.median()
    desc_df['gainP_spk_mad'] = gain_df.mad()

    time_df.columns = [item.split('_')[0] for item in time_df.columns]
    time_df.rename(columns={'rndsect':'rndisosect', 'cpisosfull':'cpisofull',
                            'rndfull':'rndisofull'}, inplace=True)
    desc_df['timeP_spk_mean'] = time_df.mean()
    desc_df['timeP_spk_std'] = time_df.std()
    desc_df['timeP_spk_med'] = time_df.median()
    desc_df['timeP_spk_mad'] = time_df.mad()

    return desc_df


plt.close('all')

def plot_stat(stat_df, kind='mean', loc='50'):
    """
    plot the stats
    input : stat_df, kind in ['mean', 'med'], loc in ['50', 'peak']
    output : matplotlib figure
    """
    if kind == 'mean':
        stat = ['_mean', '_std']
    elif kind == 'med':
        stat = ['_med', '_mad']
    else:
        print('non valid kind argument')
        return
    if loc == '50':
        mes = ['time50', 'gain50']
    elif loc == 'peak':
        mes = ['timeP', 'gainP']
    else:
        print('non valid loc argument')
        return

    colors = [stdColors['rouge'], stdColors['vert'],
              stdColors['jaune'], stdColors['bleu']]
    fig = plt.figure(figsize=(8, 8))
    title = stat[0] + stat[1] + '\n' + mes[0] + mes[1]
    fig.suptitle(title)
    # sect vm
    axes = []
    ax = fig.add_subplot(221)
    axes.append(ax)
    ax1 = fig.add_subplot(2, 2, 2, sharex=ax, sharey=ax)
    axes.append(ax1)
    ax2 = fig.add_subplot(2, 2, 3, sharex=ax)
    axes.append(ax2)
    ax3 = fig.add_subplot(2, 2, 4, sharex=ax, sharey=ax2)
    axes.append(ax3)
    # vm
    ax.set_title('vm, sector')
    vals = [item for item in stat_df.columns if '_vm' in item]
    # sector
    spread = [item for item in stat_df.index if 'sect' in item]
    df = stat_df.loc[spread].copy()
    xvals = [item for item in vals if mes[0] in item \
         and (stat[0] in item or stat[1] in item)]
    yvals = [item for item in vals if mes[1] in item \
         and (stat[0] in item or stat[1] in item)]

    x = df[xvals[0]]
    y = df[yvals[0]]
    xerr = df[xvals[1]]
    yerr = df[yvals[1]]
    for xi, yi, xe, ye, ci  in zip(x, y, xerr, yerr, colors):
        ax.errorbar(xi, yi, xerr=xe, yerr=ye,
                    fmt='s', color=ci)
    # full
    ax = axes[1]
    ax.set_title('vm, full')
    spread = [item for item in stat_df.index if 'full' in item]
    df = stat_df.loc[spread].copy()
    xvals = [item for item in vals if mes[0] in item \
             and (stat[0] in item or stat[1] in item)]
    yvals = [item for item in vals if mes[1] in item \
             and (stat[0] in item or stat[1] in item)]

    x = df[xvals[0]]
    y = df[yvals[0]]
    xerr = df[xvals[1]]
    yerr = df[yvals[1]]
    for xi, yi, xe, ye, ci  in zip(x, y, xerr, yerr, colors):
        ax.errorbar(xi, yi, xerr=xe, yerr=ye,
                    fmt='s', color=ci)
    # spikes
    ax = axes[2]
    ax.set_title('spk, sector')
    vals = [item for item in stat_df.columns if '_spk' in item]
    # sector
    spread = [item for item in stat_df.index if 'sect' in item]
    df = stat_df.loc[spread].copy()
    xvals = [item for item in vals if mes[0] in item \
             and (stat[0] in item or stat[1] in item)]
    yvals = [item for item in vals if mes[1] in item \
             and (stat[0] in item or stat[1] in item)]

    x = df[xvals[0]]
    y = df[yvals[0]]
    xerr = df[xvals[1]]
    yerr = df[yvals[1]]
    for xi, yi, xe, ye, ci  in zip(x, y, xerr, yerr, colors):
        ax.errorbar(xi, yi, xerr=xe, yerr=ye,
                    fmt='s', color=ci)
    # full
    ax = axes[3]
    ax.set_title('spk, full')
    spread = [item for item in stat_df.index if 'full' in item]
    df = stat_df.loc[spread].copy()
    xvals = [item for item in vals if mes[0] in item \
             and (stat[0] in item or stat[1] in item)]
    yvals = [item for item in vals if mes[1] in item \
             and (stat[0] in item or stat[1] in item)]
    x = df[xvals[0]]
    y = df[yvals[0]]
    xerr = df[xvals[1]]
    yerr = df[yvals[1]]
    for xi, yi, xe, ye, ci  in zip(x, y, xerr, yerr, colors):
        ax.errorbar(xi, yi, xerr=xe, yerr=ye,
                    fmt='s', color=ci)

    for i, ax in enumerate(axes):
        # lims = ax.get_ylim()
        # ax.vlines(0, lims[0], lims[1], linestyle=':', alpha=0.3)
        # lims = ax.get_xlim()
        # ax.hlines(0, lims[0], lims[1], linestyle=':', alpha=0.3)
        for spine in ['top', 'right']:
            ax.spines[spine].set_visible(False)
            if i % 2 == 0:
                ax.set_ylabel('gain')
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
        fig.text(0.99, 0.01, 'centrifigs.py:plot_stat',
                 ha='right', va='bottom', alpha=0.4)
        fig.text(0.01, 0.01, date, ha='left', va='bottom', alpha=0.4)
    return fig

# to have sig and non sig:
def sigNonSig_stat_plot():
    """ stats for sig and non sig cells """
    stat_df = extract_stat(onlySig=False)
    fig0 = plot_stat(stat_df, 'med', '50')
    stat_df = extract_stat(onlySig=True)
    fig1 = plot_stat(stat_df, 'med', '50')
    axes0 = fig0.get_axes()
    axes1 = fig1.get_axes()
    ax0 = axes0[0]
    ax1 = axes1[0]
    leg = '37 cells'
    ax0.text(0.8, 0.8, leg, transform=ax0.transAxes)
    leg = '10 cells'
    ax1.text(0.8, 0.8, leg, transform=ax1.transAxes)
    for ax in [ax0, ax1]:
        ax.set_ylim(-0.15, 0.5)
    ax0 = axes0[2]
    ax1 = axes1[2]
    for ax in [ax0, ax1]:
        ax.set_ylim(-0.16, 0.75)
        ax.set_xlim(-12, 20)
    # zeroline
    alpha = 0.3
    for ax in fig0.get_axes():
        lims = ax.get_xlim()
        ax.hlines(0, lims[0], lims[1], alpha=alpha)
        lims = ax.get_ylim()
        ax.vlines(0, lims[0], lims[1], alpha=alpha)
    for ax in fig1.get_axes():
        lims = ax.get_xlim()
        ax.hlines(0, lims[0], lims[1], alpha=alpha)
        lims = ax.get_ylim()
        ax.vlines(0, lims[0], lims[1], alpha=alpha)


stat_df = extract_stat(onlySig=False)
plot_stat(stat_df, 'med', 'peak')
plot_stat(stat_df, 'med', '50')


#%% vm
plt.close('all')
for spread in ['sect', 'full']:
    mes = 'vm'
    data50 = ld.load_50vals(mes)
    advance_df = select_50(data50, spread=spread, param='time')
    left = advance_df

    filename = 'data/cg_peakValueTime_vm.xlsx'
    data = load_peakdata(filename)
    right = normalize_peakdata_and_select(data.copy(), spread=spread, param='gain')
    fig = plot_sorted_responses(left, right, mes=mes, overlap=True)
    fig2 = horizontal_dot_plot(left, right, mes=mes)
    fig3 = scatter_lat_gain(left, right, mes=mes)
    fig4 = histo_lat_gain(left, right, mes=mes)

#%% spk
for spread in ['sect', 'full']:
    mes = 'spk'
    data50 = ld.load_50vals(mes)
    advance_df = select_50(data50, spread=spread, param='time')
    left = advance_df

    filename = 'data/cg_peakValueTime_spk.xlsx'
    data = load_peakdata(filename)
    right = normalize_peakdata_and_select(data.copy(), spread=spread, param='gain')
    fig = plot_sorted_responses(left, right, mes=mes, overlap=True)
    fig2 = horizontal_dot_plot(left, right, mes=mes)
    fig3 = scatter_lat_gain(left, right, mes=mes)
    fig4 = histo_lat_gain(left, right, mes=mes)


#%% plot energy vm
def adapt_energy_to_plot(energy_df, spread='sect'):
    df = energy_df.copy()
    #remove stats
#    cols = [col for col in df.columns if '_p' not in col] 
    #select sector
    ctr = df['ctronly'].copy()
    cols = [col for col in df.columns if spread[:3] in col]
    df = df[cols].copy()
    #normalize
    traces = [col for col in cols if '_sig' not in col]
    for col in traces:
        df[col] = (df[col] - ctr) / ctr
    return df

spread = 'sect'
mes = 'vm'
data50 = ld.load_50vals(mes)
gain_df = select_50(data50, spread=spread, param='gain', noSig=False)
left = gain_df



right = adapt_energy_to_plot(energy_df)
fig = plot_sorted_responses(left, right, mes=mes, overlap=True)

#%%cellsDepth / advance
plt.close('all')

def plot_cellDepth():
    """
    relation profondeur / latence
    """
    # cells
    data50 = ld.load_50vals('vm')
    # retain only the neuron names
    data50.reset_index(inplace=True)
    data50.Neuron = data50.Neuron.apply(lambda x: x.split('_')[0])
    data50.set_index('Neuron', inplace=True)
    # layers (.csv file include decimals in dephts -> bugs)
    filename = os.path.join(paths['cgFig'], 'cells/centri_neurons_histolog_170105.xlsx')
    df = pd.read_excel(filename)
    df.set_index('Neuron', inplace=True)
    # lay_df = pd.concat([data50, df])
    lay_df = pd.concat([data50, df], axis=1, join='inner')
    labelled = lay_df.dropna(subset=['CDLayer']).copy()
    labelled.CDLayer = labelled.CDLayer.astype('int')

    colors = {6:'k', 5:'b', 4:'g', 3:'r'}
    fig = plt.figure()
    fig.suptitle('cp iso  : layers effect')
    ax = fig.add_subplot(211)
    kind = 'cpisosect_time50'
    labelled = labelled.sort_values(by=kind, ascending=False)
    y = labelled.cpisosect_time50.to_list()
    x = labelled.index.to_list()
    z = [colors[item] for item in labelled.CDLayer.to_list()]
    # stat
    z1 = z.copy()
    z2 = []
    for a, b in zip(z1, labelled[kind + '_sig'].to_list()):
        if b == 1:
            z2.append(a)
        else:
            z2.append('w')
    ax.bar(x, y, color=z2, edgecolor=z1, linewidth=3,
           alpha=0.6, width=0.8)
    # ax.bar(x, y, color = z, alpha=0.6)
    ax.set_ylabel('advance (ms)')
    ax.set_xlabel('cell')
    lims = ax.get_xlim()
    ax.hlines(0, lims[0], lims[1], alpha=0.3)

    ax = fig.add_subplot(212)
    kind = 'cpisosect_gain50'
    y = labelled[kind].to_list()
    # stat
    z1 = z.copy()
    z2 = []
    for a, b in zip(z1, labelled[kind + '_sig'].to_list()):
        if b == 1:
            z2.append(a)
        else:
            z2.append('w')
    ax.bar(x, y, color=z2, edgecolor=z1, linewidth=3,
           alpha=0.6, width=0.8)

    # ax.bar(x, y, color = z, alpha=0.6)
    ax.set_ylabel('gain')
    ax.set_xlabel('cell')
    lims = ax.get_xlim()
    ax.hlines(0, lims[0], lims[1], alpha=0.3)

    # leg = 'black= layer 6, \n blue= layer 5, \n green= layer4, \n red=layer 3'
    axes = fig.get_axes()
    for i, ax in enumerate(axes):
        ax.set_xlim(-0.5, len(x))
        for spine in ['top', 'right']:
            ax.spines[spine].set_visible(False)
        if i == 0:
            # ax.text(0.8, 0.7, leg, transform=ax.transAxes)
            ax.xaxis.set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.text(0.80, 0.95, 'layer 3', color='r', transform=ax.transAxes)
            ax.text(0.80, 0.85, 'layer 4', color='g', transform=ax.transAxes)
            ax.text(0.80, 0.75, 'layer 5', color='b', transform=ax.transAxes)
            ax.text(0.80, 0.65, 'layer 6', color='k', alpha=0.6, transform=ax.transAxes)
            custom_ticks = np.linspace(0, 10, 2, dtype=int)
            ax.set_yticks(custom_ticks)
            ax.set_yticklabels(custom_ticks)
            ax.vlines(ax.get_xlim()[0], 0, 10, linewidth=2)
            ax.spines['left'].set_visible(False)
        else:
            ax.tick_params(axis='x', labelrotation=45)
            custom_ticks = np.linspace(0, 0.4, 2)
            ax.set_yticks(custom_ticks)
            ax.set_yticklabels(custom_ticks)
            ax.vlines(ax.get_xlim()[0], 0, 0.4, linewidth=2)
            ax.spines['left'].set_visible(False)
            
        fig.tight_layout()
    if anot:
        date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        fig.text(0.99, 0.01, 'centrifigs.py:plot_cellDepth',
                 ha='right', va='bottom', alpha=0.4)
        fig.text(0.01, 0.01, date, ha='left', va='bottom', alpha=0.4)
    return fig

fig = plot_cellDepth()

#%%
plt.close('all')

# TODO add the significativity for each condition
# ? grou by depth

def plot_cellDepth_all(spread='sect'):
    """
    relation profondeur / latence
    """
    # cells
    data50 = ld.load_50vals('vm')
    # retain only the neuron names
    data50.reset_index(inplace=True)
    data50.Neuron = data50.Neuron.apply(lambda x: x.split('_')[0])
    data50.set_index('Neuron', inplace=True)
    # layers (.csv file include decimals in dephts -> bugs)
    filename = os.path.join(paths['cgFig'], 'cells/centri_neurons_histolog_170105.xlsx')
    df = pd.read_excel(filename)
    df.set_index('Neuron', inplace=True)
    # lay_df = pd.concat([data50, df])
    lay_df = pd.concat([data50, df], axis=1, join='inner')
    labelled = lay_df.dropna(subset=['CDLayer']).copy()
    kind = 'cpisosect_time50'
    labelled.CDLayer = labelled.CDLayer.astype('int')
    # sort by response
    labelled = labelled.sort_values(by=kind, ascending=False)
    # an the sort by depth
    labelled = labelled.sort_values(by='CDLayer', ascending=True)
    colors = {6:'k', 5:'b', 4:'g', 3:'r'}
    x = labelled.index.to_list()
    y = labelled[kind].to_list()
    z = [colors[item] for item in labelled.CDLayer.to_list()]
    # stat z1 = list(depht_colors), z2 = list(white if non significant)
    z1 = z.copy()
    z2 = []
    for a, b in zip(z1, labelled[kind + '_sig'].to_list()):
        if b == 1:
            z2.append(a)
        else:
            z2.append('w')
    # figure
    fig, axes = plt.subplots(nrows=4, ncols=2, figsize=(12, 8), sharex=True)
    axes = axes.T.flatten().tolist()
    # shareY
    axr = axes[0]
    for ax in axes[0::2]:
        ax.get_shared_y_axes().join(ax, axr)
    axr = axes[1]
    for ax in axes[1::2]:
        ax.get_shared_y_axes().join(ax, axr)
    # fill axes
    # 'sect' or 'full'
    alist = [item for item in labelled.columns if spread in item]
    # remove separate the sig column
    val_list = [item for item in alist if '_sig' not in item]
    sig_list = [item for item in alist if '_sig' in item]
    # choose advance values
    cols = [item for item in val_list if 'time50' in item]
    sig_cols = [item for item in sig_list if 'time50' in item]
    for ax, col, sig_col in zip(axes[0::2], cols, sig_cols):
        ax.set_title(col.split('_')[0])
        ax.set_ylabel('advance')
        y = labelled[col].to_list()
        #stat z1 = list(depht_colors), z2 = list(white if non significant)
        z1 = z.copy()
        z2 = []
        for a, b in zip(z1, labelled[sig_col].to_list()):
            if b == 1:
                z2.append(a)
            else:
                z2.append('w')
        # ax.bar(x, y, color=z, alpha=0.6)
        ax.bar(x, y, color=z2, edgecolor=z1, linewidth=2,
               alpha=0.6, width=0.8)
        ax.set_xlabel('cell')
        ax.set_xlim(-0.5, len(x))
        lims = ax.get_xlim()
        ax.hlines(0, lims[0], lims[1], alpha=0.3)
        custom_ticks = np.linspace(0, 10, 2)
        ax.set_yticks(custom_ticks)
        ax.set_yticklabels(custom_ticks)
        ax.vlines(ax.get_xlim()[0], 0, 10, linewidth=2)
        ax.spines['left'].set_visible(False)
    # choose gain values
    cols = [item for item in val_list if 'gain50' in item]
    sig_cols = [item for item in sig_list if 'gain50' in item]
    for ax, col, sig_col in zip(axes[1::2], cols, sig_cols):
        ax.set_title(col.split('_')[0])
        ax.set_ylabel('gain')
        y = labelled[col].to_list()
        # stat z1 = list(depht_colors), z2 = list(white if non significant)
        z1 = z.copy()
        z2 = []
        for a, b in zip(z1, labelled[sig_col].to_list()):
            if b == 1:
                z2.append(a)
            else:
                z2.append('w')
        # ax.bar(x, y, color = z, alpha=0.6)
        ax.bar(x, y, color=z2, edgecolor=z1, linewidth=2,
               alpha=0.6, width=0.8)
        ax.set_xlabel('cell')
        lims = ax.get_xlim()
        ax.hlines(0, lims[0], lims[1], alpha=0.3)
        custom_ticks = np.linspace(0, 0.5, 2)
        ax.set_yticks(custom_ticks)
        ax.set_yticklabels(custom_ticks)
        ax.vlines(ax.get_xlim()[0], 0, 0.5, linewidth=2)
        ax.spines['left'].set_visible(False)

    for i, ax in enumerate(axes):
        for spine in ['top', 'right']:
            ax.spines[spine].set_visible(False)
        if i in [0, 4]:
            pass
            # ax.text(0.7, 0.7, leg, transform=ax.transAxes)
        if i not in [3, 7]:
            ax.xaxis.set_visible(False)
            ax.spines['bottom'].set_visible(False)
        else:
            ax.tick_params(axis='x', labelrotation=45)
    for ax in [axes[0], axes[4]]:
        ax.text(0.08, 0.85, 'layer 3', color='r', transform=ax.transAxes)
        ax.text(0.22, 0.85, 'layer 4', color='g', transform=ax.transAxes)
        ax.text(0.45, 0.85, 'layer 5', color='b', transform=ax.transAxes)
        ax.text(0.85, 0.85, 'layer 6', color='k', alpha=0.6, transform=ax.transAxes)
        ax.margins(0.01)

    fig.subplots_adjust(hspace=0.02)
    fig.subplots_adjust(wspace=0.2)
    fig.tight_layout()
    if anot:
        date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        fig.text(0.99, 0.01, 'centrifigs.py:plot_cellDepth_all',
                 ha='right', va='bottom', alpha=0.4)
        fig.text(0.01, 0.01, date, ha='left', va='bottom', alpha=0.4)
    return fig

# fig = plot_cellDepth()
fig1 = plot_cellDepth_all(spread='sect')
fig2 = plot_cellDepth_all(spread='full')

#%%
plt.close('all')

fig = plt.figure()
axes = []
ax = fig.add_subplot(421)
axes.append(ax)
for i in range(3, 8, 2):
    print(i)
    axes.append(fig.add_subplot(4, 2, i))
for i in range(2, 9, 2):
    axes.append(fig.add_subplot(4, 2, i))

fig, axes = plt.subplots(nrows=4, ncols=2)
axes = axes.T.flatten().tolist()
axr = axes[0]
for ax in axes[0::2]:
    ax.get_shared_y_axes().join(ax, axr)
axr = axes[1]
for ax in axes[1::2]:
    ax.get_shared_y_axes().join(ax, axr)
