

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
from matplotlib.patches import Rectangle
from datetime import datetime

#import math

#===========================
# global setup
font_size = 'medium' # large, medium
anot = True         # to draw the date and name on the bottom of the plot
#============================

def go_to_dir():
    """
    to go to the pg and file directory (spyder use)
    """
    osname = platform.system()
    username = getpass.getuser()
    if osname == 'Windows'and username == 'Benoit':
        os.chdir(r'D:\\travail\sourcecode\developing\paper\centriG')
    elif osname == 'Linux' and username == 'benoit':
        os.chdir(r'/media/benoit/data/travail/sourcecode/developing/paper/centriG')
    elif osname == 'Darwin' and username == 'cdesbois':
        os.chdir(r'/Users/cdesbois/pg/chrisPg/centriG')
    return True
go_to_dir()

#colors
stdColors = {'rouge' : [x/256 for x in [229, 51, 51]],
             'vert' : [x/256 for x in [127, 204, 56]],
             'bleu' :	[x/256 for x in [0, 125, 218]],
             'jaune' :	[x/256 for x in [238, 181, 0]],
             'violet' : [x/256 for x in [255, 0, 255]],
             'vertSombre': [x/256 for x in [0, 150, 68]]}
speedColors = {'orangeFonce' :     [x/256 for x in [252, 98, 48]],
               'orange' : [x/256 for x in [253, 174, 74]],
               'jaune' : [x/256 for x in [254, 226, 137]]}

params = {'font.sans-serif': ['Arial'],
          'font.size': 14,
          'legend.fontsize': font_size,
          'figure.figsize': (15, 5),
          'axes.labelsize': font_size,
          'axes.titlesize': font_size,
          'xtick.labelsize': font_size,
          'ytick.labelsize': font_size,
          'axes.xmargin': 0}
plt.rcParams.update(params)
plt.rcParams['axes.xmargin'] = 0            # no gap between axes and traces
#%%
def retrieve_name(var):
    """
    to retrieve the string value of a variable
    """
    callers_local_vars = inspect.currentframe().f_back.f_locals.items()
    return [var_name for var_name, var_val in callers_local_vars if var_val is var]


# adjust the y scale to allign plot for a value (use zero here)

#alignement to be performed
#see https://stackoverflow.com/questions/10481990/matplotlib-axis-with-two-scales-shared-origin/10482477#10482477

def align_yaxis(ax1, v1, ax2, v2):
    """adjust ax2 ylimit so that v2 in ax2 is aligned to v1 in ax1"""
    _, y1 = ax1.transData.transform((0, v1))
    _, y2 = ax2.transData.transform((0, v2))
    inv = ax2.transData.inverted()
    _, dy = inv.transform((0, 0)) - inv.transform((0, y1-y2))
    miny, maxy = ax2.get_ylim()
    ax2.set_ylim(miny+dy, maxy+dy)


def change_plot_trace_amplitude(ax, gain=1):
    """change the amplitude of the plot,
    doesn't change the zero location """
    lims = ax.get_ylim()
    new_lims = (lims[0]/gain, lims[1]/gain)
    ax.set_ylim(new_lims)
#
def properties(ax):
    """
    print size and attributes of an axe
    """
    size = ax.axes.xaxis.label.get_size()
    fontname = ax.axes.xaxis.label.get_fontname()
    print('xaxis:', fontname, size)
    size = ax.axes.yaxis.label.get_size()
    fontname = ax.axes.yaxis.label.get_fontname()
    print('yaxis:', fontname, size)


def fig_properties(fig):
    """
    exoplore figure properties
    """
    for ax in fig.get_axes():
        properties(ax)

#%%
plt.close('all')

def load2():
    """
    import the datafile
    return a pandasDataframe and a dictionary of contents
    """
    #____data
    filename = 'data/fig2traces.xlsx'
    data = pd.read_excel(filename)
    #centering
    middle = (data.index.max() - data.index.min())/2
    data.index = (data.index - middle)/10
    data = data.loc[-200:150]
    # nb dico : key + [values] or key + [values, (stdUp, stdDown)]
    colsdict = {
        'indVm' : ['indiVmctr', 'indiVmscpIsoStc'],
        'indSpk' : ['indiSpkCtr', 'indiSpkscpIsoStc'],
        'popVm' : ['popVmCtr', 'popVmscpIsoStc'],
        'popSpk' : ['popSpkCtr', 'popSpkscpIsoStc'],
        'popVmSig' : ['popVmCtrSig', 'popVmscpIsoStcSig',
                      ('popVmCtrSeUpSig', 'popVmCtrSeDwSig'),
                      ('popVmscpIsoStcSeUpSig', 'popVmscpIsoStcSeDwSig')],
        'popSpkSig' : ['popSpkCtrSig', 'popSpkscpIsoStcSig',
                       ('popSpkCtrSeUpSig', 'popSpkCtrSeDwSig'),
                       ('popSpkscpIsoStcSeUpSig', 'popSpkscpIsoStcSeDwSig')],
        'popVmNsig' : ['popVmCtrNSig', 'popVmscpIsoStcNSig',
                       ('popVmCtrSeUpNSig', 'popVmCtrSeDwNSig'),
                       ('popVmscpIsoStcSeUpNSig', 'popVmscpIsoStcSeDwNSig')],
        'popSpkNsig' : ['popSpkCtrNSig', 'popSpkscpIsoStcNSig',
                        ('popSpkCtrSeUpNSig', 'popSpkCtrSeDwNSig'),
                        ('popSpkscpIsoStcSeUpNSig', 'popSpkscpIsoStcSeDwNSig')],
        'sort' : ['popVmscpIsolatg', 'popVmscpIsoAmpg',
                  'lagIndiSig', 'ampIndiSig']
                }
    return data, colsdict

def plot_figure2(data, colsdict, fill=True):
    """
    plot_figure2
    """
    colors = ['k', stdColors['rouge']]
    alpha = [0.8, 0.8]

    fig = plt.figure(figsize=(14, 8)) #fig = plt.figure(figsize=(8, 8))
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
        ax.plot(data[col], color=colors[i], alpha=alpha[i],
                label=col)
    # invert the plot order for spikes
    inv_colors = colors[::-1]
    inv_alpha = alpha[::-1]
    #individual spike
    cols = colsdict['indSpk']
    ax = spkaxes[0]
    for i, col in enumerate(cols[::-1]):
        ax.fill_between(data.index, data[col],
                        color=colors[::-1][i], alpha=0.5, label=col)
        ax.plot(data[col], color=inv_colors[i],
                alpha=1, label=col, linewidth=1)
    #____ plots pop (column 1-3)
    df = data.loc[-30:35]       # limit xscale
    # pop vm
    cols = colsdict['popVm']
    ax = vmaxes[1]
    for i, col in enumerate(cols):
        ax.plot(df[col], color=colors[i], alpha=alpha[i],
                label=col)
    ax.annotate("n=37", xy=(0.2, 0.8),
                xycoords="axes fraction", ha='center')
    #popVmSig
    cols = colsdict['popVmSig']
    ax = vmaxes[2]
    #traces
    for i, col in enumerate(cols[:2]):
        ax.plot(df[col], color=colors[i], alpha=alpha[i],
                label=col)
        #errors : iterate on tuples
        for i, col in enumerate(cols[2:]):
            if fill:
                ax.fill_between(df.index, df[col[0]], df[col[1]],
                                color=colors[i], alpha=0.2)#alpha[i]/2)
            else:
                for i, col in enumerate(cols[2:]):
                    for j in [0, 1]:
                        ax.plot(df[col[j]], color=colors[i], alpha=alpha[i],
                                label=col, linewidth=0.5)
    ax.annotate("n=10", xy=(0.2, 0.8),
                xycoords="axes fraction", ha='center')
    #popVmNsig
    cols = colsdict['popVmNsig']
    ax = vmaxes[3]
    #traces
    for i, col in enumerate(cols[:2]):
        ax.plot(df[col], color=colors[i], alpha=alpha[i],
                label=col)
        #errors : iterate on tuples
        for i, col in enumerate(cols[2:]):
            if fill:
                ax.fill_between(df.index, df[col[0]], df[col[1]],
                                color=colors[i], alpha=0.2)#alpha[i]/2)
            else:
                for i, col in enumerate(cols[2:]):
                    for j in [0, 1]:
                        ax.plot(df[col[j]], color=colors[i], alpha=alpha[i],
                                label=col, linewidth=0.5)
    ax.annotate("n=27", xy=(0.2, 0.8),
                xycoords="axes fraction", ha='center')
    #pop spike
    cols = colsdict['popSpk']
    ax = spkaxes[1]
    for i, col in enumerate(cols[::-1]):
        ax.plot(df[col], color=inv_colors[i],
                alpha=1, label=col, linewidth=1)
        ax.fill_between(df.index, df[col],
                        color=inv_colors[i], alpha=0.5, label=col)
    ax.annotate("n=20", xy=(0.2, 0.8),
                xycoords="axes fraction", ha='center')

#TODO define a Spk plotmode[lines, allhist, sdFill] for popSpkSig and popSpkNsig
    #popSpkSig
    cols = colsdict['popSpkSig']
    ax = spkaxes[2]
    #traces
    for i, col in enumerate(cols[:2][::-1]):
        ax.fill_between(df.index, df[col], color=inv_colors[i],
                        alpha=inv_alpha[i]/2)
        ax.plot(df[col], color=inv_colors[i], alpha=inv_alpha[i],
                label=col, linewidth=2)
    #errors : iterate on tuples
    for i, col in enumerate(cols[2:]):
        for j in [0, 1]:
            ax.plot(df[col[j]], color=colors[i],
                    alpha=1, label=col, linewidth=0.5)
    ax.annotate("n=5", xy=(0.2, 0.8),
                xycoords="axes fraction", ha='center')
    #popSpkNsig
    cols = colsdict['popSpkNsig']
    ax = spkaxes[3]
    #traces
    for i, col in enumerate(cols[:2][::-1]):
        ax.fill_between(df.index, df[col], color=inv_colors[i],
                        alpha=inv_alpha[i]/2)
        ax.plot(df[col], color=inv_colors[i], alpha=inv_alpha[i],
                label=col, linewidth=2)
    #errors : iterate on tuples
    for i, col in enumerate(cols[2:]):
        for j in [0, 1]:
            ax.plot(df[col[j]], color=colors[i],
                    alpha=1, label=col, linewidth=0.5)
    #labels
    for ax in axes:
        for loca in ['top', 'right']:
            ax.spines[loca].set_visible(False)
    ylabels = ['membrane potential (mV)',
               'normalized membrane potential',
               '', '']
    for i, ax in enumerate(vmaxes):
        ax.axes.get_xaxis().set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.set_ylabel(ylabels[i])
    ylabels = ['firing rate (spikes/s)',
               'normalized firing rate',
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
    #lines
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
                         alpha=0.6, edgecolor='w', facecolor='r')
        ax.add_patch(rect)
        #center
    rect = Rectangle(xy=(0, -5), width=step, height=1, fill=True,
                     alpha=0.6, edgecolor='w', facecolor='k')
    ax.add_patch(rect)
    #fis individual example
    vmaxes[0].set_ylim(-4, 13)
    spkaxes[0].set_ylim(-5.5, 20)
    # align zero between plots  NB ref = first plot
    align_yaxis(vmaxes[0], 0, vmaxes[1], 0)
    align_yaxis(spkaxes[0], 0, spkaxes[1], 0)
    # adjust amplitude (without moving the zero)
    change_plot_trace_amplitude(vmaxes[1], 0.85)
    change_plot_trace_amplitude(spkaxes[1], 0.8)
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
        fig.text(0.99, 0.01, 'centrifigs.py:plot_figure2',
                 ha='right', va='bottom', alpha=0.4)
        fig.text(0.01, 0.01, date, ha='left', va='bottom', alpha=0.4)
    return fig

data, content = load2()
fig = plot_figure2(data, content)

#%%
plt.close('all')

def plot_half_figure2(data, colsdict):
    """
    plot_figure2
    """
    colors = ['k', stdColors['rouge']]
    alpha = [0.8, 0.8]

    fig = plt.figure(figsize=(8, 8))
    #build axes with sharex and sharey
    axes = []
    for i in range(4):
        axes.append(fig.add_subplot(2, 2, i+1))
    # axes list
    vmaxes = axes[:2]      # vm axes = top row
    spkaxes = axes[2:]     # spikes axes = bottom row
    #____ plots individuals (first column)
    # individual vm
    cols = colsdict['indVm']
    ax = vmaxes[0]
    for i, col in enumerate(cols):
        ax.plot(data[col], color=colors[i], alpha=alpha[i],
                label=col)
    #individual spike
    cols = colsdict['indSpk']
    ax = spkaxes[0]
    for i, col in enumerate(cols[::-1]):
        ax.plot(data[col], color=colors[::-1][i],
                alpha=1, label=col, linewidth=1)
        ax.fill_between(data.index, data[col],
                        color=colors[::-1][i], alpha=0.5, label=col)
    #____ plots pop (column 1-3)
    df = data.loc[-30:35]       # limit xscale
    # pop vm
    cols = colsdict['popVm']
    ax = vmaxes[1]
    for i, col in enumerate(cols):
        ax.plot(df[col], color=colors[i], alpha=alpha[i],
                label=col)
    ax.annotate("n=37", xy=(0.2, 0.8),
                xycoords="axes fraction", ha='center')
#    #popVmSig
#    cols = colsdict['popVmSig']
#    ax = vmaxes[2]
#    #traces
#    for i, col in enumerate(cols[:2]):
#        ax.plot(df[col], color=colors[i], alpha=alpha[i],
#                label=col)
#        #errors : iterate on tuples
#    for i, col in enumerate(cols[2:]):
#        for j in [0, 1]:
#            ax.plot(df[col[j]], color=colors[i], alpha=alpha[i],
#                    label=col, linewidth=0.5)
#    ax.annotate("n=10", xy=(0.2, 0.8),
#                xycoords="axes fraction", ha='center')
#    #popVmNsig
#    cols = colsdict['popVmNsig']
#    ax = vmaxes[3]
#    #traces
#    for i, col in enumerate(cols[:2]):
#        ax.plot(df[col], color=colors[i], alpha=alpha[i],
#                label=col)
#    #errors : iterate on tuples
#    for i, col in enumerate(cols[2:]):
#        for j in [0, 1]:
#            ax.plot(df[col[j]], color=colors[i], alpha=alpha[i],
#                    label=col, linewidth=0.5)
#    ax.annotate("n=27", xy=(0.2, 0.8),
#                xycoords="axes fraction", ha='center')
    #pop spike
    cols = colsdict['popSpk']
    ax = spkaxes[1]
    for i, col in enumerate(cols[::-1]):
        ax.plot(df[col], color=colors[::-1][i],
                alpha=1, label=col, linewidth=1)
        ax.fill_between(df.index, df[col],
                        color=colors[::-1][i], alpha=0.5, label=col)
    ax.annotate("n=20", xy=(0.2, 0.8),
                xycoords="axes fraction", ha='center')

    #TODO define a Spk plotmode[lines, allhist, sdFill] for popSpkSig and popSpkNsig
    #TODO extract 1)rows[:2],cols[:2] pannels of fig2 as original pannels
    #             2)rows[:2],cols[3:4] pannels of fig2 as independent pannels  of new fig
#    #popSpkSig
#    cols = colsdict['popSpkSig']
#    ax = spkaxes[2]
#    #traces
#    for i, col in enumerate(cols[:2]):
#        ax.plot(df[col], color=colors[i], alpha=alpha[i],
#                label=col)
#    #errors : iterate on tuples
#    for i, col in enumerate(cols[2:]):
#        for j in [0, 1]:
#            ax.plot(df[col[j]], color=colors[i], alpha=alpha[i],
#                    label=col, linewidth=0.5)
#    ax.annotate("n=5", xy=(0.2, 0.8),
#                xycoords="axes fraction", ha='center')
#    #popSpkNsig
#    cols = colsdict['popSpkNsig']
#    ax = spkaxes[3]
#    #traces
#    for i, col in enumerate(cols[:2]):
#        ax.plot(df[col], color=colors[i], alpha=alpha[i],
#                label=col)
#    #errors : iterate on tuples
#    for i, col in enumerate(cols[2:]):
#        for j in [0, 1]:
#            ax.plot(df[col[j]], color=colors[i], alpha=alpha[i],
#                    label=col, linewidth=0.5)
#    ax.annotate("n=15", xy=(0.2, 0.8),
#                xycoords="axes fraction", ha='center')

    #labels
    for ax in axes:
        for loca in ['top', 'right']:
            ax.spines[loca].set_visible(False)
    ylabels = ['membrane potential (mV)',
               'normalized membrane potential',
               '', '']
    for i, ax in enumerate(vmaxes):
        ax.axes.get_xaxis().set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.set_ylabel(ylabels[i])
    ylabels = ['firing rate (spikes/s)',
               'normalized firing rate',
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
    #lines
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
                         alpha=0.6, edgecolor='w', facecolor='r')
        ax.add_patch(rect)
        #center
    rect = Rectangle(xy=(0, -5), width=step, height=1, fill=True,
                     alpha=0.6, edgecolor='w', facecolor='k')
    ax.add_patch(rect)
    #fit individual example
    vmaxes[0].set_ylim(-4, 13)
    spkaxes[0].set_ylim(-5.5, 20)
    # align zero between plots  NB ref = first plot
    align_yaxis(vmaxes[0], 0, vmaxes[1], 0)
    align_yaxis(spkaxes[0], 0, spkaxes[1], 0)
    # adjust amplitude (without moving the zero)
    change_plot_trace_amplitude(vmaxes[1], 0.85)
    change_plot_trace_amplitude(spkaxes[1], 0.8)
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

fig = plot_half_figure2(data, content)
#%%
plt.close('all')

def plot_1quarter_figure2(data, colsdict):
    """
    plot_figure2 1st quarter
    """
    colors = ['k', stdColors['rouge']]
    alpha = [0.8, 0.8]

    fig = plt.figure(figsize=(6, 10))
    #build axes with sharex and sharey
    axes = []
    for i in range(2):
        axes.append(fig.add_subplot(2, 1, i+1))
    # axes list
    vmaxes = axes[0]      # vm axes = top row
    spkaxes = axes[1]     # spikes axes = bottom row
    #____ plots individuals (first column)
    # individual vm
    cols = colsdict['indVm']
    ax = vmaxes
    for i, col in enumerate(cols):
        ax.plot(data[col], color=colors[i], alpha=alpha[i],
                label=col)
    #individual spike
    cols = colsdict['indSpk']
    ax = spkaxes
    for i, col in enumerate(cols[::-1]):
        ax.plot(data[col], color=colors[::-1][i],
                alpha=1, label=col, linewidth=1)
        ax.fill_between(data.index, data[col],
                        color=colors[::-1][i], alpha=0.5, label=col)
    #____ plots pop (column 1-3)
    #labels
    for ax in axes:
        for loca in ['top', 'right']:
            ax.spines[loca].set_visible(False)
    ylabels = ['Membrane potential (mV)']

    vmaxes.axes.get_xaxis().set_visible(False)
    vmaxes.spines['bottom'].set_visible(False)
    vmaxes.set_ylabel(ylabels[0])
    ylabels = ['Firing rate (spikes/s)']

    spkaxes.set_ylabel(ylabels[0])
    spkaxes.set_xlabel('Time (ms)')

    # stimulations
    step = 28
    xlocs = np.arange(0, -150, -step)
    names = ['D0', 'D1', 'D2', 'D3', 'D4', 'D5']
    dico = dict(zip(names, xlocs))
    #lines
    for ax in [vmaxes, spkaxes]:
        lims = ax.get_ylim()
        for dloc in xlocs:
            ax.vlines(dloc, lims[0], lims[1], linestyle=':', alpha=0.5)
    # stim location
    ax = spkaxes
    for key in dico.keys():
        ax.annotate(key, xy=(dico[key]+3, -3), alpha=0.6, fontsize='small')
        # stim
        rect = Rectangle(xy=(dico[key], -4), width=step, height=1, fill=True,
                         alpha=0.6, edgecolor='w', facecolor='r')
        ax.add_patch(rect)
        #center
    rect = Rectangle(xy=(0, -5), width=step, height=1, fill=True,
                     alpha=0.6, edgecolor='w', facecolor='k')
    ax.add_patch(rect)
    #fit individual example
    vmaxes.set_ylim(-3, 12)
    spkaxes.set_ylim(-5.5, 17.5)
    # zerolines
    for ax in axes:
        lims = ax.get_ylim()
        ax.vlines(0, lims[0], lims[1], alpha=0.3)
        lims = ax.get_xlim()
        ax.hlines(0, lims[0], lims[1], alpha=0.3)
    fig.tight_layout()
    # remove the space between plots
    fig.subplots_adjust(hspace=0.06) #fig.subplots_adjust(hspace=0.02)

    if anot:
        date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        fig.text(0.99, 0.01, 'centrifigs.py:plot_1quarter_figure2',
                 ha='right', va='bottom', alpha=0.4)
    fig.text(0.01, 0.01, date, ha='left', va='bottom', alpha=0.4)

    return fig


fig = plot_1quarter_figure2(data, content)
#%%
plt.close('all')

def plot_2quarter_figure2(data, colsdict):
    """
    plot_figure2 1st quarter
    """
    colors = ['k', stdColors['rouge']]
    alpha = [0.8, 0.8]

    fig = plt.figure(figsize=(6, 10))
    #build axes with sharex and sharey
    axes = []
    for i in range(2):
        axes.append(fig.add_subplot(2, 1, i+1))
    # axes list
    vmaxes = axes[0]      # vm axes = top row
    spkaxes = axes[1]     # spikes axes = bottom row

    #____ plots pop (column 1-3)

    df = data.loc[-30:35]       # limit xscale
    # pop vm
    cols = colsdict['popVm']
    ax = vmaxes
    for i, col in enumerate(cols):
        ax.plot(df[col], color=colors[i], alpha=alpha[i],
                label=col)
    ax.annotate("n=37", xy=(0.2, 0.8),
                xycoords="axes fraction", ha='center', fontsize='large')

    #pop spike
    cols = colsdict['popSpk']
    ax = spkaxes
    for i, col in enumerate(cols[::-1]):
        ax.plot(df[col], color=colors[::-1][i],
                alpha=1, label=col, linewidth=1)
        ax.fill_between(df.index, df[col],
                        color=colors[::-1][i], alpha=0.5, label=col)
    ax.annotate("n=20", xy=(0.2, 0.8),
                xycoords="axes fraction", ha='center', fontsize='large')

    #labels
    for ax in axes:
        for loca in ['top', 'right']:
            ax.spines[loca].set_visible(False)
    ylabels = ['Normalized membrane potential',
               'Normalized firing rate']

    vmaxes.axes.get_xaxis().set_visible(False)
    vmaxes.spines['bottom'].set_visible(False)
    vmaxes.set_ylabel(ylabels[0])
    spkaxes.set_ylabel(ylabels[1])

    vmaxes.set_ylim(-0.1, 1)
    spkaxes.set_ylim(-0.1, 1)
    spkaxes.set_xlabel('Relative time')

    # adjust amplitude (without moving the zero)
    #change_plot_trace_amplitude(vmaxes, 0.85)
    #change_plot_trace_amplitude(spkaxes, 0.8)
    # zerolines
    for ax in axes:
        lims = ax.get_ylim()
        ax.vlines(0, lims[0], lims[1], alpha=0.3)
        lims = ax.get_xlim()
        ax.hlines(0, lims[0], lims[1], alpha=0.3)
    fig.tight_layout()
    # remove the space between plots
    fig.subplots_adjust(hspace=0.06) #fig.subplots_adjust(hspace=0.02)

    if anot:
        date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        fig.text(0.99, 0.01, 'centrifigs.py:plot_half_figure2',
                 ha='right', va='bottom', alpha=0.4)
        fig.text(0.01, 0.01, date, ha='left', va='bottom', alpha=0.4)

    return fig



fig = plot_2quarter_figure2(data, content)
#%% sigNonsig
def plot_signonsig_figure2(data, colsdict, fill=True, fillground=True):
    """
    plot_figure2
    """
    colors = ['k', stdColors['rouge']]
    alpha = [0.8, 0.8]
    # no individual : focus on initial response
    df = data.loc[-30:35]

    fig = plt.figure(figsize=(8, 8)) #fig = plt.figure(figsize=(8, 8))
    #build axes with sharex and sharey
    axes = []
    for i in range(4):
        axes.append(fig.add_subplot(2, 2, i+1))
    # axes list
    vmaxes = axes[:2]      # vm axes = top row
    spkaxes = axes[2:]     # spikes axes = bottom row
    #____ plots individuals (first column)
#    # individual vm
#    cols = colsdict['indVm']
#    ax = vmaxes[0]
#    for i, col in enumerate(cols):
#        ax.plot(data[col], color=colors[i], alpha=alpha[i],
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
#        ax.plot(df[col], color=colors[i], alpha=alpha[i],
#                label=col)
#    ax.annotate("n=37", xy=(0.2, 0.8),
#                xycoords="axes fraction", ha='center')
    #popVmSig
    cols = colsdict['popVmSig']
    ax = vmaxes[-2]
    #ax.set_title('significative population')
    #traces
    for i, col in enumerate(cols[:2]):
        ax.plot(df[col], color=colors[i], alpha=alpha[i],
                label=col)
        #errors : iterate on tuples
    for i, col in enumerate(cols[2:]):
        if fill:
            ax.fill_between(df.index, df[col[0]], df[col[1]], color=colors[i],
                            alpha=alpha[i]/2)
        else:
            for i, col in enumerate(cols[2:]):
                for j in [0, 1]:
                    ax.plot(df[col[j]], color=colors[i], alpha=alpha[i],
                            label=col, linewidth=0.5)
    #ax.annotate("n=10", xy=(0.2, 0.8),
    #            xycoords="axes fraction", ha='center')
    #popVmNsig
    cols = colsdict['popVmNsig']
    ax = vmaxes[-1]
    #ax.set_title('non significative population')
    #traces
    for i, col in enumerate(cols[:2]):
        ax.plot(df[col], color=colors[i], alpha=alpha[i],
                label=col)
    #errors : iterate on tuples
    for i, col in enumerate(cols[2:]):
        if fill:
            ax.fill_between(df.index, df[col[0]], df[col[1]], color=colors[i],
                            alpha=alpha[i]/2)
        else:
            for i, col in enumerate(cols[2:]):
                for j in [0, 1]:
                    ax.plot(df[col[j]], color=colors[i], alpha=alpha[i],
                            label=col, linewidth=0.5)
    #ax.annotate("n=27", xy=(0.2, 0.8),
    #            xycoords="axes fraction", ha='center')
#    #pop spike
#    cols = colsdict['popSpk']
#    ax = spkaxes[1]
#    for i, col in enumerate(cols[::-1]):
#        ax.plot(df[col], color=colors[::-1][i],
#                alpha=1, label=col, linewidth=1)
#        ax.fill_between(df.index, df[col],
#                        color=colors[::-1][i], alpha=0.5, label=col)
#    ax.annotate("n=20", xy=(0.2, 0.8),
#                xycoords="axes fraction", ha='center')

    #TODO define a Spk plotmode[lines, allhist, sdFill] for popSpkSig and popSpkNsig
    #TODO extract 1)rows[:2],cols[:2] pannels of fig2 as original pannels
    #             2)rows[:2],cols[3:4] pannels of fig2 as independent pannels  of new fig

    #NB for spiking response, the control should be plotted last,
    # ie revert the order
    inv_colors = colors[::-1]
    inv_alpha = alpha[::-1]
    #popSpkSig
    cols = colsdict['popSpkSig']
    ax = spkaxes[-2]
    #traces
    for i, col in enumerate(cols[:2][::-1]):
        ax.fill_between(df.index, df[col], color=inv_colors[i],
                        alpha=inv_alpha[i]/2)
        ax.plot(df[col], color=inv_colors[i], alpha=inv_alpha[i],
                label=col, linewidth=2)
    #errors : iterate on tuples
    for i, col in enumerate(cols[2:]):
        for j in [0, 1]:
            ax.plot(df[col[j]], color=colors[i],
                    alpha=1, label=col, linewidth=0.5)
    #ax.annotate("n=5", xy=(0.2, 0.8),
    #            xycoords="axes fraction", ha='center')
    #popSpkNsig
    cols = colsdict['popSpkNsig']
    ax = spkaxes[-1]
    #traces
    for i, col in enumerate(cols[:2][::-1]):
        ax.plot(df[col], color=inv_colors[i], alpha=inv_alpha[i],
                label=col, linewidth=2)
        ax.fill_between(df.index, df[col], color=inv_colors[i],
                        alpha=inv_alpha[i]/2)
    #errors : iterate on tuples
    for i, col in enumerate(cols[2:]):
        for j in [0, 1]:
            ax.plot(df[col[j]], color=colors[i],
                    alpha=1, label=col, linewidth=0.5)
    #ax.annotate("n=15", xy=(0.2, 0.8),
    #            xycoords="axes fraction", ha='center')

    #labels
    for ax in axes:
        for loca in ['top', 'right']:
            ax.spines[loca].set_visible(False)
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
        ax.set_xlabel('Relative time (ms)')

#    # stimulations
#    step = 28
#    xlocs = np.arange(0, -150, -step)
#    names = ['D0', 'D1', 'D2', 'D3', 'D4', 'D5']
#    dico = dict(zip(names, xlocs))
#    #lines
#    for ax in [vmaxes[0], spkaxes[0]]:
#        lims = ax.get_ylim()
#        for dloc in xlocs:
#            ax.vlines(dloc, lims[0], lims[1], linestyle=':', alpha=0.2)
#    # stim location
#    ax = spkaxes[0]
#    for key in dico.keys():
#        ax.annotate(key, xy=(dico[key]+3, -3), alpha=0.6, fontsize='x-small')
#        # stim
#        rect = Rectangle(xy=(dico[key], -4), width=step, height=1, fill=True,
#                         alpha=0.6, edgecolor='w', facecolor='r')
#        ax.add_patch(rect)
#        #center
#    rect = Rectangle(xy=(0, -5), width=step, height=1, fill=True,
#                     alpha=0.6, edgecolor='w', facecolor='k')
#    ax.add_patch(rect)
#    #fis individual example
#    vmaxes[0].set_ylim(-4, 13)
#    spkaxes[0].set_ylim(-5.5, 20)
#    # align zero between plots  NB ref = first plot
#    align_yaxis(vmaxes[0], 0, vmaxes[1], 0)
#    align_yaxis(spkaxes[0], 0, spkaxes[1], 0)
#    # adjust amplitude (without moving the zero)
#    change_plot_trace_amplitude(vmaxes[1], 0.85)
#    change_plot_trace_amplitude(spkaxes[1], 0.8)
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

plot_signonsig_figure2(data, content)
#%% sigNonsig
def plot_sigvm_figure2(data, colsdict, fill=True, fillground=True):
    """
    plot_figure2
    """
    colors = ['k', stdColors['rouge']]
    alpha = [0.8, 0.8]
    # no individual : focus on initial response
    df = data.loc[-30:35]

    fig = plt.figure(figsize=(4, 4)) #fig = plt.figure(figsize=(8, 8))
    #build axes with sharex and sharey
    vmaxes = []

    vmaxes.append(fig.add_subplot(1, 1, 1))
    # axes list

    cols = colsdict['popVmSig']
    ax = vmaxes[0]
    #ax.set_title('significative population')
    #traces
    for i, col in enumerate(cols[:2]):
        ax.plot(df[col], color=colors[i], alpha=alpha[i],
                label=col)
        #errors : iterate on tuples
    for i, col in enumerate(cols[2:]):
        if fill:
            ax.fill_between(df.index, df[col[0]], df[col[1]], color=colors[i],
                            alpha=alpha[i]/2)
        else:
            for i, col in enumerate(cols[2:]):
                for j in [0, 1]:
                    ax.plot(df[col[j]], color=colors[i], alpha=alpha[i],
                            label=col, linewidth=0.5)
    #ax.annotate("n=10", xy=(0.2, 0.8),
    #            xycoords="axes fraction", ha='center')
    ylabels = ['normalized membrane potential']
    for i, ax in enumerate(vmaxes):
        for loca in ['top', 'right', 'bottom']:
            ax.spines[loca].set_visible(False)
        ax.axes.get_xaxis().set_visible(True)
        ax.set_ylabel(ylabels[i])
        ax.set_ylim(-0.10, 1.2)
        ax.set_xlabel('Relative time (ms)')
        lims = ax.get_ylim()
        ax.vlines(0, lims[0], lims[1], alpha=0.2)
        lims = ax.get_xlim()
        ax.hlines(0, lims[0], lims[1], alpha=0.2)
    fig.tight_layout()
    # remove the space between plots
    #fig.subplots_adjust(hspace=0.06) #fig.subplots_adjust(hspace=0.02)
    if anot:
        date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        fig.text(0.99, 0.01, 'centrifigs.py:plot_signonsig_figure2',
                 ha='right', va='bottom', alpha=0.4)
        fig.text(0.01, 0.01, date, ha='left', va='bottom', alpha=0.4)
    return fig

plot_sigvm_figure2(data, content)
#%% NB the length of the sorted data are not the same compared to the other traces
#filename = 'fig2cells.xlsx'
#df = pd.read_excel(filename)

def plot_figure2B(pltmode, sig=True):
    """
    plot_figure2B : ranked phase advance and delta response
    sig=boolan : true <-> shown cell signification
    """
    filename = 'data/fig2cells.xlsx'
    df = pd.read_excel(filename)
    cols = df.columns[:2]
    signs = df.columns[2:]
    df.index += 1 # cells = 1 to 37

    if pltmode == 'horizontal':
        fig = plt.figure(figsize=(8, 3))
    else:
        if pltmode == 'vertical':
            fig = plt.figure(figsize=(6, 6))
    #build axes
    axes = []
    for i in range(2):
        if pltmode == 'horizontal':
            axes.append(fig.add_subplot(1, 2, i+1))
        else:
            if pltmode == 'vertical':
                axes.append(fig.add_subplot(2, 1, i+1))

    color_dic = {0 :'w', 1 : stdColors['rouge']}
    for i, ax in enumerate(axes):
        colors = [color_dic[x] for x in df[signs[i]]]
        if sig:
            axes[i].bar(df.index, df[cols[i]], edgecolor=stdColors['rouge'],
                        color=colors, label=cols[i], alpha=0.8, width=0.8)
        else:
            axes[i].bar(df.index, df[cols[i]], edgecolor=stdColors['rouge'],
                        color=stdColors['rouge'], label=cols[i],
                        alpha=0.8, width=0.8)
        if pltmode == 'horizontal':
            ax.set_xlabel('cell rank')
        else:
            if pltmode == 'vertical':
                if i == 1:
                    ax.set_xlabel('cell rank')
        axes[i].set_xlim(1, 37.7)
        for loca in ['top', 'right', 'bottom']:
            ax.spines[loca].set_visible(False)
        lims = ax.get_xlim()
        ax.hlines(0, lims[0], lims[1], alpha=0.2)
        ticks = [df.index.min(), df.index.max()]
        ax.set_xticks(ticks)
    axes[0].set_ylabel('phase advance (ms)')
    axes[1].set_ylabel('delta response')
    fig.tight_layout()

    if anot:
        date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        fig.text(0.99, 0.01, 'centrifigs.py:plot_figure2B',
                 ha='right', va='bottom', alpha=0.4)
        fig.text(0.01, 0.01, date, ha='left', va='bottom', alpha=0.4)
    if pltmode == 'vertical':
        fig.align_ylabels(axes[0:])
    return fig

#plot_figure2B('horizontal')
plot_figure2B('vertical')
#%%
#plt.close('all')

def plot_figure3(kind):
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
    #samplesize
    cellnumbers = {'pop' : 37, 'sig': 10, 'nonsig': 27}
    ncells = cellnumbers[kind]
    df = pd.read_excel(filenames[kind])
    #centering
    middle = (df.index.max() - df.index.min())/2
    df.index = df.index - middle
    df.index = df.index/10
    cols = ['CENTER-ONLY', 'CP-ISO', 'CF-ISO', 'CP-CROSS', 'RND-ISO']
    df.columns = cols
    colors = ['k', stdColors['rouge'], stdColors['vert'],
              stdColors['jaune'], stdColors['bleu']]
    #alpha = [0.5, 0.8, 0.5, 1, 0.6]
    alpha = [0.8, 0.8, 0.8, 0.8, 0.8]
    fig = plt.figure(figsize=(8, 7))
##SUGGESTION: make y dimension much larger to see maximize visual difference between traces
    #fig.suptitle(titles[kind])
    ax = fig.add_subplot(111)
    for i, col in enumerate(cols):
        #if (i == 0) or (i == 4):
            #ax.plot(df[col], color=colors[i], alpha=alpha[i], label=col, linewidth=2)
        ax.plot(df[col], color=colors[i], alpha=alpha[i], label=col, linewidth=2)
    ax.set_ylabel('Normalized membrane potential')
    ax.set_xlabel('Relative time (ms)')
    for ax in fig.get_axes():
        for loc in ['top', 'right']:
            ax.spines[loc].set_visible(False)
    ax.set_xlim(-15, 30)
    lims = ax.get_ylim()
    ax.vlines(0, lims[0], lims[1], alpha=0.2)
    #ax.vlines(0, -0.2, 1.1, alpha=0.2)
    lims = ax.get_xlim()
    ax.hlines(0, lims[0], lims[1], alpha=0.2)
    ax.set_ylim(-0.2, 1.1)

    #leg = ax.legend(loc='center right', markerscale=None, frameon=False,
    leg = ax.legend(loc=2, markerscale=None, frameon=False,
                    handlelength=0)
    for line, text in zip(leg.get_lines(), leg.get_texts()):
        text.set_color(line.get_color())
    #ax.annotate('n=' + str(ncells), xy=(0.1, 0.8),
    #            xycoords="axes fraction", ha='center')
    fig.tight_layout()

    if anot:
        date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        fig.text(0.99, 0.01, 'centrifigs.py:plot_figure3',
                 ha='right', va='bottom', alpha=0.4)
        fig.text(0.01, 0.01, date, ha='left', va='bottom', alpha=0.4)

    return fig


#fig = plot_figure3('pop')
#fig = plot_figure3('sig')
fig = plot_figure3('nonsig')

#pop all cells
#%% grouped sig and non sig

def plot_figure3_signonsig():
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
    #samplesize
    cellnumbers = {'pop' : 37, 'sig': 10, 'nonsig': 27}
    colors = ['k', stdColors['rouge'], stdColors['vert'],
              stdColors['jaune'], stdColors['bleu']]
    alpha = [0.6, 0.8, 0.5, 1, 0.6]

    fig = plt.figure(figsize=(12, 6))
    axes = []
    for i in range(2):
        axes.append(fig.add_subplot(1, 2, i+1))
    for i, kind in enumerate(['sig', 'nonsig']):
        ncells = cellnumbers[kind]
        df = pd.read_excel(filenames[kind])
        #centering
        middle = (df.index.max() - df.index.min())/2
        df.index = (df.index - middle)/10
        df = df.loc[-15:30]
        cols = ['CNT-ONLY', 'CP-ISO', 'CF-ISO', 'CP_CROSS', 'RND-ISO']
        df.columns = cols

        ax = axes[i]
        ax.set_title(titles[kind])
        ncells = cellnumbers[kind]
        for j, col in enumerate(cols):
            ax.plot(df[col], color=colors[j], alpha=alpha[j],
                    label=col, linewidth=2)
            ax.annotate('n=' + str(ncells), xy=(0.1, 0.8),
                        xycoords="axes fraction", ha='center')
        leg = ax.legend(loc='lower right', markerscale=None, frameon=False,
                        handlelength=0)
        for line, text in zip(leg.get_lines(), leg.get_texts()):
            text.set_color(line.get_color())
    axes[0].set_ylabel('normalized membrane potential')
    for ax in axes:
        ax.set_ylim(-0.1, 1.1)
        lims = ax.get_ylim()
        ax.vlines(0, lims[0], lims[1], alpha=0.2)
        lims = ax.get_xlim()
        ax.hlines(0, lims[0], lims[1], alpha=0.2)
        ax.set_xlabel('relative time (ms)')
        for loc in ['top', 'right']:
            ax.spines[loc].set_visible(False)

    fig.tight_layout()

    if anot:
        date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        fig.text(0.99, 0.01, 'centrifigs.py:plot_figure3_signonsig',
                 ha='right', va='bottom', alpha=0.4)
        fig.text(0.01, 0.01, date, ha='left', va='bottom', alpha=0.4)

    return fig

plot_figure3_signonsig()
#%%
#plt.close('all')
def plot_figure4():
    """ speed """
    filename = 'data/fig4.xlsx'
    df = pd.read_excel(filename)
    #centering
    middle = (df.index.max() - df.index.min())/2
    #OBSERVATION bottom raw 0 baseline has been decentered by police and ticks size changes
    df.index = df.index - middle
    df.index = df.index/10
    cols = ['centerOnly', '100%', '70%', '50%', '30%']
    df.columns = cols
    colors = ['k', stdColors['rouge'], speedColors['orangeFonce'],
              speedColors['orange'], speedColors['jaune']]
    alpha = [0.8, 1, 0.8, 0.8, 1]

    fig = plt.figure(figsize=(8, 6))
   # fig.suptitle(os.path.basename(filename))
    ax = fig.add_subplot(111)
    for i, col in enumerate(cols):
        ax.plot(df[col], color=colors[i], alpha=alpha[i],
                label=col)
    ax.set_ylabel('normalized membrane potential')
    #, fontname = 'Arial', fontsize = 14)
    ax.set_xlabel('relative time (ms)')
    for ax in fig.get_axes():
        for loc in ['top', 'right']:
            ax.spines[loc].set_visible(False)
    fig.tight_layout()
    #fig.legend()
    ax.set_xlim(-40, 45)
    ax.set_ylim(-0.1, 1.1)
    lims = ax.get_ylim()
    ax.vlines(0, lims[0], lims[1], alpha=0.2)
    lims = ax.get_xlim()
    ax.hlines(0, lims[0], lims[1], alpha=0.2)
    leg = ax.legend(loc='upper left', markerscale=None, frameon=False,
                    handlelength=0)
    for line, text in zip(leg.get_lines(), leg.get_texts()):
        text.set_color(line.get_color())
    ax.annotate("population average \n (n=12)", xy=(0.2, 0.8),
                xycoords="axes fraction", ha='left')

    if anot:
        date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        fig.text(0.99, 0.01, 'centrifigs.py:plot_figure4',
                 ha='right', va='bottom', alpha=0.4)
        fig.text(0.01, 0.01, date, ha='left', va='bottom', alpha=0.4)

    return fig

fig = plot_figure4()
## alpha = 0.8, figsize = 8,6, ha = 'left'
#%%
plt.close('all')

def plot_figure5():
    """
    plot_figure5
    """
    filename = 'data/fig5.xlsx'
    df = pd.read_excel(filename)
    #centering
    middle = (df.index.max() - df.index.min())/2
    df.index = df.index - middle
    df.index = df.index/10
    # rename columns
    cols = df.columns
    cols = ['center only', 'surround then center', 'surround only',
            'static linear prediction']
    dico = dict(zip(df.columns, cols))
    df.rename(columns=dico, inplace=True)
    # color parameters
    colors = ['k', stdColors['rouge'], stdColors['bleu'], stdColors['vert']]
    alpha = [0.5, 0.7, 0.8, 0.8]
    #plotting
    fig = plt.figure(figsize=(6, 8))
    # SUGGESTION increase a bit y dimension or subplots height
#    fig.suptitle(os.path.basename(filename))
    ax1 = fig.add_subplot(211)
    for i, col in enumerate(cols[:2]):
        ax1.plot(df.loc[-120:200, [col]], color=colors[i], alpha=alpha[i],
                 label=col)
    ax1.spines['bottom'].set_visible(False)
    ax1.axes.get_xaxis().set_visible(False)

    ax2 = fig.add_subplot(212, sharex=ax1, sharey=ax1)
    for i, col in enumerate(cols):
        ax2.plot(df.loc[-120:200, [col]], color=colors[i], alpha=alpha[i],
                 label=col)

    ax2.set_xlabel('time (ms)')
    # stims
    step = 21
    hlocs = np.arange(0, -110, -step)
    names = ['D0', 'D1', 'D2', 'D3', 'D4', 'D5']
    vlocs = np.linspace(-0.7, -1.6, 4)
#    vlocs = [-0.7, -1, -1.3, -1.6]
    dico = dict(zip(names, hlocs))

    #ax1
    for key in dico.keys():
        #name
        ax1.annotate(key, xy=(dico[key]+3, vlocs[0]), alpha=0.6,
                     annotation_clip=False, fontsize='small')
        #stim1
        rect = Rectangle(xy=(dico[key], vlocs[1]), width=step, height=0.3,
                         fill=True, alpha=0.6, edgecolor='w',
                         facecolor='r')
        ax1.add_patch(rect)
    #center
    rect = Rectangle(xy=(0, vlocs[2]), width=step, height=0.3, fill=True,
                     alpha=0.6, edgecolor='w', facecolor='k')
    ax1.add_patch(rect)

    st = 'surround then center'
    ax1.annotate(st, xy=(30, vlocs[1]), color=colors[1],
                 annotation_clip=False, fontsize='small')
    st = 'center only'
    ax1.annotate(st, xy=(30, vlocs[2]), color=colors[0],
                 annotation_clip=False, fontsize='small')
        # see annotation_clip=False
    ax1.set_ylim(-1.8, 4.5)

    ax2
    for key in dico.keys():
        # names
        ax2.annotate(key, xy=(dico[key]+3, vlocs[0]), alpha=0.6,
                     annotation_clip=False, fontsize='small')
        #stim1
        rect = Rectangle(xy=(dico[key], vlocs[1]), width=step, height=0.3,
                         fill=True, alpha=0.6, edgecolor='w',
                         facecolor=colors[2])
        if key == 'D0':
            rect = Rectangle(xy=(dico[key], vlocs[1]), width=step, height=0.3,
                             fill=True, alpha=0.6, edgecolor=colors[2],
                             facecolor='w')
        ax2.add_patch(rect)
        #stim2
        rect = Rectangle(xy=(dico[key], vlocs[2]), width=step, height=0.3,
                         fill=True, alpha=0.6, edgecolor='w',
                         facecolor=colors[1])
        ax2.add_patch(rect)
    #center
    rect = Rectangle(xy=(0, vlocs[3]), width=step, height=0.3, fill=True,
                     alpha=0.6, edgecolor='w', facecolor=colors[0])
    ax2.add_patch(rect)
    for i, st in enumerate(['surround only', 'surround then center', 'center only']):
        ax2.annotate(st, xy=(30, vlocs[i+1]), color=colors[2-i],
                     annotation_clip=False, fontsize='small')
    for ax in fig.get_axes():
        #leg = ax.legend(loc='upper right', markerscale=None, frameon=False,
        #                handlelength=0)
        # colored text
        #for line, text in zip(leg.get_lines(), leg.get_texts()):
            #text.set_color(line.get_color())
        ax.set_ylabel('membrane potential (mV)')
        for loc in ['top', 'right']:
            ax.spines[loc].set_visible(False)
        lims = ax.get_xlim()
        ax.hlines(0, lims[0], lims[1], alpha=0.2)
        lims = ax.get_ylim()
        ax.vlines(0, lims[0], lims[1], alpha=0.2)
        for dloc in hlocs:
            ax.vlines(dloc, lims[0], lims[1], linestyle=':', alpha=0.2)
    fig.tight_layout()

    if anot:
        date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        fig.text(0.99, 0.01, 'centrifigs.py:plot_figure5',
                 ha='right', va='bottom', alpha=0.4)
        fig.text(0.01, 0.01, date, ha='left', va='bottom', alpha=0.4)

    return fig

fig = plot_figure5()
#%%
plt.close('all')

def plot_figure5half1():
    """
    plot_figure5
    """
    filename = 'data/fig5.xlsx'
    df = pd.read_excel(filename)
    #centering
    middle = (df.index.max() - df.index.min())/2
    df.index = df.index - middle
    df.index = df.index/10
    # rename columns
    cols = df.columns
    cols = ['Center only', 'Surround then center', 'Surround only',
            'Static linear prediction']
    dico = dict(zip(df.columns, cols))
    df.rename(columns=dico, inplace=True)
    # color parameters
    colors = ['k', stdColors['rouge'], stdColors['bleu'], stdColors['vert']]
    #alpha = [0.5, 0.7, 0.8, 0.8]
    alpha = [1, 1, 1, 1]
    #plotting
    fig = plt.figure(figsize=(9, 12))
    # SUGGESTION increase a bit y dimension or subplots height
#    fig.suptitle(os.path.basename(filename))
    ax1 = fig.add_subplot(211)
    for i, col in enumerate(cols[:2]):
        ax1.plot(df.loc[-120:200, [col]], color=colors[i], alpha=alpha[i],
                 label=col)
    ax1.spines['bottom'].set_visible(True)
    ax1.axes.get_xaxis().set_visible(True)
    ax1.set_xlabel('Time (ms)')

    # stims
    step = 21
    hlocs = np.arange(0, -110, -step)
    names = ['D0', 'D1', 'D2', 'D3', 'D4', 'D5']
    vlocs = np.linspace(-0.7, -1.6, 4)
#    vlocs = [-0.7, -1, -1.3, -1.6]
    dico = dict(zip(names, hlocs))

    #ax1
    for key in dico.keys():
        #name
        ax1.annotate(key, xy=(dico[key]+3, vlocs[0]), alpha=1,
                     annotation_clip=False, fontsize='small')
        #stim1
        rect = Rectangle(xy=(dico[key], vlocs[1]), width=step, height=0.3,
                         fill=True, alpha=1, edgecolor='w',
                         facecolor='r')
        ax1.add_patch(rect)
    #center
    rect = Rectangle(xy=(0, vlocs[2]), width=step, height=0.3, fill=True,
                     alpha=1, edgecolor='w', facecolor='k')
    ax1.add_patch(rect)

    st = 'Surround then center'
    ax1.annotate(st, xy=(30, vlocs[1]), color=colors[1],
                 annotation_clip=False, fontsize='small', alpha=1)
    st = 'Center only'
    ax1.annotate(st, xy=(30, vlocs[2]), color=colors[0],
                 annotation_clip=False, fontsize='small', alpha=1)
        # see annotation_clip=False
    ax1.set_ylim(-1.8, 4.5)

    for ax in fig.get_axes():
        leg = ax.legend(loc='upper right', markerscale=None, frameon=False,
                        handlelength=0)
        # colored text
        for line, text in zip(leg.get_lines(), leg.get_texts()):
            text.set_color(line.get_color())
            text.set_alpha(1)
        ax.set_ylabel('membrane potential (mV)')
        for loc in ['top', 'right']:
            ax.spines[loc].set_visible(False)
        lims = ax.get_xlim()
        ax.hlines(0, lims[0], lims[1], alpha=0.2)
        lims = ax.get_ylim()
        ax.vlines(0, lims[0], lims[1], alpha=0.2)
        for dloc in hlocs:
            ax.vlines(dloc, lims[0], lims[1], linestyle=':', alpha=0.5)
    fig.tight_layout()

    if anot:
        date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        fig.text(0.99, 0.01, 'centrifigs.py:plot_figure5half1',
                 ha='right', va='bottom', alpha=0.4)
        fig.text(0.01, 0.01, date, ha='left', va='bottom', alpha=0.4)

    return fig

fig = plot_figure5half1()
#%%
plt.close('all')

def plot_figure5half2():
    """
    plot_figure5
    """
    filename = 'data/fig5.xlsx'
    df = pd.read_excel(filename)
    #centering
    middle = (df.index.max() - df.index.min())/2
    df.index = df.index - middle
    df.index = df.index/10
    # rename columns
    cols = df.columns
    cols = ['Center only', 'Surround then center', 'Surround only',
            'Static linear prediction']
    dico = dict(zip(df.columns, cols))
    df.rename(columns=dico, inplace=True)
    # color parameters
    colors = ['k', stdColors['rouge'], stdColors['rouge'], stdColors['vertSombre']]
    alpha = [0.5, 0.5, 1, 1]
    linewidth = [2, 2, 4, 1]
    #plotting
    fig = plt.figure(figsize=(9, 12))

    ax2 = fig.add_subplot(211)#, sharex=ax1, sharey=ax1)
    for i, col in enumerate(cols):
        ax2.plot(df.loc[-120:200, [col]], color=colors[i], alpha=alpha[i],
                 label=col, linewidth=linewidth[i])

    ax2.set_xlabel('Time (ms)')
    # stims
    step = 21
    hlocs = np.arange(0, -110, -step)
    names = ['D0', 'D1', 'D2', 'D3', 'D4', 'D5']
    vlocs = np.linspace(-0.7, -1.6, 4)
#    vlocs = [-0.7, -1, -1.3, -1.6]
    dico = dict(zip(names, hlocs))

    #ax2
    for key in dico.keys():
        #names
        ax2.annotate(key, xy=(dico[key]+3, vlocs[0]), alpha=1,
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
                         fill=True, alpha=0.5, edgecolor='w',
                         facecolor=colors[1])
        ax2.add_patch(rect)
    # center
    rect = Rectangle(xy=(0, vlocs[3]), width=step, height=0.3, fill=True,
                     alpha=0.5, edgecolor='w', facecolor=colors[0])
    ax2.add_patch(rect)
    for i, st in enumerate(['Surround only', 'Surround then center', 'Center only']):
        ax2.annotate(st, xy=(30, vlocs[i+1]), color=colors[2-i], alpha=alpha[2-i],
                     annotation_clip=False, fontsize='small')
    ax2.set_ylim(-1.8, 4.5)
    for ax in fig.get_axes():
        leg = ax.legend(loc='upper right', markerscale=None, frameon=False,
                        handlelength=0)
        # colored text
        for line, text in zip(leg.get_lines(), leg.get_texts()):
            text.set_color(line.get_color())
            text.set_alpha(line.get_alpha())
        ax.set_ylabel('membrane potential (mV)')
        for loc in ['top', 'right']:
            ax.spines[loc].set_visible(False)
        lims = ax.get_xlim()
        ax.hlines(0, lims[0], lims[1], alpha=0.2)
        lims = ax.get_ylim()
        ax.vlines(0, lims[0], lims[1], alpha=0.2)
        for dloc in hlocs:
            ax.vlines(dloc, lims[0], lims[1], linestyle=':', alpha=0.5)
    fig.tight_layout()

    if anot:
        date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        fig.text(0.99, 0.01, 'centrifigs.py:plot_figure5half2',
                 ha='right', va='bottom', alpha=0.4)
        fig.text(0.01, 0.01, date, ha='left', va='bottom', alpha=0.4)

    return fig

fig = plot_figure5half2()
#%%
def plot_figure6():
    """
    plot_figure6
    """
    filename = 'data/fig6.xlsx'
    df = pd.read_excel(filename)
    #centering
    middle = (df.index.max() - df.index.min())/2
    df.index = df.index - middle
    df.index = df.index/10
    cols = ['centerOnly', 'surroundThenCenter', 'surroundOnly',
            'sdUp', 'sdDown', 'linearPrediction']
    dico = dict(zip(df.columns, cols))
    df.rename(columns=dico, inplace=True)
    cols = df.columns
    colors = ['k', 'r', 'b', 'g', 'b', 'b']
    colors = ['k', stdColors['rouge'], stdColors['bleu'],
              stdColors['violet'], stdColors['violet'], stdColors['violet']]
    alpha = [0.5, 0.5, 0.7, 0.5, 0.5, 0.5]

    fig = plt.figure(figsize=(12, 6))
   # fig.suptitle(os.path.basename(filename))
    ax1 = fig.add_subplot(121)
    for i, col in enumerate(cols[:3]):
        ax1.plot(df[col], color=colors[i], alpha=alpha[i],
                 label=col)
    ax1.set_ylim(-0.2, 1)
    ax2 = fig.add_subplot(122, sharex=ax1)
    for i in [2, 5]:
        print('i=', i, colors[i])
        ax2.plot(df[df.columns[i]], color=colors[i], alpha=alpha[i],
                 label=df.columns[i])
    ax2.fill_between(df.index, df[df.columns[3]], df[df.columns[4]],
                     color=colors[2], alpha=0.2)

    # set fontname and fontsize for y label
    ax1.set_ylabel('normalized membrane potential (mV)')
    #, fontname = 'Arial', fontsize = 14)
    for ax in fig.get_axes():
        leg = ax.legend(loc='upper left', markerscale=None, frameon=False,
                        handlelength=0)
        # colored text
        for line, text in zip(leg.get_lines(), leg.get_texts()):
            text.set_color(line.get_color())
        ax.set_xlim(-150, 150)
        # set fontname and fontsize for x label
        ax.set_xlabel('relative time (ms)')
        #, fontname = 'Arial', fontsize = 14)
        for loc in ['top', 'right']:
            ax.spines[loc].set_visible(False)
        lims = ax.get_xlim()
        ax.hlines(0, lims[0], lims[1], alpha=0.2)
        lims = ax.get_ylim()
        ax.vlines(0, lims[0], lims[1], alpha=0.2)
    # align zero between subplots
    align_yaxis(ax1, 0, ax2, 0)
    fig.tight_layout()
    # add ref
    ref = (0, df.loc[0, ['centerOnly']])

    if anot:
        date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        fig.text(0.99, 0.01, 'centrifigs.py:plot_figure6',
                 ha='right', va='bottom', alpha=0.4)
        fig.text(0.01, 0.01, date, ha='left', va='bottom', alpha=0.4)

    return fig

fig = plot_figure6()

#%% opt
colors = ['k', stdColors['rouge'], speedColors['orangeFonce'],
          speedColors['orange'], speedColors['jaune']]
alpha = [0.8, 1, 0.8, 0.8, 1]

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
        #ax.set_xtickslabels('', minor=False)
    #(manipulate the left_axes list to reorder the plots if required)
    #axes.set_xticklabels(labels, fontdict=None, minor=False)
    #plot left
#    axes = axes[1:].append(axes[0])   # ctrl at the bottom
    cols = df.columns
    for i, ax in enumerate(left_axes):
        ax.plot(df.loc[-140:40, [cols[i]]], color='black', scalex=False,
                scaley=False, label=cols[i])
        ax.fill_between(df.index, df[cols[i]], color=colors[i])
        ax.yaxis.set_ticks(np.arange(-0.15, 0.25, 0.1))
        ax.set_xlim(-140, 40)
        ax.set_ylim(-0.15, 0.25)
    #add labels
    left_axes[3].set_ylabel('Normalized Membrane potential')
    left_axes[0].set_xlabel('Relative time to center-only onset (ms)')
    left_axes[0].xaxis.set_ticks(np.arange(-140, 41, 40))
    ticks = np.arange(-140, 41, 20)
    for i, ax in enumerate(left_axes[1:]):
        ax.set_xticks(ticks, minor=False)
        ax.tick_params(axis='x', labelsize=0)

    #plot right
    for i, col in enumerate(df.columns):
        right_ax.plot(df.loc[40:100, [col]], color=colors[i],
                      label=col, alpha=alpha[i])
        maxi = float(df.loc[30:200, [col]].max())
        right_ax.hlines(maxi, 40, 50, color=colors[i])
    right_ax.set_xlabel('Relative time to center-only onset (ms)')
    # adjust
    for ax in fig.get_axes():
        for loca in ['top', 'right']:
            ax.spines[loca].set_visible(False)
    for ax in left_axes:
        lims = ax.get_ylim()
        ax.vlines(0, lims[0], lims[1], alpha=0.5)
    # adjust spacing
    gs.update(wspace=0.2, hspace=0.05)
    # add ticks to the top
    right_ax.tick_params(axis='x', bottom=True, top=True)
    #legend
    #leg = right_ax.legend(loc='lower right', markerscale=None,
    #                      handlelength=0, framealpha=1)
    #for line, text in zip(leg.get_lines(), leg.get_texts()):
    #    text.set_color(line.get_color())

    fig.tight_layout()
    return fig

    if anot:
        date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        fig.text(0.99, 0.01, 'centrifigs.py:plot_speed_multigraph',
                 ha='right', va='bottom', alpha=0.4)
        fig.text(0.01, 0.01, date, ha='left', va='bottom', alpha=0.4)

fig = plot_speed_multigraph()

#%% test to analyse with x(t) = x(t) - x(t-1)

def plotSpeeddiff():
    colors = ['k', stdColors['rouge'], speedColors['orangeFonce'],
              speedColors['orange'], speedColors['jaune']]
    alpha = [0.5, 1, 0.8, 0.8, 1]

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
                        label=cols[i], alpha=alpha[j])
    lims = ax.get_ylim()
    ax.vlines(0, lims[0], lims[1], alpha=0.2)
    for loca in ['left', 'top', 'right']:
        ax.spines[loca].set_visible(False)
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

#%%
#plt.close('all')

def plot_figSup1(kind):
    """
    plot supplementary figure 1 : Vm with random Sector control
    input : kind in ['pop': whole population, 'sig': individually significants
    cells, 'nonsig': non significant cells]
    """
    filenames = {'pop' : 'data/figSup1.xlsx',
                 'sig': 'data/figSup1bis.xlsx',
                 'nonsig': 'data/figSup1bis2.xlsx'}
    titles = {'pop' : 'all cells',
              'sig': 'individually significant cells',
              'nonsig': 'individually non significants cells'}
    #samplesize
    cellnumbers = {'pop' : 37, 'sig': 10, 'nonsig': 27}
    ncells = cellnumbers[kind]
    df = pd.read_excel(filenames[kind])
    #centering
    middle = (df.index.max() - df.index.min())/2
    df.index = df.index - middle
    df.index = df.index/10
    cols = ['CENTER-ONLY', 'CP-ISO', 'CF-ISO', 'CP-CROSS', 'RND-ISO']
    df.columns = cols
    colors = ['k', stdColors['rouge'], stdColors['vert'],
              stdColors['jaune'], stdColors['bleu']]
    #alpha = [0.5, 0.8, 0.5, 1, 0.6]
    alpha = [0.8, 0.8, 0.8, 0.8, 0.8]
    fig = plt.figure(figsize=(8, 7))
##SUGGESTION: make y dimension much larger to see maximize visual difference between traces
    #fig.suptitle(titles[kind])
    ax = fig.add_subplot(111)
    for i, col in enumerate(cols):
        #if (i == 0) or (i == 4):
            #ax.plot(df[col], color=colors[i], alpha=alpha[i], label=col, linewidth=2)
        ax.plot(df[col], color=colors[i], alpha=alpha[i], label=col, linewidth=2)
    ax.set_ylabel('Normalized membrane potential')
    ax.set_xlabel('Relative time (ms)')
    for ax in fig.get_axes():
        for loc in ['top', 'right']:
            ax.spines[loc].set_visible(False)
    ax.set_xlim(-15, 30)
    lims = ax.get_ylim()
    ax.vlines(0, lims[0], lims[1], alpha=0.2)
    #ax.vlines(0, -0.2, 1.1, alpha=0.2)
    lims = ax.get_xlim()
    ax.hlines(0, lims[0], lims[1], alpha=0.2)
    ax.hlines(0, lims[0], lims[1], alpha=0.2)
    ax.set_ylim(-0.1, 1.1)

    #leg = ax.legend(loc='center right', markerscale=None, frameon=False,
    leg = ax.legend(loc=2, markerscale=None, frameon=False,
                    handlelength=0)
    for line, text in zip(leg.get_lines(), leg.get_texts()):
        text.set_color(line.get_color())
    #ax.annotate('n=' + str(ncells), xy=(0.1, 0.8),
    #            xycoords="axes fraction", ha='center')
    fig.tight_layout()

    if anot:
        date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        fig.text(0.99, 0.01, 'centrifigs.py:plot_figSup1',
                 ha='right', va='bottom', alpha=0.4)
        fig.text(0.01, 0.01, date, ha='left', va='bottom', alpha=0.4)

    return fig


fig = plot_figSup1('pop')
fig = plot_figSup1('sig')
fig = plot_figSup1('nonsig')

#pop all cells
#%%
plt.close('all')

def plot_figSup2(kind):
    """
    plot supplementary figure 2: Vm all conditions of FULL stimulation
    input : kind in ['pop': whole population, 'sig': individually significants
    cells, 'nonsig': non significant cells]
    """
    filenames = {'pop' : 'data/figSup2.xlsx'}#,
                 #'sig': 'data/figSup1bis.xlsx',
                 #'nonsig': 'data/figSup1bis2.xlsx'}
    titles = {'pop' : 'all cells'}#,
              #'sig': 'individually significant cells',
              #'nonsig': 'individually non significants cells'}
    #samplesize
    cellnumbers = {'pop' : 37} #, 'sig': 10, 'nonsig': 27}
    ncells = cellnumbers[kind]
    df = pd.read_excel(filenames[kind])
    #centering
    middle = (df.index.max() - df.index.min())/2
    df.index = df.index - middle
    df.index = df.index/10
    cols = ['CENTER-ONLY', 'CP-ISO', 'CF-ISO', 'CP-CROSS', 'RND-ISO']
    df.columns = cols
    colors = ['k', stdColors['rouge'], stdColors['vert'],
              stdColors['jaune'], stdColors['bleu']]
    alpha = [0.8, 0.8, 0.8, 0.8, 0.8]

    fig = plt.figure(figsize=(6, 10))
    #fig.suptitle(titles[kind])
    #fig, ax = plt.subplots(nrows = 2, ncols = 1, sharex = True, sharey = True, figsize = (8,7))
    ax1 = fig.add_subplot(211)
    for i, col in enumerate(cols):
        if i in (0, 1, 4):
            ax1.plot(df[col], color=colors[i], alpha=alpha[i], label=col, linewidth=2)
    ax1.axes.get_xaxis().set_visible(False)
    ax1.spines['bottom'].set_visible(False)
    #ax1.set_ylabel('Normalized membrane potential')
    ax1.set_ylim(-0.2, 1.1)

    ax2 = fig.add_subplot(212, sharex=ax1, sharey=ax1)
    for i, col in enumerate(cols):
        if i in (0, 1, 3):
            ax2.plot(df[col], color=colors[i], alpha=alpha[i], label=col, linewidth=2)

    ax2.axes.get_xaxis().set_visible(True)
    ax2.spines['bottom'].set_visible(True)
    #ax2.set_ylabel('Normalized membrane potential')
    ax2.set_ylim(-0.2, 1.1)
    ax2.set_xlabel('Relative time (ms)')

#    axes = list(fig.get_axes())
    #leg = ax.legend(loc='center right', markerscale=None, frameon=False,
        #leg = ax.legend(loc=2, markerscale=None, frameon=False,
        #                handlelength=0)
        #for line, text in zip(leg.get_lines(), leg.get_texts()):
        #    text.set_color(line.get_color())
    #ax.annotate('n=' + str(ncells), xy=(0.1, 0.8),
    #            xycoords="axes fraction", ha='center')

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
        fig.text(0.99, 0.01, 'centrifigs.py:plot_figSup2',
                 ha='right', va='bottom', alpha=0.4)
        fig.text(0.01, 0.01, date, ha='left', va='bottom', alpha=0.4)
    return fig

fig = plot_figSup2('pop')
#%%
plt.close('all')

def plot_figSup5(kind, stimmode):
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
            'CENTER-ONLY-FUL', 'CP-ISO-FUL', 'CF-ISO-FUL', 'CP-CROSS-FUL', 'RND-ISO-FUL']
    df.columns = cols
    colors = ['k', stdColors['rouge'], stdColors['vert'],
              stdColors['jaune'], stdColors['bleu']]
    #alpha = [0.5, 0.8, 0.5, 1, 0.6]
    alpha = [0.8, 0.8, 0.8, 0.8, 0.8]
    fig = plt.figure(figsize=(8, 7))
##SUGGESTION: make y dimension much larger to see maximize visual difference between traces
    #fig.suptitle(titles[kind])
    ax = fig.add_subplot(111)
    if stimmode == 'sec':
        for i, col in enumerate(cols[:5]):
            ax.plot(df[col], color=colors[i], alpha=alpha[i], label=col, linewidth=2)
    else:
        if stimmode == 'ful':
            for i, col in enumerate(cols[5:]):
                ax.plot(df[col], color=colors[i], alpha=alpha[i], label=col, linewidth=2)

    ax.set_ylabel('Normalized firing rate')
    ax.set_xlabel('Relative time (ms)')
    for ax in fig.get_axes():
        for loc in ['top', 'right']:
            ax.spines[loc].set_visible(False)
    ax.set_xlim(-15, 30)
    lims = ax.get_ylim()
    ax.vlines(0, lims[0], lims[1], alpha=0.2)
    #ax.vlines(0, -0.2, 1.1, alpha=0.2)
    lims = ax.get_xlim()
    ax.hlines(0, lims[0], lims[1], alpha=0.2)
    ax.set_ylim(-0.1, 1.1)

    #leg = ax.legend(loc='center right', markerscale=None, frameon=False,
    leg = ax.legend(loc=2, markerscale=None, frameon=False,
                    handlelength=0)
    for line, text in zip(leg.get_lines(), leg.get_texts()):
        text.set_color(line.get_color())
    #ax.annotate('n=' + str(ncells), xy=(0.1, 0.8),
    #            xycoords="axes fraction", ha='center')
    fig.tight_layout()
    if anot:
        date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        fig.text(0.99, 0.01, 'centrifigs.py:plot_figSup4',
                 ha='right', va='bottom', alpha=0.4)
        fig.text(0.01, 0.01, date, ha='left', va='bottom', alpha=0.4)

    return fig


#fig = plot_figSup4('pop', 'sec')
fig = plot_figSup5('pop', 'ful')

#fig = plot_figSup1('sig')
#fig = plot_figSup1('nonsig')
#%%
plt.close('all')

def plot_figSup6(kind):
    """
    plot supplementary figure 6: Vm all conditions of surround-only stimulation CP-ISO sig
    input : kind in ['minus': Surround-then-center - Center Only Vs Surround-Only,
    'plus': Surround-Only + Center only Vs Surround-then-center]
    """
    filenames = {'minus' : 'data/figSup6.xlsx',
                 'plus': 'data/figSup6Alt.xlsx'}
                 
    titles = {'minus': 'Surround-then-center minus center only',
              'plus'  : 'Surround-only plus center-only'} 
              
    
    yliminf = {'minus': -0.15,
               'plus': -0.08}
    ylimsup = {'minus': 0.4,
               'plus' : 1.14}          
    
    #samplesize
    cellnumbers = {'minus' : 12, 'plus': 12} 
    ncells = cellnumbers[kind]
    ylimtinf = yliminf[kind]
    ylimtsup = ylimsup[kind]
    df = pd.read_excel(filenames[kind])
    #centering
    middle = (df.index.max() - df.index.min())/2
    df.index = df.index - middle
    df.index = df.index/10
    cols = ['CP-Iso-Stc', 'CP-Iso-Stc-SeUp', 'CP-Iso-Stc-SeDw', 'CP-Iso-Stc-Dlp',
            'CF-Iso-Stc', 'CF-Iso-Stc-SeUp', 'CF-Iso-Stc-SeDw', 'CF-Iso-Stc-Dlp',
            'CP-Cross-Stc', 'CP-Cross-Stc-SeUp', 'CP-Cross-Stc-SeDw', 'CP-Cross-Stc-Dlp',
            'RND-Iso-Stc', 'RND-Iso-Stc-SeUp', 'RND-Iso-Stc-SeDw', 'RND-Iso-Stc-Dlp']
    df.columns = cols
    dico = dict(zip(df.columns, cols))
    df.rename(columns=dico, inplace=True)
    cols = df.columns    
    
    colors = [stdColors['rouge'], stdColors['vert'],
              stdColors['jaune'], stdColors['bleu']]
             
    stdColors1 = {'dark_rouge' : [x/256 for x in [115, 0, 34]],
                  'dark_vert' : [x/256 for x in [10, 146, 13]],
                  'dark_jaune' :	[x/256 for x in [163, 133, 16]],  
                  'dark_bleu' :	[x/256 for x in [14, 73, 118]]}
    colors1 = [stdColors1['dark_rouge'], stdColors1['dark_vert'],
               stdColors1['dark_jaune'], stdColors1['dark_bleu']]
    alpha = [0.7, 0, 0, 0.7]

    fig = plt.figure(figsize=(6, 16))
    
    j = int
    ax1 = fig.add_subplot(411)
    for i in [0,1,2]:
        ax1.plot(df[df.columns[i]], color= colors[i], alpha=alpha[i], 
                 label=df.columns[i], linewidth=2)
    for i in [3]:
        ax1.plot(df[df.columns[i]], color= colors1[i-i], alpha=alpha[i-i],
                 label=df.columns[i], linewidth=2)
    ax1.fill_between(df.index, df[df.columns[2]], df[df.columns[1]],
                     color=colors[0], alpha=0.2)
   
    x = [4,5,6]
    y = [0,1,2]
    zipped = zip(x,y)
    ax2 = fig.add_subplot(412)
    for i,j in zipped:
        ax2.plot(df[df.columns[i]], color= colors[i-i+1], alpha=alpha[j],
                 label=df.columns[i], linewidth=2)
    for i in [7]:
        ax2.plot(df[df.columns[i]], color= colors1[i-i+1], alpha=alpha[0],
                 label=df.columns[i], linewidth=2)
    ax2.fill_between(df.index, df[df.columns[6]], df[df.columns[5]],
                     color=colors[1], alpha=0.2)    
    
    x = [8,9,10]
    y = [0,1,2]
    zipped = zip (x,y)
    ax3 = fig.add_subplot(413)
    for i,j in zipped:
        ax3.plot(df[df.columns[i]], color= colors[i-i+2], alpha=alpha[j],
                 label=df.columns[i], linewidth=2)
    for i in [11]:
        ax3.plot(df[df.columns[i]], color= colors1[i-i+2], alpha=alpha[0],
                 label=df.columns[i], linewidth=2)
    ax3.fill_between(df.index, df[df.columns[10]], df[df.columns[9]],
                     color=colors[2], alpha=0.2)    
    
    x = [12,13,14]
    zipped = zip (x,y)
    ax4 = fig.add_subplot(414)
    for i,j in zipped:
        ax4.plot(df[df.columns[i]], color= colors[i-i+3], alpha=alpha[j],
                 label=df.columns[i], linewidth=2)
    for i in [15]:
        ax4.plot(df[df.columns[i]], color= colors1[i-i+3], alpha=alpha[0],
                 label=df.columns[i], linewidth=2)
    ax4.fill_between(df.index, df[df.columns[14]], df[df.columns[13]],
                     color=colors[3], alpha=0.2)    
    
    
    for ax in fig.get_axes():
        ax.set_xlim(-150, 150)
        ax.set_ylim(ylimtinf, ylimtsup)
        ax.get_xaxis().set_visible(False)
        lims = ax.get_ylim()
        ax.vlines(0, lims[0], lims[1], alpha=0.1)
        lims = ax.get_xlim()
        ax.hlines(0, lims[0], lims[1], alpha=0.1)
        for loc in ['top', 'bottom', 'right']:
            ax.spines[loc].set_visible(False)
   
    ax4.axes.get_xaxis().set_visible(True)   
    ax4.spines['bottom'].set_visible(True)
    ax4.set_xlabel('Relative time (ms)')
    
    ##axes = list(fig.get_axes())
    ##leg = ax.legend(loc='center right', markerscale=None, frameon=False,
        ##leg = ax.legend(loc=2, markerscale=None, frameon=False,
        ##                handlelength=0)
        ##for line, text in zip(leg.get_lines(), leg.get_texts()):
        ##    text.set_color(line.get_color())
    ##ax.annotate('n=' + str(ncells), xy=(0.1, 0.8),
    ##            xycoords="axes fraction", ha='center')

    fig.tight_layout()
    fig.text(-0.04, 0.5, 'Normalized membrane potential', fontsize=16,
             va='center', rotation='vertical')
    # remove the space between plots
    fig.subplots_adjust(hspace=0.1)

    if anot:
        date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        fig.text(0.99, 0.01, 'centrifigs.py:plot_figSup6',
                 ha='right', va='bottom', alpha=0.4)
        fig.text(0.01, 0.01, date, ha='left', va='bottom', alpha=0.4)
    return fig

fig = plot_figSup6('minus')
fig = plot_figSup6('plus')

#%%
plt.close('all')

def plot_figSup7():
    """
    plot supplementary figure 2: Vm all conditions of FULL stimulation
    input : kind in ['pop': whole population, 'sig': individually significants
    cells, 'nonsig': non significant cells]
    """
    filenames = ['data/figSup7a.xlsx', 'data/figSup7b.xlsx'] 
    titles = ['High speed', 'Low speed']
              
    
    filename = filenames[0]
    df = pd.read_excel(filename)
    #centering
    middle = (df.index.max() - df.index.min())/2
    df.index = df.index - middle
    df.index = df.index/10
    cols = ['scp-Iso-Stc-HighSpeed', 'scp-Cross-Stc-HighSpeed']#,
           # 'scp-Cross-Stc-LowSpeed', 'scp-Iso-Stc-LowSpeed']
    df.columns = cols
    dico = dict(zip(df.columns, cols))
    df.rename(columns=dico, inplace=True)
    cols = df.columns        
        
    colors = [stdColors['rouge'], stdColors['jaune']]
    
    alpha = [0.7, 0.7]

    fig = plt.figure(figsize=(6, 10))
    ax1 = fig.add_subplot(211)
    for i, col in enumerate (cols[:2]):
        ax1.fill_between(df.index, df[col], color=colors[i],  alpha=alpha[i], linewidth=2)         
   
    ax1.axes.get_xaxis().set_visible(False)
    ax1.spines['bottom'].set_visible(False)
    ax1.set_ylim(0, 5.5)
    
    
    filename = filenames[1]
    df = pd.read_excel(filename)
    #centering
    middle = (df.index.max() - df.index.min())/2
    df.index = df.index - middle
    df.index = df.index/10
    cols = ['scp-Cross-Stc-LowSpeed', 'scp-Iso-Stc-LowSpeed']
    df.columns = cols
    dico = dict(zip(df.columns, cols))
    df.rename(columns=dico, inplace=True)
    cols = df.columns        
    colors = [stdColors['jaune'], stdColors['rouge']]
    
    ax2 = fig.add_subplot(212)
    for i, col in enumerate (cols[:2]):
        ax2.fill_between(df.index, df[col], color=colors[i],  alpha=alpha[i], linewidth=2)
    
    ax2.axes.get_xaxis().set_visible(True)
    ax2.spines['bottom'].set_visible(True)
    ax2.set_ylim(0, 11.5)
    ax2.set_xlabel('Relative time (ms)')
   
    ax1.annotate('High speed : 100°/s', xy=(0.2, 0.95),
                xycoords="axes fraction", ha='center')

    ax2.annotate('Low speed : 5°/s', xy=(0.2, 0.95),
                 xycoords="axes fraction", ha='center')

    for ax in fig.get_axes():
        ax.set_xlim(-300, 300)
        lims = ax.get_ylim()
        ax.vlines(0, lims[0], lims[1], alpha=0.1)
        lims = ax.get_xlim()
        ax.hlines(0, lims[0], lims[1], alpha=0.1)
        for loc in ['top', 'right']:
            ax.spines[loc].set_visible(False)

    fig.tight_layout()
    fig.text(-0.04, 0.5, 'Normalized membrane potential', fontsize=16,
             va='center', rotation='vertical')
    # remove the space between plots
    fig.subplots_adjust(hspace=0.1)

    if anot:
        date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        fig.text(0.99, 0.01, 'centrifigs.py:plot_figSup2',
                 ha='right', va='bottom', alpha=0.4)
        fig.text(0.01, 0.01, date, ha='left', va='bottom', alpha=0.4)
    return fig

fig = plot_figSup7()
#%% fig supp3 bars

def new_columns_names(cols):
    def convert_to_snake(camel_str):
        """ camel case to snake case """
        temp_list = []
        for letter in camel_str:
            if letter.islower():
                temp_list.append(letter)
            elif letter.isdigit():
                temp_list.append(letter)            
            else:
                temp_list.append('_')
                temp_list.append(letter)
        result = "".join(temp_list)
        return result.lower()
    newcols = [convert_to_snake(item) for item in cols]
    newcols = [item.replace('vms', 'vm_s_') for item in newcols]
    newcols = [item.replace('vmf', 'vm_f_') for item in newcols]
    newcols = [item.replace('spks', 'spk_s_') for item in newcols]
    newcols = [item.replace('spkf', 'spk_f_') for item in newcols]
    return newcols

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
#rename using snake_case
cols = new_columns_names(df.columns)
df.columns = cols
#check stimulations
print_keys(cols)
# build key listing
# ex [['vm'],['s', 'f'],['cp', 'cf', 'rnd'],['iso', 'cross'],['stc'],
#['dlat50', 'dgain50'],['indisig']]
keys = build_keys_list(cols)

#%%
#latency advance
sec_lat = [item for item in cols if '_s_' in item and '_dlat50' in item]
adf = df[sec_lat]

# retriving the numbers:
# latency cpiso
cond = 'vm_s_cp_iso_stc_dlat50'
signi = cond + '_indisig'
mean = adf.loc[adf[signi] > 0, cond].mean()
std = adf.loc[adf[signi] > 0, cond].std()
print(cond, 'mean= %2.2f, std: %2.2f'% (mean, std))
# !!! doesnt suit with the figure !!!
#%%
def extract_values(df, stim_kind='s', measure='lat'):
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
        #cond = rec[0]
        signi = cond + '_indisig'
        pop_num = len(adf)
        signi_num = len(adf.loc[adf[signi]>0, cond])
        percent = round((signi_num / pop_num) * 100)
        leg_cond = cond.split('_')[2] + '-' + cond.split('_')[3]
        pop_dico[leg_cond] = [pop_num, signi_num, percent]
        # descr
        moy = adf.loc[adf[signi]>0, cond].mean()
        stdm = adf.loc[adf[signi]>0, cond].sem()
        resp_dico[leg_cond] = [moy, moy+stdm, moy-stdm]
    return pop_dico, resp_dico

def autolabel(ax, rects):
    # attach some text labels
    for rect in rects:
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2., 1.05*height,
                '%d' % int(height) + '%' ,
                ha='center', va='bottom')

def plot_cell_contribution(df):
    colors = [stdColors['rouge'], stdColors['vert'], 
              stdColors['jaune'], stdColors['bleu']]
    fig = plt.figure(figsize=(8,8))
    ax = fig.add_subplot(221)
    ax.set_title('latency advance (nb cells)')
    stim = 's'
    mes='lat'
    pop_dico, resp_dico = extract_values(df, stim, mes) 
    x = pop_dico.keys()
    height = [pop_dico[item][-1] for item in pop_dico.keys()]
    colors = colors
    bar = ax.bar(x, height, color=colors, width=0.95, alpha=0.8)
    autolabel(ax, bar)
    ax.set_ylabel('SECTOR')
    ax.xaxis.set_visible(False)
    axt = ax.twinx()
    height = [resp_dico[item][0] for item in resp_dico.keys()]
    bar = axt.bar(x, height, edgecolor='k', width=0.1, fc=(1,1,1,0))
    

    ax = fig.add_subplot(222, sharey=ax)
    ax.set_title('delta response (nb cells)')
    stim = 's'
    mes='gain'
    pop_dico, resp_dico = extract_values(df, stim, mes)
    x = pop_dico.keys()
    height = [pop_dico[item][-1] for item in pop_dico.keys()]
    colors = colors
    bar = ax.bar(x, height, color=colors, width=0.95, alpha=0.8)
    autolabel(ax, bar)
    ax.xaxis.set_visible(False)
    axt = ax.twinx()
    height = [resp_dico[item][0] for item in resp_dico.keys()]
    bar = axt.bar(x, height, edgecolor='k', width=0.1, fc=(1,1,1,0))

    ax = fig.add_subplot(223)
    stim = 'f'
    mes='lat'
    pop_dico, resp_dico = extract_values(df, stim, mes)
    x = pop_dico.keys()
    height = [pop_dico[item][-1] for item in pop_dico.keys()]
    colors = colors
    bar = ax.bar(x, height, color=colors, width=0.95, alpha=0.8)
    autolabel(ax, bar)
    ax.set_ylabel('FULL')
    axt = ax.twinx()
    height = [resp_dico[item][0] for item in resp_dico.keys()]
    bar = axt.bar(x, height, edgecolor='k', width=0.1, fc=(1,1,1,0))

    ax = fig.add_subplot(224, sharey=ax)
    stim = 'f'
    mes='gain'
    pop_dico, resp_dico = extract_values(df, stim, mes)
    x = pop_dico.keys()
    height = [pop_dico[item][-1] for item in pop_dico.keys()]
    colors = colors
    bar = ax.bar(x, height, color=colors, width=0.95, alpha=0.8)
    autolabel(ax, bar)
    axt = ax.twinx()
    height = [resp_dico[item][0] for item in resp_dico.keys()]
    bar = axt.bar(x, height, edgecolor='k', width=0.1, fc=(1,1,1,0))

    for ax in fig.get_axes():
        for loca in ['left', 'top', 'right']:
            ax.spines[loca].set_visible(False)
            ax.tick_params(axis='x', labelrotation=45)
            ax.yaxis.set_ticklabels([])
            ax.tick_params(axis = 'y', length=0)
    fig.tight_layout()

plot_cell_contribution(df)