

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
#import math

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
             'violet' : [x/256 for x in [255, 0, 255]]}
speedColors = {'orangeFonce' :     [x/256 for x in [252, 98, 48]],
               'orange' : [x/256 for x in [253, 174, 74]],
               'jaune' : [x/256 for x in [254, 226, 137]]}
# define the font size to be used
params = {'font.sans-serif': ['Arial'],
          'font.size': 14,
          'legend.fontsize': 'small',
          'figure.figsize': (15, 5),
          'axes.labelsize': 'small',
          'axes.titlesize':'small',
          'xtick.labelsize':'small',
          'ytick.labelsize':'small',
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
                ax.fill_between(df.index, df[col[0]], df[col[1]], color=colors[i], 
                                alpha=alpha[i]/2) 
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
                                color=colors[i], alpha=alpha[i]/2) 
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
    #TODO extract 1)rows[:2],cols[:2] pannels of fig2 as original pannels
    #             2)rows[:2],cols[3:4] pannels of fig2 as independent pannels  of new fig
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
    return fig



fig = plot_half_figure2(data, content)

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
    ax.set_title('significative population')
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
    ax.annotate("n=10", xy=(0.2, 0.8),
                xycoords="axes fraction", ha='center')
    #popVmNsig
    cols = colsdict['popVmNsig']
    ax = vmaxes[-1]
    ax.set_title('non significative population')
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
    ax.annotate("n=27", xy=(0.2, 0.8),
                xycoords="axes fraction", ha='center')
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
    ax.annotate("n=5", xy=(0.2, 0.8),
                xycoords="axes fraction", ha='center')
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
    ax.annotate("n=15", xy=(0.2, 0.8),
                xycoords="axes fraction", ha='center')

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
        ax.set_xlabel('relative time (ms)')

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
    return fig

plot_signonsig_figure2(data, content)
#%% NB the length of the sorted data are not the same compared to the other traces
#filename = 'fig2cells.xlsx'
#df = pd.read_excel(filename)

def plot_figure2B(sig=True):
    """
    plot_figure2B : ranked phase advance and delta response
    sig=boolan : true <-> shown cell signification
    """
    filename = 'data/fig2cells.xlsx'
    df = pd.read_excel(filename)
    cols = df.columns[:2]
    signs = df.columns[2:]
    df.index += 1 # cells = 1 to 37

    fig = plt.figure(figsize=(8, 2))
    #build axes
    axes = []
    for i in range(2):
        axes.append(fig.add_subplot(1, 2, i+1))
    color_dic = {0 :'w', 1 : stdColors['rouge']}
    for i, ax in enumerate(axes):
        colors = [color_dic[x] for x in df[signs[i]]]
        if sig :
            axes[i].bar(df.index, df[cols[i]], edgecolor=stdColors['rouge'],
                    color=colors, label=cols[i], alpha=0.8, width=0.8)
        else:
            axes[i].bar(df.index, df[cols[i]], edgecolor=stdColors['rouge'],
                    color=stdColors['rouge'], label=cols[i], 
                    alpha=0.8, width=0.8)            
        ax.set_xlabel('cell rank')
        for loca in ['top', 'right', 'bottom']:
            ax.spines[loca].set_visible(False)
        lims = ax.get_xlim()
        ax.hlines(0, lims[0], lims[1], alpha=0.2)
        ticks = [df.index.min(), df.index.max()]
        ax.set_xticks(ticks)
    axes[0].set_ylabel('phase advance (ms)')
    axes[1].set_ylabel('delta response')
    fig.tight_layout()
    return fig

plot_figure2B()

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
    titles = {'pop' : 'recorded cells',
              'sig': 'individually significant cells',
              'nonsig': 'individually non significants cells'}
    #samplesize
    cellnumbers= {'pop' : 37, 'sig': 10, 'nonsig': 27}
    ncells = cellnumbers[kind]
    df = pd.read_excel(filenames[kind])
    #centering
    middle = (df.index.max() - df.index.min())/2
    df.index = df.index - middle
    df.index = df.index/10
    cols = ['CNT-ONLY', 'CP-ISO', 'CF-ISO', 'CP_CROSS', 'RND-ISO']
    df.columns = cols
    colors = ['k', stdColors['rouge'], stdColors['vert'],
              stdColors['jaune'], stdColors['bleu']]
    alpha = [0.6, 0.8, 0.5, 1, 0.6]

    fig = plt.figure(figsize=(8, 7))
##SUGGESTION: make y dimension much larger to see maximize visual difference between traces
    fig.suptitle(titles[kind])
    ax = fig.add_subplot(111)
    for i, col in enumerate(cols):
        ax.plot(df[col], color=colors[i], alpha=alpha[i], label=col, linewidth=2)
    ax.set_ylabel('normalized membrane potential')
    ax.set_xlabel('relative time (ms)')
    for ax in fig.get_axes():
        for loc in ['top', 'right']:
            ax.spines[loc].set_visible(False)
    ax.set_xlim(-15, 30)
    lims = ax.get_ylim()
    ax.vlines(0, lims[0], lims[1], alpha=0.2)
    lims = ax.get_xlim()
    ax.hlines(0, lims[0], lims[1], alpha=0.2)

    leg = ax.legend(loc='center right', markerscale=None, frameon=False,
                    handlelength=0)
    for line, text in zip(leg.get_lines(), leg.get_texts()):
        text.set_color(line.get_color())
    ax.annotate('n=' + str(ncells), xy=(0.1, 0.8),
                xycoords="axes fraction", ha='center')
    fig.tight_layout()
    return fig


fig = plot_figure3('pop')
fig = plot_figure3('sig')
fig = plot_figure3('nonsig')


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
    cellnumbers= {'pop' : 37, 'sig': 10, 'nonsig': 27}
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
    cols = ['centerOnly', '100%', '70%', '80%', '50%']
    df.columns = cols
    colors = ['k', stdColors['rouge'], speedColors['orangeFonce'],
              speedColors['orange'], speedColors['jaune']]
    alpha = [0.5, 1, 0.8, 0.8, 1]

    fig = plt.figure(figsize=(8, 4))
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
    leg = ax.legend(loc='lower right', markerscale=None, frameon=False,
                    handlelength=0)
    for line, text in zip(leg.get_lines(), leg.get_texts()):
        text.set_color(line.get_color())
    ax.annotate("population average \n (n=12)", xy=(0.2, 0.8),
                xycoords="axes fraction", ha='center')
    return fig

fig = plot_figure4()

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

    #ax2
    for key in dico.keys():
        # names
        ax2.annotate(key, xy=(dico[key]+3, vlocs[0]), alpha=0.6,
                     annotation_clip=False, fontsize='small')
        # stim1
        rect = Rectangle(xy=(dico[key], vlocs[1]), width=step, height=0.3,
                         fill=True, alpha=0.6, edgecolor='w',
                         facecolor=colors[2])
        if key == 'D0':
            rect = Rectangle(xy=(dico[key], vlocs[1]), width=step, height=0.3,
                             fill=True, alpha=0.6, edgecolor=colors[2],
                             facecolor='w')
        ax2.add_patch(rect)
        # stim2
        rect = Rectangle(xy=(dico[key], vlocs[2]), width=step, height=0.3,
                         fill=True, alpha=0.6, edgecolor='w',
                         facecolor=colors[1])
        ax2.add_patch(rect)
    # center
    rect = Rectangle(xy=(0, vlocs[3]), width=step, height=0.3, fill=True,
                     alpha=0.6, edgecolor='w', facecolor=colors[0])
    ax2.add_patch(rect)
    for i, st in enumerate(['surround only', 'surround then center', 'center only']):
        ax2.annotate(st, xy=(30, vlocs[i+1]), color=colors[2-i],
                     annotation_clip=False, fontsize='small')
    for ax in fig.get_axes():
        leg = ax.legend(loc='upper right', markerscale=None, frameon=False,
                        handlelength=0)
        # colored text
        for line, text in zip(leg.get_lines(), leg.get_texts()):
            text.set_color(line.get_color())
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
    return fig

fig = plot_figure5()

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
    alpha = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5]

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
    plot the speed effect of centirgabor protocol
    """
    fig = plt.figure(figsize=(12, 8))
    fig.suptitle('aligned on Center-Only stimulus onset (t=0)')
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
    #(manipulate the left_axes list to reorder the plots if required)

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
    left_axes[2].set_ylabel('Normalized Membrane potential')
    left_axes[-1].set_xlabel('Relative time to center-only onset (ms)')
    left_axes[-1].xaxis.set_ticks(np.arange(-140, 41, 20))
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
    leg = right_ax.legend(loc='lower right', markerscale=None,
                          handlelength=0, framealpha=1)
    for line, text in zip(leg.get_lines(), leg.get_texts()):
        text.set_color(line.get_color())

    fig.tight_layout()
    return fig

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
    return fig

fig = plotSpeeddiff()
