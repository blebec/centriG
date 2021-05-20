#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from datetime import datetime
from importlib import reload

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config
import general_functions as gfunc
import load.load_traces as ltra
import old.old_figs as ofig

plt.rcParams.update(config.rc_params())
paths = config.build_paths()
std_colors = config.std_colors()
anot = True

paths['save'] = os.path.join(paths['owncFig'], 'pythonPreview', 'proposal')

#%%

plt.close('all')

def plot_figure3(datadf, stdcolors, **kwargs):
    """
    plot_figure3
    input :
        datadf (nb cols = ctr then 'kind_rec_spread_dir_or')
        stdcolors
        kwargs in (first value = default)
            substract boolan -> present as (data - centerOnly)(False),
            anot boolan -> add title and footnote (True)
            age -> data reference (new, old),
            addleg -> boolan to add legend (False)
            addinsert -> boolean to add insert (False)
    output:
        plt.figure()
    """
    # defined by kwargs
    substract = kwargs.get('substract', False)
    anot = kwargs.get('anot', True)
    age = kwargs.get('age', 'new')
    addleg = kwargs.get('addleg', False)
    addinsert = kwargs.get('addinsert', False)
    filename = kwargs.get('file', '')
    #defined in dataframe columns (first column = ctr))
    kind, rec, spread,  *_ = data_df.columns.to_list()[1].split('_')
    titles = dict(pop='all cells',
                  sig='individually significant cells',
                  nsig='individually non significants cells')
    # centering
    df = datadf.copy()
    middle = (df.index.max() - df.index.min())/2
    df.index = (df.index - middle)/10
    # cols = ['CENTER-ONLY', 'CP-ISO', 'CF-ISO', 'CP-CROSS', 'RND-ISO']
    colors = [stdcolors[st] for st in 
              ['k', 'red', 'green', 'yellow', 'blue', 'blue']]
    alphas = [0.8, 1, 0.8, 0.8, 0.8, 0.8]
    # subtract the centerOnly response (ref = df['CENTER-ONLY'])
    if substract:
        ref = df[df.columns[0]]
        df = df.subtract(ref, axis=0)
    # remove rdsect
    cols = df.columns.to_list()
    while any(st for st in cols if 'sect_rd' in st):
        cols.remove(next(st for st in cols if 'sect_rd' in st))
    #buils labels
    labels = cols[:]
    labels = [n.replace('full_rd_', 'full_rdf_') for n in labels]
    for i in range(3):
        for item in labels:
            if len(item.split('_')) < 6:
                j = labels.index(item)
                labels[j] = item + '_ctr'
    labels = [st.split('_')[-3] for st in labels]
    #plot
    fig = plt.figure(figsize=(6.5, 5.5))
    fig.suptitle(titles[kind], alpha=0.4)
    if anot:
        title = '{} {} ({} {})'.format(kind, age, rec, spread)
        fig.suptitle(title, alpha=0.4)
    ax = fig.add_subplot(111)
    for i, col in enumerate(cols):
        ax.plot(df[col], color=colors[i], alpha=alphas[i], label=labels[i],
                linewidth=2)
    # bluePoint
    x = 0
    y = df.loc[0][df.columns[0]]
    vspread = .02  # vertical spread for realign location
    # ax.plot(x, y, 'o', color='tab:gray', ms=10, alpha=0.5)
    ax.vlines(x, y + vspread, y - vspread, linewidth=4, color='tab:gray')
    ax.axvline(x, linewidth=2, color='tab:blue', linestyle=':')
    # ax.plot(0, df.loc[0][df.columns[0]], 'o', color=colors[0],
    #         ms=10, alpha=0.5)
    #refs
    ax.axvline(0, alpha=0.3)
    ax.axhline(0, alpha=0.2)
    #labels
    ax.set_ylabel('Normalized membrane potential')
    ax.set_xlabel('Relative time (ms)')
    for ax in fig.get_axes():
        for loc in ['top', 'right']:
            ax.spines[loc].set_visible(False)
    #set limits
    ax.set_xlim(-15, 30)
    ax.set_ylim(-0.2, 1.1)
    custom_ticks = np.arange(0, 1.1, 0.2)
    ax.set_yticks(custom_ticks)
    custom_ticks = np.arange(-10, 31, 10)
    ax.set_xticks(custom_ticks)
    # ax.annotate('n=' + str(ncells), xy=(0.1, 0.8),
    #             xycoords="axes fraction", ha='center')
    # insert subplot inside this one (broader x axis)
    if addinsert:
        axins = fig.add_axes([.5, .21, .42, .25], facecolor='w', alpha=0.2)
        for i, col in enumerate(cols):
            if i in [0, 4]:
                axins.plot(df[col], color=colors[i], alpha=alphas[i], label=col,
                           linewidth=2)
        axins.set_xlim(-150, 30)
        for spine in ['left', 'top']:
            axins.spines[spine].set_visible(False)
        axins.yaxis.tick_right()
        axins.patch.set_edgecolor('w')
        axins.patch.set_alpha(0)
        axins.axvline(0, alpha=0.3)
    # siubtract the centerOnly
    if substract:
        ax.set_xlim(-45, 120)
        ax.set_ylim(-0.15, 0.4)
        custom_ticks = np.arange(-40, 110, 20)
        ax.set_xticks(custom_ticks)
        # max_x center only
        ax.axvline(21.4, alpha=0.5, color='k')
        # end_x of center only
        #(df['CENTER-ONLY'] - 0.109773).abs().sort_values().head()
        ax.axvline(88, alpha=0.3)
        ax.axvspan(0, 88, facecolor='k', alpha=0.2)

        ax.text(0.45, 0.9, 'center only response \n start | peak | end',
                transform=ax.transAxes, alpha=0.5)
        ax.set_ylabel('Norm vm - Norm centerOnly')
    fig.tight_layout()

    if anot:
        if addleg:
            ax.legend()
            fig.text(0.11, 0.91, filename)
        date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        fig.text(0.99, 0.01, 'pop_traces.py:plot_figure3(' + kind + ')',
                 ha='right', va='bottom', alpha=0.4)
        fig.text(0.01, 0.01, date, ha='left', va='bottom', alpha=0.4)
    return fig


#params (load)
select = dict(age='new', rec='vm', kind='sig')
# select['align'] = 'p2p'

data_df, file = ltra.load_intra_mean_traces(paths, **select)
#params (plot)
select['file'] = file
select['substract'] = True
select['addleg'] = False
select['addinsert'] = False
fig = plot_figure3(data_df, std_colors, **select)

# to use with fillingf in:
ax = fig.get_axes()[0]
ax.set_xlim(-150, 150)
ax.set_ylim(-0.15, 0.35)
ax.set_xticks(np.linspace(-150, 150, 7))
ax.set_yticks(np.linspace(-0.1, 0.3, 5))

save = False
if save:
    dirname = os.path.join(paths['owncFig'],
                            'pythonPreview', 'fillingIn', 'indFill_popFill')
    file = 'pop_substraction.png'
    fig.savefig(os.path.join(dirname, file))

#%% nb use select to change the parameters
plt.close('all')


figs = []
for kind in ['pop', 'sig']:
    select['age'] = 'new'
    select['kind'] = kind
    select['leg'] = False
    data_df, _ = ltra.load_intra_mean_traces(paths, **select)
    #no data with 'pop' and 'p2p'
    if len(data_df) > 0:
        for substract in [True, False]:
            select['substract'] = substract
            figs.append(plot_figure3(data_df, std_colors, **select))
lims = [0, 0]
for fig in figs:
    lim = fig.get_axes()[0].get_ylim()
    if lim[0] < lims[0]:
        lims[0] = lim[0]
    if lim[1] > lims[1]:
        lims[1] = lim[1]
for fig in figs:
    fig.get_axes()[0].set_ylim(lims)


#fig = plot_figure3('nsig')
#fig = plot_figure3(std_colors, kind='sig', substract=True, age='old')
#fig2 = plot_figure3(std_colors, kind='pop', substract=True, age='old')

#pop all cells
#%% grouped sig and non sig
plt.close('all')
fig1 = ofig.plot_3_signonsig(std_colors, anot=anot)
fig2 = ofig.plot_3_signonsig(std_colors, substract=True, anot=anot)

#%% same scale + extend
for fig in [fig1, fig2]:
    for ax in fig.get_axes():
        ax.set_ylim(-0.2, 1.1)
        ax.set_xlim(-45, 120)

#%%all conditions
plt.close('all')
for age in ['new']: #, 'old']:
    for kind in ['pop', 'sig', 'nsig']:
        for rec in ['vm', 'spk']:
            for spread in ['sect', 'full']:
                # print('______')
                # print(kind, age, rec, spread)
                df, f = ltra.load_intra_mean_traces(paths, kind=kind, age=age,
                                                    rec=rec, spread=spread)
                plot_figure3(df, std_colors, kind=kind, age=age,
                             rec=rec, spread=spread)

#%%
plt.close('all')
#%%
dico = dict(
    age=['old', 'new'][1],
    kind=['pop', 'sig', 'nsig'][1],
    spread=['sect', 'full'][0],
    rec=['vm', 'spk'][0]
    )
peak = False
if peak:
    dico['align'] = 'p2p'

dico['kind'] = 'pop'
df1, f1 = ltra.load_intra_mean_traces(paths, **dico)
if len(df1) > 0:
    fig1 = plot_figure3(df1, std_colors, **dico)

dico['kind'] = 'sig'
df2, f2 = ltra.load_intra_mean_traces(paths, **dico)
if len(df2) > 0:
    fig2 = plot_figure3(df2, std_colors, **dico)

dico['kind'] = 'nsig'
df3, f3 = ltra.load_intra_mean_traces(paths, **dico)
if len(df3) > 0:
    fig3 = plot_figure3(df3, std_colors, **dico)

#%%
plt.close('all')

def plot_trace2x2(dflist, stdcolors, **kwargs):
    """
    plot_figure3
    input :
        datadf (nb cols = ctr then 'kind_rec_spread_dir_or')
        stdcolors
        kwargs in (first value = default)
            substract boolan -> present as (data - centerOnly)(False),
            anot boolan -> add title and footnote (True)
            age -> data reference (new, old),
            addleg -> boolan to add legend (False)
            addinsert -> boolean to add insert (False)
    output:
        plt.figure()
    """
    # defined by kwargs
    substract = kwargs.get('substract', False)
    anot = kwargs.get('anot', True)
    age = kwargs.get('age', 'new')
    addleg = kwargs.get('addleg', False)
    addinsert = kwargs.get('addinsert', False)
    controls = kwargs.get('controls', True)
    #defined in dataframe columns (first column = ctr))
    titles = dict(pop='all cells',
                  sig='individually significant cells',
                  nsig='individually non significants cells')

    # cols = ['CENTER-ONLY', 'CP-ISO', 'CF-ISO', 'CP-CROSS', 'RND-ISO']
    colors = [stdcolors[color] for color in 'red green yellow blue blue'.split()]
    colors.insert(0, [0,0,0])
    alphas = [0.8, 1, 0.8, 0.8, 0.8, 0.8]

    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(17.6, 17.6),
                             sharex=True, sharey=True)
    axes = axes.flatten()
    #fig.suptitle(titles[kind], alpha=0.4)
    # if anot:
    #     title = '{} {} ({} {})'.format(kind, age, rec, spread)
    #     fig.suptitle(title, alpha=0.4)
    dico = kwargs
    dico['age'] = ['old', 'new'][1]
    dico['kind'] = ['pop', 'sig', 'nsig'][1]

    spreads = ['sect', 'full']*2
    recs = ['vm']*2 + ['spk']*2
    for i, ax in enumerate(axes):
        dico['spread'] = spreads[i]
        dico['rec'] = recs[i]
        ax_title = f"{dico['rec']} {dico['spread']}"
        ax.set_title(ax_title)
        #data
        datadf, file = ltra.load_intra_mean_traces(paths, **dico)
        # centering
        df = datadf.copy()
        middle = (df.index.max() - df.index.min())/2
        df.index = (df.index - middle)/10
        # subtract the centerOnly response (ref = df['CENTER-ONLY'])
        if substract:
            ref = df[df.columns[0]]
            df = df.subtract(ref, axis=0)
        # remove rdsect
        cols = df.columns.to_list()
        while any(st for st in cols if 'sect_rd' in st):
            cols.remove(next(st for st in cols if 'sect_rd' in st))
        #build labels
        labels = cols[:]
        labels = [n.replace('full_rd_', 'full_rdf_') for n in labels]
        for i in range(3):
            for item in labels:
                if len(item.split('_')) < 6:
                    j = labels.index(item)
                    labels[j] = item + '_ctr'
        labels = [st.split('_')[-3] for st in labels]
        #plot
        ax.text(0.06, 0.91, file, transform=ax.transAxes,
                horizontalalignment='left', alpha=0.4)
        # with controls:
        if controls:
            for i, col in enumerate(cols):
                ax.plot(df[col], color=colors[i], alpha=alphas[i], label=labels[i],
                        linewidth=2)
        else:
            for i, col in enumerate(cols[:2]):
                ax.plot(df[col], color=colors[i], alpha=alphas[i], label=labels[i],
                        linewidth=2)
        # bluePoint
        x = 0
        y = df.loc[0][df.columns[0]]
        vspread = .06  # vertical spread for realign location
        # ax.plot(x, y, 'o', color='tab:gray', ms=10, alpha=0.5)
        ax.vlines(x, y + vspread, y - vspread, linewidth=4, color='tab:gray')
        ax.axvline(x, linewidth=2, color='tab:blue', linestyle=':')

        #refs
#        ax.axvline(0, alpha=0.3)
        ax.axhline(0, alpha=0.2)
        #labels
        for loc in ['top', 'right']:
            ax.spines[loc].set_visible(False)
        # if i == 0:
        #     ax.set_ylabel('Normalized membrane potential')
        # if i == 2:
        #     ax.set_ylabel('Normalized firing rate')
        # if i > 1:

        if substract:
            ax.set_xlim(-45, 120)
            ax.set_ylim(-0.15, 0.4)
            custom_ticks = np.arange(-40, 110, 20)
            ax.set_xticks(custom_ticks)
            # max_x center only
            ax.axvline(21.4, alpha=0.5, color='k')
            # end_x of center only
            #(df['CENTER-ONLY'] - 0.109773).abs().sort_values().head()
            ax.axvline(88, alpha=0.3)
            ax.axvspan(0, 88, facecolor='k', alpha=0.2)

            ax.text(0.45, 0.9, 'center only response \n start | peak | end',
                    transform=ax.transAxes, alpha=0.5)
            ax.set_ylabel('Norm vm - Norm centerOnly')

        else:
            axes[2].set_xlabel('Relative time (ms)')
            axes[3].set_xlabel('Relative time (ms)')
            axes[0].set_ylabel('Normalized membrane potential')
            axes[2].set_ylabel('Normalized firing rate')
            #set limits
            ax.set_xlim(-30, 50)
            ax.set_ylim(-0.2, 1.1)
            custom_ticks = np.arange(0, 1.1, 0.2)
            ax.set_yticks(custom_ticks)
            custom_ticks = np.arange(-20, 45, 10)
            ax.set_xticks(custom_ticks)
            # ax.annotate('n=' + str(ncells), xy=(0.1, 0.8),
            # xycoords="axes fraction", ha='center')
        if addinsert:
            # insert subplot inside this one (broader x axis)
            axins = ax.inset_axes([.4, .11, .42, .25], facecolor='w', alpha=0.2)
            for i, col in enumerate(cols):
                if i in [0, 4]:
                    axins.plot(df[col], color=colors[i], alpha=alphas[i], label=col,
                               linewidth=2)
                axins.set_xlim(-150, 30)
            for spine in ['left', 'top']:
                axins.spines[spine].set_visible(False)
            axins.yaxis.tick_right()
            axins.patch.set_edgecolor('w')
            axins.patch.set_alpha(0)
            axins.axvline(0, alpha=0.3)
    fig.tight_layout()
    if anot:
        if addleg:
            for ax in axes:
                ax.legend()
        date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        fig.text(0.99, 0.01, 'pop_traces.py:plot_trace2x2',
                 ha='right', va='bottom', alpha=0.4)
        fig.text(0.01, 0.01, date, ha='left', va='bottom', alpha=0.4)
    return fig


dico = dict(
    age=['old', 'new'][1],
    kind=['pop', 'sig', 'nsig'][2],
    spread=['sect', 'full'][0],
    rec=['vm', 'spk'][1],
    anot = True,
    addleg = False,
    addinsert = False,
    substract = False,
    controls = [True, False][0]
    )

peak = False
if peak:
    dico['align'] = 'p2p'

fig = plot_trace2x2([], std_colors, **dico)
save = False
if save:
    fig.savefig(os.path.join(paths['save'], 'plot_trace2x2.png'))


#%%



def plot_trace_1x2(stdcolors, **kwargs):
    """
    plot_figure3
    input :
        datadf (nb cols = ctr then 'kind_rec_spread_dir_or')
        stdcolors
        kwargs in (first value = default)
            substract boolan -> present as (data - centerOnly)(False),
            anot boolan -> add title and footnote (True)
            age -> data reference (new, old),
            addleg -> boolan to add legend (False)
            addinsert -> boolean to add insert (False)
    output:
        plt.figure()
    """
    # defined by kwargs
    substract = kwargs.get('substract', False)
    anot = kwargs.get('anot', True)
    addleg = kwargs.get('addleg', False)
    addinsert = kwargs.get('addinsert', False)
    controls = kwargs.get('controls', True)
    spread = kwargs.get('spread', 'sect')
    #defined in dataframe columns (first column = ctr))

    title = "significant cells, (time U energy U filling-in)"
    # cols = ['CENTER-ONLY', 'CP-ISO', 'CF-ISO', 'CP-CROSS', 'RND-ISO']
    colors = [stdcolors[color]
              for color in 'red green yellow blue blue'.split()]
    colors.insert(0, [0,0,0]) # black
    alphas = [0.8, 1, 0.8, 0.8, 0.8, 0.8]

    #data
    nbcells = dict(sect = [20, 10],
                   full = [15, 7])  # [vm, spk]
    files = dict(sect = 'union_idx_fill_sig_sector.xlsx',
                 full = 'union_idx_fill_sig_full.xlsx')
    filename = os.path.join(paths['owncFig'], 'data', 'averageTraces',
                        'controlsFig', files[spread])
    datadf = pd.read_excel(filename, engine='openpyxl')
    cols = gfunc.new_columns_names(datadf.columns)
    cols = [item.replace('sig_', '') for item in cols]
    cols = [item.replace('_stc', '') for item in cols]
    cols = [st.replace('_iso', '') for st in cols]
    cols = [st.replace('__', '_') for st in cols]
    cols = [st.replace('_.1', '') for st in cols]
    datadf.columns = cols

    # adjust time scale
    middle = (datadf.index.max() - datadf.index.min())/2
    datadf.index = (datadf.index - middle)/10

    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(8.5, 8.5),
                             sharex=True, sharey=False)
    axes = axes.flatten()
    fig.suptitle(title, alpha=0.4)

    recs = ['vm'] + ['spk']
    leg_dic = dict(vm = 'Vm', spk='Spikes')
    cols = [st for st in datadf.columns if spread in st]

    # plot
    for i, ax in enumerate(axes):
        # rec = recs[i]
        # ax_title = f"{rec} {spread}"
        # ax.set_title(ax_title)
        rec = recs[i]        
        txt = leg_dic[rec]
        ax.text(0.7, 0.9, txt, ha='left', va='center', 
                transform=ax.transAxes, size='large')
        
        sec = [st for st in cols if rec in st]
        # replace rd sector by full sector
        sec = [st.replace('sect_rd', 'full_rd') for st in sec]

        df = datadf[sec].copy()
        if substract:
            # subtract the centerOnly response (ref = df['CENTER-ONLY'])
            ref = df[df.columns[0]]
            df = df.subtract(ref, axis=0)
        #build labels
        labels = sec[:]
        labels = [item.split('_')[0] + '_' + item.split('_')[-1]
                  for item in labels]
        labels = [item.replace('rd', 'frd') for item in labels]
        # plot
        for i, col in enumerate(sec):
            ax.plot(df[col], color=colors[i], alpha=alphas[i], label=labels[i],
                    linewidth=1.5)
        # bluePoint
        x = 0
        y = df.loc[0][df.columns[0]]
        vspread = .06  # vertical spread for realign location
        # ax.plot(x, y, 'o', color='tab:gray', ms=10, alpha=0.5)
        ax.vlines(x, y + vspread, y - vspread, linewidth=4, color='tab:gray')
        ax.axvline(x, linewidth=2, color='tab:blue', linestyle=':')

    # adjust
    y_labels = ['Normalized Vm',
                'Normalized Firing Rate']
    x_labels = ['', 'Relative Time (ms)']
    for i, ax in enumerate(axes):
        #labels
        ax.set_xlabel(x_labels[i])
        ax.set_ylabel(y_labels[i])
        # ax.annotate('n={}'.format(nbcells[spread][i]), xy=(0.1, 0.8),
                    # xycoords="axes fraction", ha='center')
        txt = 'n={}'.format(nbcells[spread][i])
        ax.text(0.1, 0.8, txt, ha='left', va='center', 
                transform=ax.transAxes, size='large')

        #refs
        ax.axhline(0, alpha=0.2, color='k')
        for loc in ['top', 'right']:
            ax.spines[loc].set_visible(False)

        if substract:
            ax.set_xlim(-45, 120)
            ax.set_ylim(-0.15, 0.4)
            custom_ticks = np.arange(-40, 110, 20)
            ax.set_xticks(custom_ticks)
            # max_x center only
            ax.axvline(21.4, alpha=0.5, color='k')
            # end_x of center only
            #(df['CENTER-ONLY'] - 0.109773).abs().sort_values().head()
            ax.axvline(88, alpha=0.3)
            ax.axvspan(0, 88, facecolor='k', alpha=0.2)

            ax.text(0.45, 0.9, 'center only response \n start | peak | end',
                    transform=ax.transAxes, alpha=0.5)
            ax.set_ylabel('Norm vm - Norm centerOnly')

        else:
            #set limits
            ax.set_xlim(-20, 60)
            ax.set_ylim(-0.1, 1.1)
            custom_ticks = np.arange(0, 1.1, 0.2)
            ax.set_yticks(custom_ticks)
            custom_ticks = np.arange(-10, 60, 10)
            ax.set_xticks(custom_ticks)

        axes[1].set_ylim(-0.1, 0.85)


        if addinsert:
            # insert subplot inside this one (broader x axis)
            axins = ax.inset_axes([.4, .11, .42, .25], facecolor='w', alpha=0.2)
            for i, col in enumerate(cols):
                if i in [0, 4]:
                    axins.plot(df[col], color=colors[i], alpha=alphas[i], label=col,
                               linewidth=2)
                axins.set_xlim(-150, 30)
            for spine in ['left', 'top']:
                axins.spines[spine].set_visible(False)
            axins.yaxis.tick_right()
            axins.patch.set_edgecolor('w')
            axins.patch.set_alpha(0)
            axins.axvline(0, alpha=0.3)
    fig.tight_layout()
    if anot:
        if addleg:
            for ax in axes:
                ax.legend()
        date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        fig.text(0.99, 0.01, 'pop_traces.py:plot_trace_1x2',
                 ha='right', va='bottom', alpha=0.4)
        fig.text(0.01, 0.01, date, ha='left', va='bottom', alpha=0.4)
        txt = 'pop traces           {}'.format(spread)
        fig.text(0.5, 0.01, txt, ha='center', va='bottom', alpha=0.4)
    return fig


plt.close('all')

dico = {'age': 'new',
 'kind': 'sig',
 'spread': 'sect',
 'rec': 'vm',
 'anot': True,
 'addleg': False,
 'addinsert': False,
 'substract': False,
 'controls': True}

dico['spread'] = 'sect'

fig = plot_trace_1x2(std_colors, **dico)
save = False
if save:
    # file = 'vmSpkUFill_' + dico['spread'] + '.pdf'
    # fig.savefig(os.path.join(paths['save'], 'popfill', file))
    #update current
    if dico['spread'] == 'sect':
        file = 'f8_vmSpkUFill_' + dico['spread']
        folder = os.path.join(paths['owncFig'],
                              'pythonPreview', 'current', 'fig')
        for ext in ['.png', '.pdf']:
            filename = os.path.join(folder, (file + ext))
            fig.savefig(filename)

