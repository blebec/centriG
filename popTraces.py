#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from datetime import datetime
from importlib import reload

import numpy as np
import matplotlib.pyplot as plt

from centriG import config
import centriG.load.load_traces as ltra
import centriG.old.old_figs as ofig

paths = config.build_paths()
std_colors = config.std_colors()
anot = True
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
    colors = ['k', stdcolors['red'], stdcolors['green'],
              stdcolors['yellow'], stdcolors['blue'], stdcolors['blue']]
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
    for i in range(3):
        for item in labels:
            if (len(item.split('_')) < 6):
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
    ax.plot(0, df.loc[0][df.columns[0]], 'o', color=colors[0],
            ms=10, alpha=0.5)
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
#    insert = False
    if addinsert:
        """
        insert subplot inside this one (broader x axis)
        """
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
        fig.text(0.99, 0.01, 'popTraces.py:plot_figure3(' + kind + ')',
                 ha='right', va='bottom', alpha=0.4)
        fig.text(0.01, 0.01, date, ha='left', va='bottom', alpha=0.4)
    return fig


#params (load)
select = dict(age='new', rec='vm', kind='sig')

data_df, file = ltra.load_intra_mean_traces(paths, **select)
print('loaded {}'.format(_))
# TODO recéuprer le nome du fichier pout le nombre de cellules
#params (plot)
select['file'] = file
select['substract'] = True
select['addleg'] = True
select['addinsert'] = False
fig = plot_figure3(data_df, std_colors, **select)

#%% nb use select to change the parameters
plt.close('all')

figs = []
for kind in ['pop', 'sig']:
    select['age'] = 'new'
    select['kind'] = kind
    select['leg'] = False
    data_df, _ = ltra.load_intra_mean_traces(paths, **select)
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
                df, f = ltra.load_intra_mean_traces(paths, kind=kind, age=age, rec=rec, spread=spread)
                plot_figure3(df, std_colors, kind=kind, age=age, rec=rec, spread=spread)

#%%
plt.close('all')
#%%
dico = dict(
    age=['old', 'new'][1],
    kind=['pop', 'sig', 'nsig'][2],
    spread=['sect', 'full'][0],
    rec=['vm', 'spk'][1]
    )

dico['kind'] = 'pop'
df1, f1 = ltra.load_intra_mean_traces(paths, **dico)
fig1 = plot_figure3(df1, std_colors, **dico)

dico['kind'] = 'sig'
df2, f2 = ltra.load_intra_mean_traces(paths, **dico)
fig2 = plot_figure3(df2, std_colors, **dico)

dico['kind'] = 'nsig'
df3, f3 = ltra.load_intra_mean_traces(paths, **dico)
fig3 = plot_figure3(df3, std_colors, **dico)

