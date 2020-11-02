


import os
from datetime import datetime
from imp import reload

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="ticks")

import config
import load.load_data as ldat
paths=config.build_paths()
std_colors = config.std_colors()
anot = True

plt.rcParams['axes.xmargin'] = 0.05
plt.rcParams['axes.ymargin'] = 0.05
        
#%%
def load_all_indices():
    amp = 'gain'
    kind = 'vm'
    res_df = pd.DataFrame()
    for kind in ['vm', 'spk'][::-1]:
        dfs = []
        for amp in ['gain', 'engy']:
            #load
            df = ldat.load_cell_contributions(rec=kind, amp=amp)
            #remove sig
            cols = df.columns
            cols = [item for item in cols if '_sig' not in item]
            df = df[cols]
            dfs.append(df)
            #merge
        mg_df = pd.merge(dfs[0].reset_index(), dfs[1].reset_index()).set_index('Neuron')
        #conds
        stims = list(set([item.split('_')[0] for item in mg_df.columns]))
        params = list(set([item.split('_')[1] for item in mg_df.columns]))

        if len(res_df.columns) < 1:
            cols = params.copy()
            cols.extend(['kind', 'stim', 'spread'])
            res_df = pd.DataFrame(columns=cols)

        for stim in stims:
            afilter = [('_').join([stim, item]) for item in params]    
            df = mg_df[afilter].reset_index(drop=True)
            df.columns = params
            df['kind'] = kind
            df['stim'] = stim[:-4]
            df['spread'] = stim[-4:]
            res_df = res_df.append(df, ignore_index=True)

    #change column order for plotting convenience
    cols = res_df.columns.tolist()
    if cols[0] != 'time50':
        cols[0], cols[1] = cols[1], cols[0]
        res_df = res_df.reindex(columns=cols)
    return res_df

indices_df = load_all_indices()

#%%

def do_pair_plot(df, kind='vm', spread='sect'):
    """
    pair plot of indices for the different conditions
    """    
    if kind not in ['vm', 'spk']:
        print('kind should be in [vm, spk]')
        return
    if spread not in ['sect', 'full']:
        print('spread should be in [sect, full]')
        return
    colors = [std_colors[item] for item in 
              ['blue', 'yellow', 'green', 'red']] 
    hue_order = ['rdiso', 'cpcrx', 'cfiso', 'cpiso']
    sns.set_palette(sns.color_palette(colors))
    
    nb_stims = len(df.stim.unique())
    
    sub_df = df.loc[df.kind == kind].copy()
    sub_df = sub_df.loc[sub_df.spread == spread]    
    
    g = sns.pairplot(sub_df, kind='reg', diag_kind='hist',
                     hue='stim', hue_order=hue_order,
                     markers=['p']*nb_stims,
                     diag_kws={'alpha': 0.5, 'edgecolor':'k'},
                     corner=True) 
    title = kind + '  ' + spread
    g.fig.suptitle(title)
    g.fig.tight_layout()
    # g.fig.subplots_adjust(top=0.96, bottom=0.088, left=0.088,
    #                       right=0.9, hspace=0.1, wspace=0.1)
    return g.fig


plt.close('all')

save = False
savePath = os.path.join(paths['owncFig'], 
                        'pythonPreview', 'proposal', '5_enerPeakOrGain')

for spread in ['sect', 'full']:
    for kind in ['vm', 'spk']:
        fig = do_pair_plot(indices_df, kind=kind, spread=spread)    
        if save:
            file = 'pairplot' \
                + kind[:1].upper() + kind[1:] \
                + spread[:1].upper() + spread[1:] \
                + '.png'
            filename = os.path.join(savePath, file)
            fig.savefig(filename)
#%% see https://blog.insightdatascience.com/data-visualization-in-python-advanced-functionality-in-seaborn-20d217f1a9a6
plt.close('all')
g = sns.pairplot(indices_df, kind='reg', diag_kind='kde', hue='kind', palette='Set1')
                 #plot_kws={"alpha":.3, 'edgecolor': None},
                 #diag_kws={})

g = g.map_offdiag(plt.scatter, s=35, alpha=0.3)
g = g.map_diag(plt.hist, stacked=True, edgecolor='k', alpha=.6)

g.fig.suptitle('indices')
# g.set(alpha=.5)
g =  g.map_lower(sns.regplot)
# g.map_upper(sns.residplot)

#%%
plt.close('all')

def do_lmplot_plot(df, kind='vm', spread='sect'):
    """
    pair plot of indices for the different conditions
    """    
    if kind not in ['vm', 'spk']:
        print('kind should be in [vm, spk]')
        return
    if spread not in ['sect', 'full']:
        print('spread should be in [sect, full]')
        return
    #colors
    colors = [std_colors[item] for item in ['blue', 'yellow', 'green', 'red']] 
    hue_order = ['rdiso', 'cpcrx', 'cfiso', 'cpiso']
    sns.set_palette(sns.color_palette(colors))
    
    nb_stims = len(df.stim.unique())
    
    sub_df = df.loc[df.kind == kind].copy()
    sub_df = sub_df.loc[sub_df.spread == spread]    
        
    g1 = sns.lmplot(x='time50', y ='gain50', hue='stim', hue_order=hue_order, 
                   data= sub_df)
    g2 = sns.lmplot(x='time50', y ='engy', hue='stim', hue_order=hue_order, 
                   data= sub_df)
    title = kind + '  ' + spread
    for g in [g1, g2]:
        g.fig.suptitle(title)
        g.fig.tight_layout()
        g.fig.subplots_adjust(right=0.902)
    return g1.fig, g2.fig

save = False
savePath = os.path.join(paths['owncFig'], 
                        'pythonPreview', 'proposal', 'enerPeakOrGain')
mes= ['Gain', 'Engy']
for spread in ['sect', 'full']:
    for kind in ['vm', 'spk']:
        fig1, fig2 = do_lmplot_plot(indices_df, kind=kind, spread=spread)    
        if save:
            for i, fig in enumerate([fig1, fig2]):
                file = 'lmplot' + '_' \
                    + kind[:1].upper() + kind[1:] \
                    + spread[:1].upper() + spread[1:] \
                    + '_' + mes[i] + '.png'
                filename = os.path.join(savePath, file)
                fig.savefig(filename)

#%%
plt.close('all')
from itertools import combinations
kinds = ['vm', 'spk']
spreads = ['sect', 'full']
mes = ['time50', 'gain50', 'engy']
    
for kind in kinds:
    for spread in spreads:
        for combo in combinations(mes, 2):
            x = combo[0]
            y = combo[1]
            df = indices_df.loc[indices_df.kind == kind]
            g = sns.jointplot(x=x, y=y, data= df, kind='reg')
            g.fig.suptitle(kind + '  ' + spread)
    
