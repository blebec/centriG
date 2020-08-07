from datetime import datetime
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="ticks")
import config
import load_data as ldat
std_colors = config.std_colors()
anot = True
        
#%%
def load_all_indices():
    amp = 'gain'
    kind = 'vm'
    res_df = pd.DataFrame()
    for kind in ['vm', 'spk'][::-1]:
        dfs = []
        for amp in ['gain', 'engy']:
            #load
            df = ldat.load_cell_contributions(kind=kind, amp=amp)
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
            cols.append('kind')
            cols.append('stim')
            cols.append('spread')
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
    print('pre ', cols)
    if cols[0] != 'time50':
        cols[0], cols[1] = cols[1], cols[0]
        res_df = res_df.reindex(columns=cols)
    print('post ', cols)
    return res_df

indices_df = load_all_indices()
#%%
plt.close('all')

colors = [{'spk':'r', 'vm':'b'}[item] for item in indices_df.kind.values]
fig = plt.figure(figsize=(10, 10))
#axes= axes.flatten()
ax = fig.add_subplot(111)
pd.plotting.scatter_matrix(indices_df, ax=ax, alpha=0.3, 
                           color=colors, 
                           hist_kwds={'bins': 20}, s=100)
fig.suptitle('Vm')
fig.legend()
#%%
plt.close('all')


#df = sns.load_dataset("iris")
g = sns.pairplot(indices_df.loc[indices_df.kind == 'vm'], kind='reg', diag_kind='hist',
                  hue='stim', markers=['p']) 
g.fig.suptitle('Vm')
 
g = sns.pairplot(indices_df.loc[indices_df.kind == 'spk'], kind='reg', diag_kind='kde',
                  hue='stim', markers=['2']*8)
g.fig.suptitle('spk')

#%%
plt.close('all')


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
    colors = [std_colors[item] for item in ['blue', 'yellow', 'green', 'red']] 
    hue_order = ['rdiso', 'cpcrx', 'cfiso', 'cpiso']
    sns.set_palette(sns.color_palette(colors))
    
    nb_stims = len(df.stim.unique())
    
    sub_df = df.loc[df.kind == kind].copy()
    sub_df = sub_df.loc[sub_df.spread == spread]    
    
    g = sns.pairplot(sub_df, kind='reg', diag_kind='kde',
                     hue='stim', hue_order=hue_order,
                     markers=['p']*nb_stims) 
    title = kind + '  ' + spread
    g.fig.suptitle(title)
    g.fig.tight_layout()
    g.fig.subplots_adjust(right=0.902)
    return g.fig
    
for spread in ['sect', 'full']:
    for kind in ['vm', 'spk']:
        fig = do_pair_plot(indices_df, kind=kind, spread=spread)    
    
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
from scipy.stats import pearsonr

def corrfunc(x, y, **kws):
    (r, p) = pearsonr(x, y)
    ax = plt.gca()
    ax.annotate("r = {:.2f} ".format(r),
                xy=(.1, .9), xycoords=ax.transAxes)
    ax.annotate("p = {:.3f}".format(p),
                xy=(.4, .9), xycoords=ax.transAxes)

df = indices_df.loc[indices_df.kind == 'vm']
g = sns.jointplot(x='time50', y = 'gain50', data= df, kind='reg')
g = sns.jointplot(x='time50', y = 'engy', data= df, kind='reg')
g = sns.jointplot(x='gain50', y = 'engy', data= df, kind='reg')


#%%
sns.set(style="ticks", color_codes=True)

g = sns.JointGrid(x="time50", y="gain50", data=df)

g = g.plot(sns.regplot, sns.distplot)


#%%
g = sns.JointGrid(x="time50", y="gain50", data=df)

g = g.plot_joint(sns.scatterplot, color=".5")
g= g.plot_marginals(sns.distplot, kde=True, color='.5')
