from datetime import datetime
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

import config
import load_data as ldat
std_colors = config.std_colors()
anot = True

#%%
for amp in ['gain', 'engy']:
    for kind in ['vm', 'spk']:
        df = ldat.load_cell_contributions(kind=kind, amp=amp)

res_df = pd.DataFrame()
        
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
            res_df = pd.DataFrame(columns=cols)

        for stim in stims:
            afilter = [('_').join([stim, item]) for item in params]    
            df = mg_df[afilter].reset_index(drop=True)
            df.columns = params
            df['kind'] = kind
            df['stim'] = stim
            res_df = res_df.append(df, ignore_index=True)

    #changre column order for plotting convenience
    cols = res_df.columns.tolist()
    cols[0], cols[1] = cols[1], cols[0]
    res_df = res_df.reindex(columns=cols)
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
import seaborn as sns
sns.set(style="ticks")

#df = sns.load_dataset("iris")
g = sns.pairplot(indices_df.loc[indices_df.kind == 'vm'], kind='reg', diag_kind='kde',
                 markers=['p'])
g.fig.suptitle('Vm')
 
g = sns.pairplot(indices_df.loc[indices_df.kind == 'spk'], kind='reg', diag_kind='kde',
                 markers=['2'])
g.fig.suptitle('spk')

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
