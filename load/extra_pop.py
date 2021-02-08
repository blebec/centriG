#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 21 12:42:53 2021

@author: cdesbois
"""

import os
from datetime import datetime
# from importlib import reload

# import matplotlib.gridspec as gridspec
# import matplotlib.patches as patches
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
# #from mpl_toolkits.axes_grid1.inset_locator import inset_axes
# from matplotlib import markers
# from matplotlib.patches import Rectangle
# from pandas.plotting import table

import config
# import fig_proposal as figp
# import general_functions as gfunc
# import load.load_data as ldat
# import load.load_traces as ltra
# import old.old_figs as ofig

# import itertools


# nb description with pandas:
pd.options.display.max_columns = 30

#===========================
# global setup
# NB fig size : 8.5, 11.6 or 17.6 cm

anot = True           # to draw the date and name on the bottom of the plot
std_colors = config.std_colors()
speed_colors = config.speed_colors()
plt.rcParams.update(config.rc_params())
paths = config.build_paths()
os.chdir(paths['pg'])

plt.rcParams.update({
    'font.sans-serif': ['Arial'],
     'font.size': 14,
     'legend.fontsize': 'small',
     'figure.figsize': (11.6, 5),
     'figure.dpi': 100,
     'axes.labelsize': 'medium',
     'axes.titlesize': 'medium',
     'xtick.labelsize': 'medium',
     'ytick.labelsize': 'medium',
     'axes.xmargin': 0})


#%%
def load_latences(sheet=0):
    """
    load the xcel file
    Parameters
    ----------
    sheet : the sheet number in the scel file, int, optional
        DESCRIPTION. The default is 0.

    Returns
    -------
    pandas dataframe
    """
    sheet = ['EXP 1', 'EXP 2'][sheet]
    paths['data'] = os.path.join(paths['owncFig'], 'infos_extra')
    file = 'Tableau_info_integrales_latences.xlsx'
    filename = os.path.join(paths['data'], file)
    df = pd.read_excel(filename, sheet_name=sheet, header=1)
    # adapt columns
    cols = [_.lower().strip() for _ in df.columns]
    cols = [_.replace(' ', '_') for _ in cols]
    cols = [_.replace('__', '_') for _ in cols]
    cols = [_.replace('topsynchro', 'top') for _ in cols]
    cols = [_.replace('latency', 'lat') for _ in cols]
    cols = [_.replace('half_height', 'hh') for _ in cols]
    cols = [_.replace('latence', 'lat') for _ in cols]
    cols = [_.replace('lat_onset', 'latOn') for _ in cols]
    cols = [_.replace('-_top', '-top') for _ in cols]
    cols = [_.replace('hh_lat', 'hhlat') for _ in cols]
    cols = [_.replace('lat_hh', 'hhlat') for _ in cols]
    cols = [_.replace('_-top', '') for _ in cols]
    cols = [_.replace('surround', 's') for _ in cols]
    cols = [_.replace('integral', 'int') for _ in cols]
    cols = [_.replace('toptime', 'top') for _ in cols]


    cols[0] = 'channel'
    # clean row1 replace exp and pre
    # print message
    print('='*10)
    print( 'NB messages removed : {}'.format(df.loc[0].dropna()))
    print('='*10)
    df.drop(df.index[0], inplace=True)
    # rename columns
    df.columns = cols
    # remove empty columns
    df = df.dropna(how='all', axis=1)
    # clean columns
    df[df.columns[0]] = df[df.columns[0]].apply(lambda x: x.split(' ')[1])
    df.layers = df.layers.apply(lambda x: x.split(' ')[1])
    return df

sheet = 0
data_df = load_latences(sheet)

#%%
datadf = data_df.copy()
print(datadf)

print(datadf.describe())

#%%
plt.close('all')

fig, axes = plt.subplots(nrows=2, ncols=3)
axes = axes.flatten()
# fig = plt.figure()
# ax = fig.add_subplot(111)
datadf[datadf.columns[:9]].boxplot(ax=axes[0])

datadf[datadf.columns[[9,10,15,19,12]]].boxplot(ax=axes[1])

datadf[datadf.columns[12:15]].boxplot(ax=axes[2])


datadf[datadf.columns[16:-7]].boxplot(ax=axes[3])

datadf[datadf.columns[-7:-4]].boxplot(ax=axes[4])

datadf[datadf.columns[-4:]].boxplot(ax=axes[-1])

for label in axes[3].get_xticklabels():
    label.set_ha('right')
    label.set_rotation(45)
fig.tight_layout()

#%% replace ± 3mad by nan
def clean_df(df, mult=3):
    """
    replace by nan values outside med ± mult*mad

    """
    total = 0
    count = 1
    while count > 0:
        count = 0
        for col in df.columns:
            if df[col].dtype != float:
                pass
            else:
                ser = df[col]
                med = ser.median()
                mad = ser.mad()
                if (len(ser.loc[ser > (med + mult * mad)]) > 0) or \
                    len(ser.loc[ser < (med - mult * mad)]) > 0:
                        # print('_' * 10)
                        # print(ser.name)
                    num = len(ser.loc[ser > (med + 3 * mad)])
                    num += len(ser.loc[ser < (med - 3 * mad)])
                    print ('{:.0f} values to remove for {:s}'.format(num, ser.name))
                    total += num
                    # print('-' * 10)
                    df[col] = df[col].apply(lambda x: x if x > (med - 3 * mad) else np.nan)
                    df[col] = df[col].apply(lambda x: x if x < (med + 3 * mad) else np.nan)
                    count += 1
    print('='*10)
    print ('removed {} values'.format(total))
    return df

data_df = clean_df(data_df, mult=4)

#%%
# pltconfig = config.rc_params()
# pltconfig['axes.titlesize'] = 'small'
plt.rcParams.update({'axes.titlesize': 'small'})

plt.close('all')

def plot_all_histo(df):

    fig, axes = plt.subplots(nrows=4, ncols=6, figsize=(21, 16))
    cols = df.mean().index.tolist()
    cols = cols[1:]
    axes = axes.flatten()
    for i, col in enumerate(cols):
        ax = axes[i]
        df[col].hist(bins=20, ax=ax, density=True)
        ax.set_title(col)
        med = df[col].median()
        mad = df[col].mad()
        ax.axvline(med, color='tab:orange')
        ax.text(0.6, 0.6, f'{med:.1f}±{mad:.1f}', ha='left', va='bottom',
                transform=ax.transAxes, size='small', color='tab:orange',
                backgroundcolor='w')
        for spine in ['left', 'top', 'right']:
            ax.spines[spine].set_visible(False)
            ax.set_yticks([])
            # q0 = df[col].quantile(q=0.02)
            # q1 = df[col].quantile(q=0.98)
            # ax.set_xlim(q0, q1)

    fig.tight_layout()
    return fig

fig = plot_all_histo(data_df)
save = False
if save:
    file = 'allHisto' + str(sheet) + '.pdf'
    dirname = os.path.join(paths['owncFig'], 'pythonPreview', 'extra')
    filename = os.path.join(dirname, file)
    fig.savefig(filename)



#%% test dotplot

#TODO use floats not integers
# indiquer le nombre de réponses/conditions
# indiquer le stroke interval (frame ~ 7 msec)
# cahanger D0 par t0 = D0, t0 = D1

plt.close('all')

def plot_latencies(datadf, lat_mini=10, lat_maxi=80, sheet=sheet):
    """
    plot the latencies
    input :
        df : pandasDataFrame
        lat_mini : start time to use (values before are dropped)
        lat_maxi : end time to use (values after are removed)
    output :
        matplotlib figure
    """
    isi = {0: 27.8, 1: 34.7}
    isi_shit = isi.get(sheet, None)
    #data filtering
    df = datadf[datadf.columns[[1,3,4,5,6,8,7]]].copy()
    cols = df.columns[1:]
    for col in cols:
        df[col] = df[col].apply(lambda x: x if x < lat_maxi else np.nan)
        df[col] = df[col].apply(lambda x: x if x > lat_mini else np.nan)
    # select columns
    d0_cols = df.columns[:4]
    d1_cols = df.columns[[0,4,5,6]]

    # layer depths limits
    d = 0
    depths = []
    for _ in df.layers.value_counts().values[:-1]:
        d += _
        depths.append(d)
    depths.insert(0, 0)
    depths.append(df.index.max())

    # plotting
    fig = plt.figure(figsize=(15, 12))
    gs = GridSpec(3,3)
    # vertical histogram/kde
    v0 = fig.add_subplot(gs[0, :2])
    # scatter plot
    ax0 = fig.add_subplot(gs[1, :2], sharex=v0)
    ax1 = fig.add_subplot(gs[2, :2], sharex=ax0, sharey=ax0)
    # horizontal histogram
    h0 = fig.add_subplot(gs[1, 2])
    h1 = fig.add_subplot(gs[2, 2], sharex=h0, sharey=h0)
    # cells counts (electrodes)
    c0 = fig.add_subplot(gs[0, 2])

    colors = ['tab:blue', 'tab:red', 'tab:orange', 'tab:green']

    for k, dcols in enumerate([d0_cols, d1_cols]):
    # k=0
    # dcols = d0_cols
        ax = [ax0, ax1][k]
        vax = v0 #[v0, v1][k]
        hax = [h0, h1][k]
        for i, col in enumerate(dcols[1:]):     # drop layers column
        # i=0
        # col = cols[0]
            x = df[col].values.tolist()     # latency value in [0, 100] msec
            y = df[col].index.tolist()      # electrodes / depths
            ax.plot(x, y, '.', color=colors[i+k], markersize=10, alpha=0.5,
                    label=col)
            # ax.axvline(cop[col].median(), color=colors[i], alpha=0.5, linewidth=3)
            meds = df.groupby('layers')[col].median()
            mads = df.groupby('layers')[col].mad()
            for j, med in enumerate(meds):
                ax.vlines(med, depths[j], depths[j+1], color=colors[i+k],
                          alpha=0.5, linewidth=3)
            ax.legend(loc='upper right')
            txt = 'med : {:.0f}±{:02.0f} ({:.0f}, {:.0f}, {:.0f})'.format(
                df[col].median(), df[col].mad(),
                meds.values[0], meds.values[1], meds.values[2])
            ax.text(x=1, y=0.43 - i/8, s=txt, color=colors[i+k],
                    va='bottom', ha='right', transform=ax.transAxes)
            ## vertical histogramm
            # v_height, v_width = np.histogram(x, bins=20, range=(0,100),
            #                                  density=True)
            # vax.bar(v_width[:-1], v_height, width=5, color=colors[i+k],
            #         align='edge', alpha=0.4)
            kde = stats.gaussian_kde([_ for _ in x if not np.isnan(_)])
            x_kde = np.linspace(0,100, 20)
            if k == 0:
                vax.plot(x_kde, kde(x_kde), color=colors[i+k], alpha=0.6,
                         linewidth=2)
            else:
                vax.plot(x_kde, kde(x_kde), color=colors[i+k], alpha=0.6,
                         linestyle=':', linewidth=3)
            ## horizontal histogramm
            y = [0.3*(i-1)+_ for _ in range(len(meds))]
            # histo
            # hax.barh(y=y, width=meds.values, xerr=mads.values, height=0.3,
            #          color=colors[i+k], alpha=0.4)
            # box
            hax.barh(y=y,
                     width=mads.values, left=(meds - mads).values,
                     height=0.3, color=colors[i+k], alpha=0.4)
            hax.barh(y=y,
                     width=mads.values, left=meds.values,
                     height=0.3, color=colors[i+k], alpha=0.4)
            hax.barh(y=y,
                     width=1, left=meds.values - .5,
                     height=0.3, color=colors[i+k], alpha=1)
    # shift
    col = d0_cols[3]        # 'latOn_d0_(s+c)_150°/s'
    x = (df[col] + isi_shit).values.tolist()     # latency value in [0, 100] msec
    y = df[col].index.tolist()      # electrodes / depths
    kde = stats.gaussian_kde([_ for _ in x if not np.isnan(_)])
    x_kde = np.linspace(0,100, 20)
    v0.plot(x_kde, kde(x_kde), color='tab:grey', alpha=0.5, linestyle='-.',
                         linewidth=2, label='150°/sec_I.S.I._shifted')
    v0.legend(loc=2)

    meds = df.groupby('layers')[col].median() + isi_shit
    mads = df.groupby('layers')[col].mad()
    y = [0.3*(2-1)+_ for _ in range(len(meds))]
    h0.barh(y=y,
           width=mads.values, left=(meds - mads).values,
           height=0.3, color='tab:grey', alpha=0.4)
    h0.barh(y=y,
             width=mads.values, left=meds.values,
             height=0.3, color='tab:grey', alpha=0.4)
    h0.barh(y=y,
             width=1, left=meds.values - .5,
             height=0.3, color='tab:grey', alpha=1)

    ## plot nb of cells
    cells = df.groupby('layers').count()
    #cells = cells / len(df)     # normalise
    allcolors = colors[:-1] + colors[1:]
    # x = cells.columns
    alphas = [0.4, 0.6, 0.4]
    # ls = ['-']*3 + [':']*3
    x = list(range(len(cells.columns)))
    for i in range(len(cells))[::-1]:
        y = cells.iloc[i].values
        c0.bar(x=x, height=y, bottom=depths[i], alpha=alphas[i],
               color=allcolors, edgecolor='tab:grey')

    # references lines
    v0.axvline(df[d0_cols[1]].median(), color='tab:blue', alpha=0.5)
    h0.axvline(df[d0_cols[1]].median(), color='tab:blue', alpha=0.5)
    h1.axvline(df[d0_cols[1]].median(), color='tab:blue', alpha=0.5)
    v0.set_title('KDE (plain D0, dotted D1)', color='tab:grey')

    # scatters
    ax1.set_xlabel('msec')
    ax1.set_xlim((0, 100))
    ax1.set_ylim(ax.get_ylim()[::-1])    # O=surfave, 65=deep
    for i, ax in enumerate([ax0, ax1]):
        ax.set_ylabel('depth')
        ax.set_xlabel('D{} based time (msec)'.format(i))
        for spine in ['top', 'right']:
            ax.spines[spine].set_visible(False)
        for d in depths:
            ax.axhline(d, color='tab:grey', alpha=0.5)

    # vertical histo/kde
    for ax in [v0]: #, v1]:
        ax.set_yticks([])
        for spine in ['left', 'top', 'right']:
            ax.spines[spine].set_visible(False)
    for i, col in enumerate(df.columns[1:]):
        diff_med = (df[d0_cols[1]] - df[col]).median()
        diff_mad = (df[d0_cols[1]] - df[col]).mad()
        diff_mean = (df[d0_cols[1]] - df[col]).mean()
        diff_std = (df[d0_cols[1]] - df[col]).std()
        txt = 'diff mean : {:.1f}'.format(diff_mean)
        ax.text(x=1, y=0.9 - i/8, s=txt, color=allcolors[i],
                va='bottom', ha='right', transform=ax.transAxes)
    # cells
    for ax in [c0]:
        for spine in ['bottom', 'right']:
            ax.spines[spine].set_visible(False)
        for d in depths[:-1]:
            ax.axhline(d, color='tab:grey', alpha=0.5)

        ax.set_xticks([])
        ax.set_ylim(64, 0)
        # ax.set_title('responses detected')
        ax.set_xlabel('protocols', color='tab:grey')
        ax.set_ylabel('depth')
        ax.set_title('nb of detections', color='tab:grey')

    # horizontal histo
    h0.set_title('med ± mad', color='tab:grey')
    h0.set_ylim(h0.get_ylim()[::-1])
    labels = list(df.layers.unique())
    for i, ax in enumerate([h0, h1]):
        # ax.set_xticks([])
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels)
        ax.axvline(0, color='tab:grey')
        ax.set_xlabel('D{} based time (msec)'.format(i))
        ax.set_ylabel('layers')
        for spine in ['top', 'right']:
            ax.spines[spine].set_visible(False)
            
    for ax in fig.get_axes():
        ax.tick_params(colors='tab:grey')
        ax.spines['bottom'].set_color('tab:grey')
        ax.spines['top'].set_color('gray')
        ax.xaxis.label.set_color('tab:grey')
        ax.yaxis.label.set_color('tab:grey')
        
        
    if anot:
        txt = 'out of [{}, {}] msec range were excluded'.format(lat_mini, lat_maxi)
        fig.text(0.5, 0.01, txt, ha='center', va='bottom', alpha=0.4)
        date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        fig.text(0.99, 0.01, 'load/extra_pop/plot_latencies',
                 ha='right', va='bottom', alpha=0.4)
        fig.text(0.01, 0.01, date, ha='left', va='bottom', alpha=0.4)
    fig.tight_layout()

    v0.text(x=1, y=0.5, s='KDE, plain <-> D0, dotted <-> D1', color='tab:gray',
            va='bottom', ha='right', transform=ax.transAxes)
    return fig

plt.close('all')

fig = plot_latencies(data_df, lat_mini=0, lat_maxi=80)

save = False
if save:
    file = 'latencies' + str(sheet) + '.pdf'
    dirname = os.path.join(paths['owncFig'], 'pythonPreview', 'extra')
    filename = os.path.join(dirname, file)
    fig.savefig(filename)

#%%  to be adpated

# def statsmodel_diff_mean(df, param=params):
#     df = df.dropna()
#     # extract correlation
#     y = df.diffe
#     x = df.moy
#     # build the model & apply the fit
#     x = sm.add_constant(x) # constant intercept term
#     model = sm.OLS(y, x)
#     fitted = model.fit()
#     print(fitted.summary())

#     #make prediction
#     x_pred = np.linspace(x.min()[1], x.max()[1], 50)
#     x_pred2 = sm.add_constant(x_pred) # constant intercept term
#     y_pred = fitted.predict(x_pred2)
#     print(y_pred)

#     fig = plt.figure(figsize=(8, 6))
#     ax = fig.add_subplot(111)
#     # x = 'ip1m'  # PVC
#     # y = 'ip2m'  # jug
#     sm.graphics.mean_diff_plot(m1=df.jug, m2=df.cvp, ax=ax)
#     ax.plot(x_pred, y_pred, color='tab:red', linewidth=2, alpha=0.8)
#     txt = 'difference = {:.2f} + {:.2f} mean'.format(
#         fitted.params['const'], fitted.params['moy'])
#     ax.text(13.5, -1, txt, va='bottom', ha='right', color='tab:red')

#     ax.axvline(df.moy.mean(), color='tab:orange', linewidth=2, alpha=0.6)
#     txt = 'measures = \n {:.2f} ± {:.2f}'.format(
#         df.moy.mean(), df.moy.std())
#     ax.text(8.7, -2.7, txt, color='tab:orange', va='center', ha='right')

#     ax.axhline(df.diffe.mean(), color='tab:orange', linewidth=2, alpha=0.6)
#     txt = 'differences = \n {:.2f} ± {:.2f}'.format(
#         df.diffe.mean(), df.diffe.std())
#     ax.text(13.5, 0.6, txt, color='tab:orange', va='center', ha='right')

#     ax.set_ylabel('jug - cvp    (mmHg)')  # ip2m - ip1m
#     ax.set_xlabel('mean jug|cvp    (mmHg)')
#     ax.axhline(0, color='tab:blue', alpha=0.6)
#     for spine in ['left', 'top', 'right', 'bottom']:
#         ax.spines[spine].set_visible(False)
#     fig.text(0.99, 0.01, 'cDesbois', ha='right', va='bottom', alpha=0.4, size=12)
#     fig.text(0.01, 0.01, param['file'], ha='left', va='bottom', alpha=0.4)
#     return fig

# df = datadf[datadf.columns[3:9]].copy()
