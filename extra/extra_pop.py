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
     'axes.xmargin': 0.05})

paths['data'] = os.path.join(paths['owncFig'], 'data', 'data_extra')

#% to load the csv
def load_csv_latencies(filename):
    # blocs de 3 3 stim = une vitesse
    # avant dernière = blanc
    # dernière = D1 sc 150 °/C
    # avant dernière = blank
    # t0 = D1 pour s0 et dernière SC150
    # t0 = D0 pour le reste (including blank)

    # on
    on_speeds = dict(enumerate([25, 50, 100, 150, None], start=0))
    on_stim = dict(enumerate(['d0_0c', 'd1_s0', 'd0_sc'], start=0))
    last = {'d1_sc', 150}
    blast = {'d0_blank'}
    # hh
    hh_stim = dict(enumerate(['0c', 's0', 'sc', 'spc'], start=1))
    hh_speeds = dict(enumerate([25, 50, 100, 150, None], start=1))
    # integral


    onDict = {
        '1' : '0c_25',
        '2' : 's0_25',
        '3' : 'sc_25',
        '10' : '0c_150',
        '11' : 's0_150',
        '12' : 'sc_150'
        }

    stimDict = {
        '1' : '0c',
        '2' : 's0',
        '3' : 'sc',
        '4' : 'spc'
            }

    speedDict = {
        '1' : '25',
        '4' : '150'
        }

    spkDict = {
        '1' : 'mua',
        '2' : 'lfp'
        }

    df = pd.read_csv(filename, sep=';', header=None)
    df = df.set_index(df.columns[0]).T
    cols = df.columns
    cols = [_.replace('_latency_PG0.VEC_PST_HALF_LATENCY_TOP_AE', '') for _ in cols]
    cols = [_.replace('set_latency_PG0.FIRST_CROSS_LIST', '') for _ in cols]
    cols = [_.replace('egral_PG0.INTEGRAL_LATENCY_LIST', '') for _ in cols]
    cols = [_.replace('nificativity_PG0.SIG_LIST_LAT', '') for _ in cols]

    hhs = [_ for _ in cols if _.startswith('hh')]
    ons = [_ for _ in cols if _.startswith('on')]
    ints = [_ for _ in cols if _.startswith('int')]
    sigs = [_ for _ in cols if _.startswith('sig')]
    #on names
    new = []
    maxi = len(ons) -1
    for i, item in enumerate(ons):
        if i == maxi - 1:
            txt = 'd0_blk'
        elif i == maxi:
            txt = '{}_{}'.format(on_stim[2], on_speeds[3])
            txt = txt.replace('d0', 'd1')
        else:
            a, b = divmod(i, 3)
            txt = '{}_{}'.format(on_stim[b], on_speeds[a])
        # print('{}   {}'.format(txt, item))
        new.append('on_' + txt)
    ons = new

    newcols = []
    for item in [hhs, ons, ints, sigs]:
        newcols.extend(item)
    # for item in cols:
    #     mes, cond = item.split('[')
    #     if item.startswith('hh'):
    #         a, b, c = cond.strip('[').strip(']').split(',')
    #         txt = '{}_{}_{}'.format( stimDict.get(a, a),
    #                                 speedDict.get(b,b),
    #                                 spkDict.get(c, c))
    #         newcols.append('{}_{}'.format(mes, txt))
    #     elif item.startswith(('on', 'int')):
    #         a = onDict.get(cond.strip(']'), cond)
    #         txt = '{}_{}'.format(mes, a)
    #         newcols.append(txt)
    #     else:
    #         newcols.append(item)

    df.columns = newcols
    # remove the last line
    df = df.dropna()
    # normalise sig (ie 1 or 0 instead of 10 or 0)
    if [_ for _ in df.columns if 'sig' in _ ]:
        df[[_ for _ in df.columns if 'sig' in _ ]] /= 10

    # layers
    layers = pd.read_csv(os.path.join(os.path.dirname(filename),
                                      'layers.csv'), sep=';')
    layers = layers.set_index(layers.columns[0]).T

    cols = [_ for _ in layers.columns if _ in os.path.basename(filename)]
    if len(cols) == 1:
        col = cols[0]
    else:
        raise Exception ('no layers definition')
    ser = layers[col]
    ser.name= 'layers'
    ser = ser.apply(lambda x: x.split('_')[-1])
    ser = ser.astype('str')
    df['layers'] = ser.values
    layersDict = {
        '2' : '2/3',
        '3' : '2/3',
        '5' : '5/6',
        '6' : '5/6'
        }
    df.layers = df.layers.apply(lambda x: layersDict.get(x, x))
    return df

csvLoad = False
if csvLoad:
    files = os.listdir(paths['data'])
    files = [_ for _ in files if _[:4].isdigit()]

    # file = files[3]
    file = '1319_CXLEFT_TUN25_s30_csv_test.csv'
    file = '2019_CXRIGHT_TUN21_s30_csv_test.csv'
    sheet=file
    file_name = os.path.join(paths['data'], file)

    data_df = load_csv_latencies(file_name)

#%
def load_latencies(sheet=0):
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
    cols = [_.replace('lat_onset', 'on') for _ in cols]
    cols = [_.replace('-_top', '-top') for _ in cols]
    cols = [_.replace('hh_lat', 'hhlat') for _ in cols]
    cols = [_.replace('lat_hh', 'hhlat') for _ in cols]
    cols = [_.replace('_-top', '') for _ in cols]
    cols = [_.replace('surround', 's') for _ in cols]
    cols = [_.replace('integral', 'int') for _ in cols]
    cols = [_.replace('toptime', 'top') for _ in cols]
    cols = [_.replace('_(10_=_yes;_0_=_no)', '') for _ in cols]

    cols = [_.replace('(s+c)', 'sc') for _ in cols]
    cols = [_.replace('s+c', 'sc') for _ in cols]
    cols = [_.replace('_s_', '_s0') for _ in cols]
    cols = [_.replace('(c)', '0c') for _ in cols]

    cols = [_.replace('°/s', '') for _ in cols]
    cols = [_.replace('(', '_') for _ in cols]
    cols = [_.replace(')', '_') for _ in cols]
    cols = [_.replace('__', '_') for _ in cols]
    cols = [_.strip('_') for _ in cols]

    cols = [_.replace('δ', 'D_') for _ in cols]
    cols = [_.replace('center', '0c') for _ in cols]
    cols = [_.replace('s_', 's0_') for _ in cols]

    cols = [_.replace('d0_0c', 'd0_0c_25') for _ in cols]
    # cols = [_.replace('(150°/s)', '150') for _ in cols]
    # cols = [_.replace('150°/s', '150') for _ in cols]
    # cols = [_.replace('(25)', '_25') for _ in cols]


#        selection = ['on_d0_0c_25', 'on_d0_sc_25', 'on_d0_sc_150',
#                 'on_d1_s0_25', 'on_d1_sc_150', 'on_d1_s0_150']

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
    df.channel = df.channel.apply(lambda x: int(x.split()[-1]))
    df.set_index('channel', inplace=True)
    df.layers = df.layers.apply(lambda x: x.split(' ')[1])
    df.int_d0 = df.int_d0.astype(float)
    df.significancy = (df.significancy/10).astype(bool)
    return df

#% replace ± 3mad by nan
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


def print_global_stats(statsdf, statssigdf):
    """ ^print description """
    for col in stats_df.columns:
        print('mean {:.2f} ±({:.2f})\t sig {:.2f} ±({:.2f}) \t {}'.format(
            statsdf.loc['mean',[col]][0], statsdf.loc['std',[col]][0],
            statssigdf.loc['mean', [col]][0], statssigdf.loc['std', [col]][0],
            col))

params = {0 : '1319_CXLEFT',
          1 : '2019_CXRIGHT'}
sheet = 0
data_df = load_latencies(sheet)
data_df = clean_df(data_df, mult=4)
stats_df = data_df.describe()
stats_df_sig = data_df[data_df.significancy].describe()

#%% desribe basics

def plot_boxplots(datadf, removemax=True):
    ons = [_ for _ in datadf. columns if _.startswith('on')]
    hhtimes = [_ for _ in datadf.columns if 'hhtime' in _]
    ints = [_ for _ in datadf.columns if 'int' in _]

    # remove values of 100 <-> no detection
    if removemax:
        for col in datadf.columns:
            if data_df[col].dtypes == float:
                datadf[col] = datadf[col].apply(
                    lambda x : x if x < 100 else np.nan)

    fig, axes = plt.subplots(nrows=1, ncols=3)
    axes = axes.flatten()
    for i, dats in enumerate([ons, hhtimes, ints]):
        ax = axes[i]
        ax.boxplot(datadf[dats].dropna())
        txt = dats[0].split('_')[0]
        if dats == ons:
            txt = 'ons_latency'
            labels = ['_'.join(_.split('_')[1:]) for _ in dats]
        elif dats == hhtimes:
            txt = 'hh_times'
            labels = ['_'.join(_.split('_hhtime_lat_')[:]) for _ in dats]
        elif dats == ints:
            txt = 'integrals'
            labels = ['_'.join(_.split('_')[1:]) for _ in dats]
        med = datadf[dats[0]].median()
        mad = datadf[dats[0]].mad()
        ax.axhline(med, color='tab:blue', linewidth=3, alpha=0.7)
        ax.axhline(med + 2*mad, color='tab:blue',
                   linewidth=2, linestyle=':', alpha=0.7)
        ax.axhline(med - 2*mad, color='tab:blue',
                   linewidth=2, linestyle=':', alpha=0.7)
        ax.set_title(txt)
        ax.set_xticklabels(labels, rotation=45, ha='right')
        txt = 'med ± 2*mad'
        ax.text(x=ax.get_xlim()[0], y=med  + 2* mad + 5, s=txt,
                color='tab:blue',  va='bottom', ha='left')
    for ax in axes:
        for spine in ['top', 'right']:
            ax.spines[spine].set_visible(False)

    if anot:
        txt = 'file= {} ({})'.format(params.get(sheet, sheet), sheet)
        fig.text(0.5, 0.01, txt,
                 ha='center', va='bottom', alpha=0.4)
        date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        fig.text(0.99, 0.01, 'extra/extra_pop/plot_boxplots',
                 ha='right', va='bottom', alpha=0.4)
        txt = '{} {}'.format(date, '_'.join(str(sheet).split('_')[:3]))
        fig.text(0.01, 0.01, txt, ha='left', va='bottom', alpha=0.4)
    fig.tight_layout()
    return fig


plt.close('all')
fig = plot_boxplots(data_df)

save=False
if save:
    file = 'boxPlot_' + str(sheet) + '.pdf'
    dirname = os.path.join(paths['owncFig'], 'pythonPreview', 'extra')
    filename = os.path.join(dirname, file)
    fig.savefig(filename)

#%%
# pltconfig = config.rc_params()
# pltconfig['axes.titlesize'] = 'small'
plt.rcParams.update({'axes.titlesize': 'small'})


def plot_all_histo(df):

    fig, axes = plt.subplots(nrows=4, ncols=7, figsize=(21, 16))
    axes = axes.flatten()
    cols = []
    for col in df.columns:
        if df[col].dtype == 'float64':
            cols.append(col)
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
    for ax in fig.get_axes():
        for spine in ['left', 'top', 'right']:
            ax.spines[spine].set_visible(False)
            ax.set_yticks([])
            # q0 = df[col].quantile(q=0.02)
            # q1 = df[col].quantile(q=0.98)
            # ax.set_xlim(q0, q1)
    if anot:
        txt = 'file= {} ({})'.format(params.get(sheet, sheet), sheet)
        fig.text(0.5, 0.01, txt,
                 ha='center', va='bottom', alpha=0.4)
        date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        fig.text(0.99, 0.01, 'extra/extra_pop/plot_all_histo',
                 ha='right', va='bottom', alpha=0.4)
        txt = '{} {}'.format(date, '_'.join(str(sheet).split('_')[:3]))
        fig.text(0.01, 0.01, txt, ha='left', va='bottom', alpha=0.4)
    fig.tight_layout()

    return fig

plt.close('all')
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


def plot_latencies(datadf, lat_mini=10, lat_maxi=80, sheet=sheet, xcel=False):
    """
    plot the latencies
    input :
        df : pandasDataFrame
        lat_mini : start time to use (values before are dropped)
        lat_maxi : end time to use (values after are removed)
    output :
        matplotlib figure
    """
    isi = {0: 27.8, 1: 34.7,
           '1319_CXLEFT_TUN25_s30_csv_test.csv' : 27.8}
    isi_shift = isi.get(sheet, 0)
    #data filtering
    # xcel = False
    if xcel:
        df = datadf[datadf.columns[[1,3,4,5,6,8,7]]].copy()
        selection = ['on_d0_0c', 'on_d0_sc_25', 'on_d0_sc_150',
                     'on_d1_s0_25', 'on_d1_s0_150', 'on_d1_sc_150']
    else:
        selection = ['on_d0_0c_25', 'on_d0_sc_25', 'on_d0_sc_150',
                 'on_d1_s0_25', 'on_d1_sc_150', 'on_d1_s0_150']
    selection.insert(0, 'layers')
    df = datadf[selection].copy()
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
    x = (df[col] + isi_shift).values.tolist()     # latency value in [0, 100] msec
    y = df[col].index.tolist()      # electrodes / depths
    kde = stats.gaussian_kde([_ for _ in x if not np.isnan(_)])
    x_kde = np.linspace(0,100, 20)
    v0.plot(x_kde, kde(x_kde), color='tab:grey', alpha=0.5, linestyle='-.',
                         linewidth=2, label='150°/sec_I.S.I._shifted')
    v0.legend(loc=2)

    meds = df.groupby('layers')[col].median() + isi_shift
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
        fig.text(0.99, 0.01, 'extra/extra_pop/plot_latencies',
                 ha='right', va='bottom', alpha=0.4)
        txt = '{} , data <-> sheet {}'.format(date, '_'.join(str(sheet).split('_')[:3]))
        fig.text(0.01, 0.01, txt, ha='left', va='bottom', alpha=0.4)
    fig.tight_layout()

    v0.text(x=1, y=0.5, s='KDE, plain <-> D0, dotted <-> D1', color='tab:gray',
            va='bottom', ha='right', transform=ax.transAxes)
    return fig


new = False
if new :
    sheet = 0
    data_df = load_latencies(sheet)
    data_df = clean_df(data_df, mult=4)

#sheet = file

plt.close('all')
fig = plot_latencies(data_df, lat_mini=0, lat_maxi=80, sheet=sheet, xcel=True)

save = False
if save:
    sheet = str(sheet)
    file = 'latencies_' + str('_'.join(sheet.split('_')[:3])) + '.pdf'
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


#%% significancy
data_df.loc[data_df.significancy > 0, ['layers', 'significancy']]

sig = data_df.loc[data_df.significancy > 0, ['layers', 'significancy']].groupby('layers').count()
allpop = data_df.groupby('layers')['significancy'].count()


sigDf = pd.DataFrame(pd.Series(allpop))


#%%

def plot_d1_d0_low(datadf, sheet):
    """
    plot the d1 d0 relation
    input:
        datadf : pandas dataframe
        sheet : the sheet number from the related xcel file
    """
    # layer depths limits
    d = 0
    depths = []
    for _ in datadf.layers.value_counts().values[:-1]:
        d += _
        depths.append(d)
    depths.insert(0, 0)
    depths.append(datadf.index.max())

    # latencies
    select = ['layers', 'on_d0_0c_25', 'on_d1_s0_25']
    # remove outliers
    df = datadf[select].copy()
    for col in df.columns[1:]:
        df[col] = df[col].apply(lambda x: x if x < 100 else np.nan)
        df[col] = df[col].apply(lambda x: x if x > 1 else np.nan)

    fig = plt.figure(figsize=(10, 12))
    fig.suptitle('manip {}   (med ± mad values)'.format(sheet))

    # d1 vs do
    ax = fig.add_subplot(221)
    subdf = df[[select[1], select[2]]].dropna()
    cols = subdf.columns
    ax.scatter(subdf[cols[0]], subdf[cols[1]],
               marker='o', s=65, alpha=0.8, color='tab:blue')
    med = subdf[cols[0]].median()
    mad = subdf[cols[0]].mad()
    ax.axvspan(med-mad, med+mad, color='tab:blue', alpha=0.3)
    txt = '{:.0f}±{:.0f}'.format(med, mad)
    ax.text(med, subdf.max().max(), txt, 
            va='top', ha='center', color='tab:blue')
    med = subdf[cols[1]].median()
    mad = subdf[cols[1]].mad()
    ax.axhspan(med-mad, med+mad, color='tab:blue', alpha=0.3)
    txt = '{:.0f}±{:.0f}'.format(med, mad)
    ax.text(subdf.max().max(), med , txt, 
            va='center', ha='right', color='tab:blue')
    # toto  use floor and ceil
    lims = (subdf.min().min() - 5, subdf.max().max() + 5)
    ax.set_ylim(lims)
    ax.set_xlim(lims)
    ax.plot(lims, lims)
    # regress
    slope, intercept = np.polyfit(subdf[cols[0]], subdf[cols[1]], 1)
    xs = (subdf[cols[0]].min(),
          subdf[cols[0]].max())
    fxs = (intercept + slope * subdf[cols[0]].min(),
           intercept + slope * subdf[cols[0]].max())
    ax.plot(xs, fxs, color='tab:red', linewidth=2, alpha=0.8)

    ax.set_xlabel('_'.join(cols[0].split('_')[1:]))
    ax.set_ylabel('_'.join(cols[1].split('_')[1:]))

    # diff vs depth
    ax = fig.add_subplot(222)
    subdf = (df[select[2]] - df[select[1]]).dropna()
    ax.plot(subdf, 'o', alpha=0.8, ms=10, color='tab:blue')
    med = subdf.median()
    mad = subdf.mad()
    ax.axhline(med, color='tab:blue', linewidth=3, alpha=0.7)
    ax.axhline(med + 2*mad, color='tab:blue',
               linewidth=2, linestyle=':', alpha=0.7)
    ax.axhline(med - 2*mad, color='tab:blue',
               linewidth=2, linestyle=':', alpha=0.7)
    txt = '{:.0f}±{:.0f}'.format(med, mad)
    ax.text(ax.get_xlim()[1], med, txt, 
            va='bottom', ha='right', color='tab:blue')
    # layers
    ax.axvspan(depths[1], depths[2], color='tab:grey', alpha=0.3)
    txt = 'layer IV'
    ax.text(x=(depths[1] + depths[2])/2, y = ax.get_ylim()[1], s=txt,
           va='top', ha='center', color='tab:grey')
    ax.text(ax.get_xlim()[1], med + 2*mad, 'med+2*mad',
            va='bottom', ha='right', color='tab:blue')

    #ax.set_ylim((ax.get_ylim)()[::-1])
    ax.axhline(0, color='tab:gray')
    ax.set_xlabel('depth (electrode nb)')
    # regress
    slope, intercept = np.polyfit(subdf.index, subdf.values, 1)
    xs = (subdf.index.min(),
          subdf.index.max())
    fxs = (intercept + slope * xs[0],
           intercept + slope * xs[1])
    # ? fit interest in that case
    # ax.plot(xs, fxs, color='tab:red', linewidth=2, alpha=0.8)

    # labels
    # ax.set_xlabel('_'.join(cols[0].split('_')[1:]))
    # ax.set_ylabel('_'.join(cols[1].split('_')[1:]))
    txt = '{}-{}'.format(select[2].split('_')[1:][0], select[1].split('_')[1:][0])
    ax.set_ylabel(txt)

    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)


    # diff/ref
    ax = fig.add_subplot(223)
    subdf = pd.DataFrame(df[select[1]].copy())
    txt = '{}-{}'.format(select[2].split('_')[1:][0], select[1].split('_')[1:][0])
    subdf[txt] = df[select[2]]-df[select[1]]
    subdf = subdf.dropna()
    cols = subdf.columns
    ax.scatter(subdf[cols[0]], subdf[cols[1]],
               marker='o', s=65, alpha=0.8, color='tab:blue')
    med = subdf[cols[0]].median()
    mad = subdf[cols[0]].mad()
    ax.axvspan(med-mad, med+mad, color='tab:blue', alpha=0.3)
    txt = '{:.0f}±{:.0f}'.format(med, mad)
    ax.text(med, ax.get_ylim()[1], txt, 
            va='top', ha='center', color='tab:blue')
    med = subdf[cols[1]].median()
    mad = subdf[cols[1]].mad()
    ax.axhline(med, color='tab:blue', linewidth=3, alpha=0.7)
    ax.axhline(med + 2*mad, color='tab:blue',
               linewidth=2, linestyle=':', alpha=0.7)
    ax.axhline(med - 2*mad, color='tab:blue',
               linewidth=2, linestyle=':', alpha=0.7)
    ax.axhline(0, color='k')
    # labels
    ax.set_ylabel('{}'.format(cols[1]))
    ax.set_xlabel('{}'.format(cols[0]))
    txt = '{:.0f}±{:.0f}'.format(med, mad)
    ax.text(ax.get_xlim()[1], med, txt, 
            va='bottom', ha='right', color='tab:blue')
    ax.text(ax.get_xlim()[1], med + 2*mad, 'med+2*mad',
            va='bottom', ha='right', color='tab:blue')
    # regress
    slope, intercept = np.polyfit(subdf[cols[0]], subdf[cols[1]], 1)
    xs = (subdf[cols[0]].min(),
          subdf[cols[0]].max())
    fxs = (intercept + slope * xs[0],
           intercept + slope * xs[1])
    ax.plot(xs, fxs, color='tab:red', linewidth=2, alpha=0.8)

    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)

    # diff/mean
    ax = fig.add_subplot(224)
    subdf = pd.DataFrame(df[select[2]] - df[select[1]]).dropna()
    txt = '{}-{}'.format(select[2].split('_')[1:][0], select[1].split('_')[1:][0])
    subdf.columns= [txt]
    subdf['moy'] = df[[select[2], select[1]]].mean(axis=1)
    cols = subdf.columns
    ax.plot(subdf[cols[1]], subdf[cols[0]], 'o', alpha=0.8, ms=10, color='tab:blue')
    # ax.scatter(((df[select[1]] + df[select[2]])/2).tolist(), (df[select[2]]-df[select[1]]).tolist(),
    #            marker='o', s=65, alpha=0.8, color='tab:blue')
    med = subdf[cols[1]].median()
    mad = subdf[cols[1]].mad()
    ax.axvspan(med-mad, med+mad, color='tab:blue', alpha=0.3)
    txt = '{:.0f}±{:.0f}'.format(med, mad)
    ax.text(med, ax.get_ylim()[1], txt, 
            va='top', ha='center', color='tab:blue')
    med = subdf[cols[0]].median()
    mad = subdf[cols[0]].mad()
    ax.axhline(med, color='tab:blue', linewidth=3, alpha=0.7)
    ax.axhline(med + 2*mad, color='tab:blue',
               linewidth=2, linestyle=':', alpha=0.7)
    ax.axhline(med - 2*mad, color='tab:blue',
               linewidth=2, linestyle=':', alpha=0.7)
    ax.axhline(0, color='k')
    ax.set_ylabel(cols[0])
    ax.set_xlabel('{} d0 d1'.format(cols[1]))
    txt = '{:.0f}±{:.0f}'.format(med, mad)
    ax.text(1, 0.55, txt, va='bottom', ha='right',
            transform=ax.transAxes, color='tab:blue')
    ax.text(ax.get_xlim()[1], med + 2*mad, 'med+2*mad',
            va='bottom', ha='right', color='tab:blue')

    # regress
    slope, intercept = np.polyfit(subdf[cols[1]], subdf[cols[0]], 1)
    xs = (subdf[cols[1]].min(),
          subdf[cols[1]].max())
    fxs = (intercept + slope * xs[0],
           intercept + slope * xs[1])
    ax.plot(xs, fxs, color='tab:red', linewidth=2, alpha=0.8)

    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)

    if anot:
        txt = 'file= {} ({})'.format(params.get(sheet, sheet), sheet)
        fig.text(0.5, 0.01, txt,
                 ha='center', va='bottom', alpha=0.4)
        date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        fig.text(0.99, 0.01, 'extra/extra_pop/plot_d1_d0_low',
                 ha='right', va='bottom', alpha=0.4)
        txt = '{} , data <-> sheet {}'.format(date, '_'.join(str(sheet).split('_')[:3]))
        fig.text(0.01, 0.01, txt, ha='left', va='bottom', alpha=0.4)

    fig.tight_layout()
    return fig


plt.close('all')
for sheet in range(2):
    # sheet = 1
    print(sheet)
    data_df = load_latencies(sheet)
    data_df = clean_df(data_df, mult=4)
    fig = plot_d1_d0_low(data_df, sheet)

    save = True
    if save:
        sheet = str(sheet)
        file = 'latOn_d1d0_low_' + str('_'.join(sheet.split('_')[:3])) + '.pdf'
        dirname = os.path.join(paths['owncFig'], 'pythonPreview', 'extra')
        filename = os.path.join(dirname, file)
        fig.savefig(filename)


#%%
plt.close('all')


def plot_d1_d2_high(datadf, sheet, shift=True):
    isi = {0: 27.8, 1: 34.7,
           '1319_CXLEFT_TUN25_s30_csv_test.csv' : 27.8}
    isi_shift = isi.get(sheet, 0)
    select = ['layers', 'latOn_d0_(c)',
              'latOn_d1_s_(25°/s)', 'latOn_d0_(s+c)_150°/s']
    df = datadf[select].copy()
    for col in df.columns[1:]:
        df[col] = df[col].apply(lambda x: x if x < 100 else np.nan)
        df[col] = df[col].apply(lambda x: x if x > 1 else np.nan)
    if shift:
        df[select[3]] = df[select[3]] + isi_shift
    fig = plt.figure(figsize=(12,6))
    fig.suptitle(sheet)
    ax = fig.add_subplot(121)
    ax.scatter(df[select[1]].tolist(), df[select[2]].tolist(),
               marker='o', s=65,
               alpha=0.6, color='tab:blue',
               label='_'.join(select[2].split('_')[1:]))
    label = '_'.join(select[3].split('_')[1:])
    if shift:
        label += '+ {} msec'.format(isi_shift)
    ax.scatter(df[select[1]].tolist(), df[select[3]].tolist(),
               marker='o', s=65,
               alpha=0.6, color='tab:orange',
               label= label)

    lims = (df[df.columns[1:]].min().min() - 5,
            df[df.columns[1:]].max().max() + 5)
    ax.set_ylim(lims)
    ax.set_xlim(lims)
    ax.plot(lims, lims)   # diag
    ax.set_xlabel(select[1])
    ax.set_ylabel('_'.join(select[3].split('_')[:1]))
    ax.legend()
    ax = fig.add_subplot(122)
    ax.plot(df.index, df[select[2]] - df[select[3]], 'o', alpha = 0.8, ms=10,
            color='tab:blue')
    med = (df[select[2]] - df[select[3]]).median()
    mad = (df[select[2]] - df[select[3]]).mad()
    txt = '{:.0f} ± {:.0f} msec'.format(med, mad)
    ax.text(1, 0.6, txt, va='bottom', ha='right',
           transform=ax.transAxes, color='tab:blue')
    ax.axhline(med, color='tab:blue', linewidth=3, alpha=0.5)
    ax.set_ylabel('{}  minus  {}'.format(select[2], select[3]))
    #ax.set_ylim((ax.get_ylim)()[::-1])
    ax.axhline(0, color='tab:gray')
    ax.set_xlabel('depth')
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    return fig


plt.close('all')
for sheet in range(2):
    # sheet = 1
    print(sheet)
    data_df = load_latencies(sheet)
    data_df = clean_df(data_df, mult=4)
    fig = plot_d1_d2_high(data_df, sheet)

    save=False
    if save:
        sheet = str(sheet)
        file = 'latOn_d2d1_high_' + str('_'.join(sheet.split('_')[:3])) + '.pdf'
        dirname = os.path.join(paths['owncFig'], 'pythonPreview', 'extra')
        filename = os.path.join(dirname, file)
        fig.savefig(filename)
