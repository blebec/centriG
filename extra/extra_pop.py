#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 21 12:42:53 2021

@author: cdesbois
"""

import os
from datetime import datetime
from math import floor, ceil
# from importlib import reload

# import matplotlib.gridspec as gridspec
# import matplotlib.patches as patches
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Rectangle
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


def load_csv(filename):
    """
    load the csv file containing the latencies

    """
    # filename = '/Users/cdesbois/ownCloud/cgFigures/data/data_extra/2019_CXRIGHT_TUN21_s30_csv_test_noblank.csv'
    # header
    header = []
    with open(filename, 'r') as f:
        for i, line in enumerate(f):
            if i > 1:
                break
            else:
                header.append(line)
    header = [_.replace('\n', '') for _ in header]
    header[0] = header[0].split(';')
    header[0] = [_.replace('total number of stim', 'nbstim') for _ in header[0]]
    header[0] = [_.replace(' ', '') for _ in header[0]]
    header[1] = header[1].replace('speed;', 'speed:')
    temp = [tuple(_.split(':')) for _ in header[0]]
    params = {a:b for a,b in temp}
    params['nbstim'] = int(float(params['nbstim']))
    params['blank'] = params['blank'].lower() in ['true', 1]
    temp = header[1].split(':')
    temp[1] = [float(_) for _ in temp[1].split(';') if _]
    temp[1] = [int(_) for _ in temp[1] if _]
    params.update({temp[0]: temp[1]})
    del temp, header
    params['file'] = os.path.basename(file_name)

    # blocs de 3 3 stim = une vitesse
    # avant dernière = blanc
    # dernière = D1 sc 150 °/C
    # avant dernière = blank
    # t0 = D1 pour s0 et dernière SC150
    # t0 = D0 pour le reste (including blank)

    df = pd.read_csv(filename, skiprows=2, sep=';', header=None)
    df = df.set_index(df.columns[0]).T
    cols = df.columns
    cols = [_.replace('_latency_PG0.VEC_PST_HALF_LATENCY_TOP_AE', '') for _ in cols]
    cols = [_.replace('set_latency_PG0.FIRST_CROSS_LIST', '') for _ in cols]
    cols = [_.replace('egral_PG0.INTEGRAL_LATENCY_LIST', '') for _ in cols]
    cols = [_.replace('nificativity_PG0.SIG_LIST_LAT', '') for _ in cols]
    #______________________
    # rename the conditions
    hhs = [_ for _ in cols if _.startswith('hh')]
    ons = [_ for _ in cols if _.startswith('on')]
    ints = [_ for _ in cols if _.startswith('int')]
    sigs = [_ for _ in cols if _.startswith('sig')]

    speeds = params['speed']
    speeds = [str(_) for _ in speeds]
    speeds = [x for three in zip(speeds,speeds,speeds) for x in three]
    speeds.append('00')
    speeds.append('150')
    print('='*30)
    print('len(speeds)={}  len(ons)={}'.format(len(speeds), len(ons)))

    stims = ['0c', 's0', 'sc'] * (len(ons)//3)
    stims.append('00')
    stims.append('sc')
    print('='*30)
    print('len(stims)={}  len(ons)={}'.format(len(stims), len(ons)))

    times = ['d1' if _ == 's0' else 'd0' for _ in stims ]
    times[-1] = 'd1'
    print('='*30)
    print('len(times)={}  len(ons)={}'.format(len(times), len(ons)))
    # replace names
    protocs = ['_'.join(('on',a,b,c)) for a,b,c in zip(times, stims, speeds)]
    for i, col in enumerate(protocs,1):
        index = cols.index('on[{}]'.format(i))
        cols[index] = col
    protocs = ['_'.join(('int',a,b,c)) for a,b,c in zip(times, stims, speeds)]
    for i, col in enumerate(protocs,1):
        index = cols.index('int[{}]'.format(i))
        cols[index] = col
    cols = [_.replace('sig[1]', 'significancy') for _ in cols]
# TODO = to be checked
    protocs = ['_'.join(('hh',a,b,c)) for a,b,c in zip(times, stims, speeds)]
    protocs = protocs[:len(hhs)]        # not all teh conditions are present
    for i, col in enumerate(protocs,1):
        for item in cols:
            if 'hh[' in item:
                index = cols.index(item)
                cols[index] = col
                print(item)
                break
    df.columns = cols

    # remove the last line
    df = df.dropna()
    # normalise sig (ie 1 or 0 instead of 10 or 0)
    for col in df.columns:
        if 'sig' in col:
            print(col)
            df[col] = df[col]/10
            df[col] = df[col].astype(bool)
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
    return df, params

#%
def load_latencies(sheet='0'):
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
    sheet = ['EXP 1', 'EXP 2'][int(sheet)]
    paths['xcel'] = os.path.join(paths['owncFig'], 'infos_extra')
    file = 'Tableau_info_integrales_latences.xlsx'
    filename = os.path.join(paths['xcel'], file)
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

        # selection = ['on_d0_0c_25', 'on_d0_sc_25', 'on_d0_sc_150',
        #         'on_d1_s0_25', 'on_d1_sc_150', 'on_d1_s0_150']

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
    df.rf_bigger_than_stim = df.rf_bigger_than_stim.apply(
        lambda x: False if x=='NO' else True)
    return df


def extract_layers(df):
    """
    Parameters
    ----------
    df : pandas dataFrame with a layer column
    Returns
    -------
    dico : key = layer, value = tuple eletrode number range
    """
    d = 0
    depths = []
    for _ in df.layers.value_counts().values[:-1]:
        d += _
        depths.append(d)
    depths.insert(0, 0)
    depths.append(df.index.max())
    vals = list(zip(depths[:-1], depths[1:]))
    keys = df.layers.unique()
    dico = dict(zip(keys, vals))
    return dico

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
                df.loc[df[col] == 'None', [col]] = np.nan
            else:
                ser = df[col]
                med = ser.median()
                mad = ser.mad()
                if (len(ser.loc[ser > (med + mult * mad)]) > 0) or \
                    len(ser.loc[ser < (med - mult * mad)]) > 0:
                    num = len(ser.loc[ser > (med + 3 * mad)])
                    num += len(ser.loc[ser < (med - 3 * mad)])
                    print ('{:.0f} values to remove for {:s}'.format(
                        num, ser.name))
                    total += num
                    # print('-' * 10)
                    df.loc[df[col] < (med - 3*mad), [col]] = np.nan
                    df.loc[df[col] > (med + 3*mad), [col]] = np.nan
                    # df[col] = df[col].apply(
                    #     lambda x: x if x > (med - 3 * mad) else np.nan)
                    # df[col] = df[col].apply(
                    #     lambda x: x if x < (med + 3 * mad) else np.nan)
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


csvLoad = True
if csvLoad:
    paths['data'] = os.path.join(paths['owncFig'], 'data', 'data_extra')
    files = [file for file in os.listdir(paths['data']) if file[:4].isdigit()]
    # files = ['1319_CXLEFT_TUN25_s30_csv_test_noblank.csv',
    #          '2019_CXRIGHT_TUN21_s30_csv_test_noblank.csv']
    file = files[1]
    sheet=file
    file_name = os.path.join(paths['data'], file)
    data_df, params = load_csv(file_name)
else:
    params = {'0' : '1319_CXLEFT',
              '1' : '2019_CXRIGHT'}
    sheet = '0'
    data_df = load_latencies(sheet)


layers_loc = extract_layers(data_df)
# only significant part
data_df = data_df[data_df.significancy]
#data_df = data_df[data_df.significancy]
# clean
data_df = clean_df(data_df, mult=4)
stats_df = data_df.describe()
# stats_df_sig = data_df[data_df.significancy].describe()

#%% desribe basics

def plot_boxplots(datadf, removemax=True, params=params, mes=None):
    """
    stat boxplot description for latencies
    input:
        datadf = pandas dataframe
        removemax : boolean to remove values of 100 msec
        params:  dicionary (containing the speeds used)
        measure in [on, hh and inte], default=None ie all
    """
    ons = [_ for _ in datadf. columns if _.startswith('on')]
    hhs = [_ for _ in datadf.columns if _.startswith('hh')]
    ints = [_ for _ in datadf.columns if _.startswith('int')]
    # group by stimulation

    if mes is None:
        ons = [_ for _ in ons if _.split('_')[-1] in ('25', '150')]
        hhs = [_ for _ in hhs if _.split('_')[-1] in ('25', '150')]
        ints = [_ for _ in ints if _.split('_')[-1] in ('25', '150')]
        conds =  [ons, hhs, ints]
    else:
        dico = {'on':ons, 'hh':hhs, 'inte':ints}
        kind = dico.get(mes, None)
        bysim = sorted(kind, key=lambda x: x.split('_')[2])
        #remove blank
        for item in bysim:
            if item.split('_')[2] == '00':
                bysim.remove(item)
        r = len(params['speed'])
        conds = [bysim[:r], bysim[r:2*r], bysim[2*r:]]

    # remove values of 100 <-> no detection
    if removemax:
        for col in datadf.columns:
            if data_df[col].dtypes == float:
                 datadf.loc[datadf[col] > 100, [col]] = np.nan

    fig, axes = plt.subplots(nrows=1, ncols=3)
    axes = axes.flatten()

    for i, dats in enumerate(conds):
        ax = axes[i]
        txt = dats[0].split('_')[0]
        ax.set_title(txt)
        for col in dats:
            if len(datadf[col].dropna()) == 0:
                dats.remove(col)
        ax.boxplot(datadf[dats].dropna(), meanline=True, showmeans=True)
        labels = ['_'.join(_.split('_')[1:]) for _ in dats]
        med = datadf[dats[0]].median()
        mad = datadf[dats[0]].mad()
        ax.axhline(med, color='tab:blue', linewidth=3, alpha=0.7)
        ax.axhline(med + 2*mad, color='tab:blue',
                   linewidth=2, linestyle=':', alpha=0.7)
        ax.axhline(med - 2*mad, color='tab:blue',
                   linewidth=2, linestyle=':', alpha=0.7)
        ax.set_xticklabels(labels, rotation=45, ha='right')
        txt = 'med ± 2*mad'
        ax.text(x=ax.get_xlim()[0], y=med  + 2* mad, s=txt,
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

# import time
# start_time = time.time()

plt.close('all')
fig = plot_boxplots(data_df, mes='on')
# print("--- %s seconds ---" % (time.time() - start_time))

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
    """
    histograms for all the reponses
    """
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

#%%
def plot_on_histo(datadf, removemax=True, sheet=sheet,
                  diff=False, shift=False, hh=False):

    df = datadf.copy()
    # extract columns of interest
    ons = [_ for _ in df. columns if _.startswith('on')]
    hhs = [_ for _ in df.columns if 'hh' in _]
    ints = [_ for _ in df.columns if 'int' in _]
    # limit for csv file
    ons = [_ for _ in ons if _.split('_')[-1] in ('25', '150')]
    ons.remove('on_d0_0c_150')
    hhs = [_ for _ in hhs if _.split('_')[-1] in ('25', '150')]
  #  hhs.remove('hh_d0_0c_150')
    ints = [_ for _ in ints if _.split('_')[-1] in ('25', '150')]
    if hh:
        cols = hhs
    else:
        cols = ons
    dats = [_ for _ in cols if 'd0' in _]
    dats += [_ for _ in cols if 'd1' in _]

    isi = {'0': 27.8, '1': 34.7,
           '1319_CXLEFT_TUN25_s30_csv_test.csv' : 27.8,
           '1319_CXLEFT_TUN25_s30_csv_test_noblank.csv' : 27.8}
    isi_shift = isi.get(sheet, 0)
    if shift:
        df['on_d0_sc_150'] += isi_shift
        # remove values of 100 <-> no detection
    if removemax:
        for col in df.columns:
            if df[col].dtypes == float:
                df.loc[df[col] > 100, col] = np.nan
    # plotting
    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(15,12),
                             sharex=True, sharey=True)
    txt = '{} {}'.format(params.get(sheet, sheet), 'on latencies')
    if diff:
        txt = '{} ({})'.format(txt, 'cond - centerOnly')
    if shift:
        txt = '{} ({})'.format(txt, 'on_d0_sc_150 shifted')
    fig.suptitle(txt)

    axes = axes.flatten()
    colors = ('tab:blue', 'tab:red', 'tab:orange',
              'tab:red', 'tab:orange', 'tab:green')
    med = df[dats[0]].median()
    mad = df[dats[0]].mad()
    maxi = 0
    for i, col in enumerate(dats):
        ax = axes[i]
        if diff:
            ser = (df[dats[0]] - df[col]).dropna()
            med = ser.median()
            mad = ser.mad()
            ax.axvspan(med - mad, med+mad, color= colors[i], alpha=0.3)
            ax.axvline(med, color=colors[i], alpha=0.5, linewidth=2)
        else :
            ser = df[col].dropna()
            ax.axvspan(med - mad, med+mad, color='tab:blue', alpha=0.3)
            ax.axvline(med, color='tab:blue', alpha=0.5, linewidth=2)
        # txt = '_'.join(col.split('_')[1:])
        txt = col
        ax.hist(ser, bins=20, color=colors[i], alpha=0.7, rwidth=0.9)
        # scale if diff
        maxi = max(0, max(np.histogram(ser, bins=20)[0]))
        ax.set_title(txt)
        ax.axvline(0, color='tab:grey', linewidth=2, alpha = 0.5)
    for ax in axes:
        for spine in ['top', 'right']:
            ax.spines[spine].set_visible(False)
    if diff:
        ax.set_ylim(0, maxi)
        for ax in axes:
            ax.axvline(0, color='tab:blue', linewidth=2)

    for i in [0,3]:
        axes[i].set_ylabel('nb of electrodes')
    for i in [3,4,5]:
        axes[i].set_xlabel('time')

    if anot:
        txt = 'file= {} ({})'.format(params.get(sheet, sheet), sheet)
        fig.text(0.5, 0.01, txt,
                 ha='center', va='bottom', alpha=0.4)
        date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        fig.text(0.99, 0.01, 'extra/extra_pop/plot_on_histo',
                 ha='right', va='bottom', alpha=0.4)
        txt = '{} {}'.format(date, '_'.join(str(sheet).split('_')[:3]))
        fig.text(0.01, 0.01, txt, ha='left', va='bottom', alpha=0.4)
    fig.tight_layout()
    return fig

def save_fig(fig, diff, shift, hh, sheet, paths):
    sheet = str(sheet)
    txt = 'latOn_histo_' + str('_'.join(sheet.split('_')[:3]))
    if hh:
        txt += '_hh'
    if shift:
        txt += '_shift'
    if diff:
        txt += '_diff'
    file = txt + '.pdf'
    dirname = os.path.join(paths['owncFig'], 'pythonPreview', 'extra')
    filename = os.path.join(dirname, file)
    fig.savefig(filename)


plt.close('all')
# what=[True, False]
# for diff in what:
#     for shift in what :
#         for hh in what:
#             fig = plot_on_histo(data_df, diff=diff, shift=shift, hh=hh, removemax=True)
#           #  save_fig(fig, diff, shift, hh, sheet, paths)

fig = plot_on_histo(data_df, diff=False, shift=False, hh=True, removemax=True)

# save_fig(fig, diff, shift, hh)

#%% scatter individuals
def plot_on_scatter(datadf, removemax=True, sheet=sheet,
                  diff=False, shift=False, hh=False, layersloc=layers_loc):

    df = datadf.copy()
    ons = [_ for _ in df. columns if _.startswith('on')]
    hhs = [_ for _ in df.columns if _.startswith('hh')]
    ints = [_ for _ in df.columns if _.startswith('int')]
    # limit for csv file
    ons = [_ for _ in ons if _.split('_')[-1] in ('25', '150')]
    ons.remove('on_d0_0c_150')
    hhs = [_ for _ in hhs if _.split('_')[-1] in ('25', '150')]
    ints = [_ for _ in ints if _.split('_')[-1] in ('25', '150')]
    # kind of plot
    if hh:
        cols = hhs
    else:
        cols = ons
    dats = [_ for _ in cols if _.startswith('on_d0')]
    dats += ([_ for _ in cols if _.startswith('on_d1')])



    isi = {'0': 27.8, '1': 34.7,
           '1319_CXLEFT_TUN25_s30_csv_test.csv' : 27.8,
           '2019_CXRIGHT_TUN21_s30_csv_test_noblank.csv' : 34.7}
# TODO check (old = 27.8 & 34.7 new ? and 21 for 150°/sec
# ? ISI + stroke duration
    isi_shift = isi.get(sheet, 0)
    if shift:
        df['on_d0_sc_150'] += isi_shift
    # remove values of 100 <-> no detection
    if removemax:
        for col in df.columns:
            if df[col].dtypes == float:
                df.loc[df[col] == 100, col] = np.nan
    if hh:
        ons = hhs
    # plotting
    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(15,12),
                             sharex=True, sharey=True)
    txt = '{} {}'.format(params.get(sheet, sheet), 'on latencies')
    if diff:
        txt = '{} ({})'.format(txt, 'cond - centerOnly')
    if shift:
        txt = '{} ({})'.format(txt, 'on_d0_sc_150 shifted')
    fig.suptitle(txt)

    axes = axes.flatten()
    colors = ('tab:blue', 'tab:red', 'tab:orange', 'tab:red', 'tab:orange', 'tab:green')
    # global stat
    ref_med = df[ons[0]].median()
    ref_mad = df[ons[0]].mad()

    bylayer = pd.DataFrame(layersloc, index=['dmin', 'dmax']).T
    bylayer['ref_meds'] = df.groupby('layers')[ons[0]].median()
    bylayer['ref_mads'] = df.groupby('layers')[ons[0]].mad()

    for i, col in enumerate(ons):
        ax = axes[i]
        #layers
        ax.axhspan(bylayer.loc['4'].dmin, bylayer.loc['4'].dmax, color='tab:grey', alpha=0.2)
        if diff:
            # ser = (df[ons[0]] - df[col]).dropna()
            df['toplot'] = df[ons[0]] - df[col]
            temp = df[['layers', 'toplot']].dropna()
        else :
            # ser = df[col].dropna()
            temp = df[['layers', col]].dropna()
            df['toplot'] = df[col]
            temp = df[['layers', 'toplot']].dropna()
            # refs
            # for j, (med, mad) in enumerate(zip(ref_meds, ref_mads)):
            for j in range(len(bylayer)):
                ymin, ymax, med, mad = bylayer.iloc[j][
                    ['dmin', 'dmax', 'ref_meds', 'ref_mads']]
                # ymin = depths[j]
                # ymax = depths[j+1]
                if not np.isnan(med):
                    ax.vlines(med, ymin=ymin, ymax=ymax, color='tab:blue',
                              alpha=0.5, linewidth=2)
                    ax.vlines(med-mad, ymin=ymin, ymax=ymax, color='tab:blue',
                              alpha=0.5, linewidth=2, linestyle=':')
                    ax.vlines(med+mad, ymin=ymin, ymax=ymax, color='tab:blue',
                              alpha=0.5, linewidth=2, linestyle=':')
        # intervals
        # for j in range(len(depths[:-1])):
            # ymin, ymax = depths[j], depths[j+1]
            # med = ser.loc[ymin:ymax].median()
            # mad = ser.loc[ymin:ymax].mad()
        bylayer['meds'] = temp.groupby('layers').median()
        bylayer['mads'] = temp.groupby('layers').mad()
        for j in range(len(bylayer)):
            ymin, ymax, med, mad = bylayer.iloc[j][
                ['dmin', 'dmax', 'meds', 'mads']]
            # test if values are present
            if not np.isnan(med):
                ax.vlines(med, ymin=ymin, ymax=ymax, color=colors[i],
                           alpha=0.5, linewidth=2)
                ax.add_patch(Rectangle((med - mad, ymin), width=2*mad,
                                       height=(ymax - ymin), color=colors[i],
                                       alpha=0.3, linewidth=0.5))
        txt = col
        ax.scatter(temp.toplot.values, temp.index, color=colors[i], alpha=0.7)
        # scale if diff
        # maxi = max(0, max(np.histogram(ser, bins=20)[0]))
        ax.set_title(txt)
        ax.axvline(0, color='tab:grey', linewidth=2, alpha = 0.5)

    # ax.set_ylim(ax.get_ylim()[::-1])
    ax.set_ylim(bylayer.dmax.max(), bylayer.dmin.min())

    for ax in axes:
        for spine in ['top', 'right']:
            ax.spines[spine].set_visible(False)
    if diff:
        for ax in axes:
            ax.axvline(0, color='tab:blue', linewidth=2)

    for i in [0,3]:
        axes[i].set_ylabel('electrode (depth)')
    for i in [3,4,5]:
        axes[i].set_xlabel('time')

    if anot:
        txt = 'file= {} ({})'.format(params.get(sheet, sheet), sheet)
        fig.text(0.5, 0.01, txt,
                 ha='center', va='bottom', alpha=0.4)
        date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        fig.text(0.99, 0.01, 'extra/extra_pop/plot_on_scatter',
                 ha='right', va='bottom', alpha=0.4)
        txt = '{} {}'.format(date, '_'.join(str(sheet).split('_')[:3]))
        fig.text(0.01, 0.01, txt, ha='left', va='bottom', alpha=0.4)
    fig.tight_layout()
    return fig

def save_scatter(fig, diff, shift, hh, sheet=sheet, paths=paths):
    sheet = str(sheet)
    txt = 'on_scatter_' + str('_'.join(sheet.split('_')[:3]))
    if hh:
        txt += '_hh'
    if shift:
        txt += '_shift'
    if diff:
        txt += '_diff'
    file = txt + '.pdf'
    dirname = os.path.join(paths['owncFig'], 'pythonPreview', 'extra')
    filename = os.path.join(dirname, file)
    fig.savefig(filename)


plt.close('all')
# diff = True
# shift = True
# hh = False
# what=[True, False]
# for diff in what:
#     for shift in what :
#         for hh in what:
#             fig = plot_on_scatter(data_df, diff=diff, shift=shift, hh=hh,
#                                   removemax=True)
#             save_scatter(fig, diff, shift, hh)

fig = plot_on_scatter(data_df, diff=False, shift=True,
                      hh=False, removemax=False)


#%% test dotplot
plt.rcParams.update({'axes.titlesize': 'medium'})

#TODO use floats not integers
# indiquer le nombre de réponses/conditions
# indiquer le stroke interval (frame ~ 7 msec)
# cahanger D0 par t0 = D0, t0 = D1


def plot_latencies(datadf, lat_mini=10, lat_maxi=80, sheet=sheet, xcel=False,
                   layersloc=layers_loc):
    """
    plot the latencies
    input :
        df : pandasDataFrame
        lat_mini : start time to use (values before are dropped)
        lat_maxi : end time to use (values after are removed)
    output :
        matplotlib figure
    """
    isi = {'0': 27.8, '1': 34.7,
           '1319_CXLEFT_TUN25_s30_csv_test.csv' : 27.8,
           '2019_CXRIGHT_TUN21_s30_csv_test_noblank.csv' : 34.7}
    isi_shift = isi.get(sheet, 0)
    #data filtering
    # xcel = False
    if xcel:
        df = datadf[datadf.columns[[1,3,4,5,6,8,7]]].copy()
        selection = ['on_d0_0c_25', 'on_d0_sc_25', 'on_d0_sc_150',
                     'on_d1_s0_25', 'on_d1_s0_150', 'on_d1_sc_150']
    else:
        selection = ['on_d0_0c_25', 'on_d0_sc_25', 'on_d0_sc_150',
                 'on_d1_s0_25', 'on_d1_sc_150', 'on_d1_s0_150']
    selection.insert(0, 'layers')
    df = datadf[selection].copy()
    cols = df.columns[1:]
    for col in cols:
        df.loc[df[col] > lat_maxi, col] = np.nan
        df.loc[df[col] < lat_mini, col] = np.nan
    # select columns (d0 based, d1 based)
    d0_cols = df.columns[:4]
    d1_cols = df.columns[[0,4,5,6]]
    # depth
    bylayer = pd.DataFrame(layersloc, index=['dmin', 'dmax']).T

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
    # k = 0
    # dcols = d0_cols
        ax = [ax0, ax1][k]
        vax = v0 #[v0, v1][k]
        hax = [h0, h1][k]
        for i, col in enumerate(dcols[1:]):     # drop layers column
        # i = 0
        # col = cols[i]
            # depth / time
            x = df[col].values.tolist()     # latency value in [0, 100] msec
            y = df[col].index.tolist()      # electrodes / depths
            ax.plot(x, y, '.', color=colors[i+k], markersize=10, alpha=0.5,
                    label=col)
            # ax.axvline(cop[col].median(), color=colors[i], alpha=0.5, linewidth=3)
            # meds = df.groupby('layers')[col].median()
            # mads = df.groupby('layers')[col].mad()
            bylayer['meds'] = df.groupby('layers')[col].median()
            bylayer['mads'] = df.groupby('layers')[col].mad()
            for j in range(len(bylayer)):
                ymin, ymax, med, mad = bylayer.iloc[j][
                    ['dmin', 'dmax', 'meds', 'mads']]
                # test if values are present
                if not np.isnan(med):
                    ax.vlines(med, ymin, ymax, color=colors[i+k],
                          alpha=0.5, linewidth=3)
            # for j, med in enumerate(meds):
            #     ax.vlines(med, depths[j], depths[j+1], color=colors[i+k],
            #               alpha=0.5, linewidth=3)
            ax.legend(loc='upper right')
            txt = 'med : {:.0f}±{:02.0f} ({:.0f}, {:.0f}, {:.0f})'.format(
                df[col].median(), df[col].mad(),
                bylayer.meds.values[0], bylayer.meds.values[1],
                bylayer.meds.values[2])
            ax.text(x=1, y=0.43 - i/8, s=txt, color=colors[i+k],
                    va='bottom', ha='right', transform=ax.transAxes)

            ## vertical histogramm
            # v_height, v_width = np.histogram(x, bins=20, range=(0,100),
            #                                  density=True)
            # vax.bar(v_width[:-1], v_height, width=5, color=colors[i+k],
            #         align='edge', alpha=0.4)
            xvals = [_ for _ in x if not np.isnan(_)]
            if len(xvals) > 0:
                kde = stats.gaussian_kde([_ for _ in x if not np.isnan(_)])
                x_kde = np.linspace(0,100, 20)
                if k == 0:
                    vax.plot(x_kde, kde(x_kde), color=colors[i+k], alpha=0.6,
                             linewidth=2)
                else:
                    vax.plot(x_kde, kde(x_kde), color=colors[i+k], alpha=0.6,
                             linestyle=':', linewidth=3)

            ## horizontal histogramm (with nan not plotted)
            bylayer['y'] = [0.3*(i-1)+_ for _ in range(len(bylayer))]
            # med - mad / med + mad
            bylayer['width'] = 2 * bylayer['mads']
            bylayer['left'] = bylayer.meds - bylayer.mads
            y, width, left = bylayer[['y', 'width','left']].dropna().T.values.tolist()
            hax.barh(y=y,
                     width=width,
                     left=left,
                     height=0.3, color=colors[i+k], alpha=0.4)
            # med
            bylayer['width'] = 1
            bylayer['left'] = bylayer.meds - 0.5
            y, left = bylayer[['y', 'left']].dropna().T.values.tolist()
            hax.barh(y=y,
                     width=1,
                     left=left,
                     height=0.3, color=colors[i+k], alpha=1)
    # shift on vertical histo
    col = d0_cols[3]        # 'latOn_d0_(s+c)_150°/s'
    x = (df[col] + isi_shift).values.tolist()     # latency value in [0, 100] msec
    y = df[col].index.tolist()      # electrodes / depths
    kde = stats.gaussian_kde([_ for _ in x if not np.isnan(_)])
    x_kde = np.linspace(0,100, 20)
    v0.plot(x_kde, kde(x_kde), color='tab:grey', alpha=0.5, linestyle='-.',
                         linewidth=2, label='150°/sec_I.S.I._shifted')
    v0.legend(loc=2)
    # shift on horizontal histo
    bylayer['y'] = [0.3*(2-1)+_ for _ in range(len(bylayer))]
    bylayer['meds'] = df.groupby('layers')[col].median() + isi_shift
    bylayer['mads'] = df.groupby('layers')[col].mad()
    # med - mad / med + mad
    bylayer['width'] = 2 * bylayer['mads']
    bylayer['left'] = bylayer.meds - bylayer.mads
    y, width, left = bylayer[['y', 'width','left']].dropna().T.values.tolist()
    h0.barh(y=y,
             width=width,
             left=left, height=0.3, color='tab:grey', alpha=0.4)
    # med
    bylayer['width'] = 1
    bylayer['left'] = bylayer.meds - 0.5
    y, left = bylayer[['y', 'left']].dropna().T.values.tolist()
    h0.barh(y=y,
             width=1,
             left=left,
             height=0.3, color='tab:grey', alpha=1)

    ## plot nb of cells (by layer -> pb if some layers are not presented)
    bylayer = bylayer.join(df.groupby('layers').count())
    allcolors = colors[:-1] + colors[1:]
    alphas = [0.4, 0.6, 0.4]
    # nb of protocols
    protocols = df.columns[1:]
    x = list(range(len(protocols)))
    # cells by layer
    for i in range(len(bylayer)):
        y = bylayer[protocols].iloc[i].values
        c0.bar(x=x, height=y, bottom=bylayer.iloc[i].dmin,
               alpha=alphas[i], color=allcolors, edgecolor='tab:grey')

    # references lines
    v0.axvline(df[d0_cols[1]].median(), color='tab:blue', alpha=0.5)
    h0.axvline(df[d0_cols[1]].median(), color='tab:blue', alpha=0.5)
    h1.axvline(df[d0_cols[1]].median(), color='tab:blue', alpha=0.5)
    v0.set_title('KDE (plain D0, dotted D1)', color='tab:grey')

    # scatters
    ax1.set_xlabel('msec')
    ax1.set_xlim((0, 100))
    # ax1.set_ylim(ax.get_ylim()[::-1])    # O=surfave, 65=deep
    ax1.set_ylim(bylayer.dmax.max(), bylayer.dmin.min())
    for i, ax in enumerate([ax0, ax1]):
        ax.set_ylabel('depth')
        ax.set_xlabel('D{} based time (msec)'.format(i))
        for spine in ['top', 'right']:
            ax.spines[spine].set_visible(False)
        for d in bylayer.dmin:
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
        c0.set_ylim(bylayer.dmax.max(), bylayer.dmin.min())
        for d in bylayer.dmin:
            ax.axhline(d, color='tab:grey', alpha=0.5)

        ax.set_xticks([])
        # ax.set_ylim(64, 0)
        # ax.set_title('responses detected')
        ax.set_xlabel('protocols', color='tab:grey')
        ax.set_ylabel('depth')
        ax.set_title('nb of detections', color='tab:grey')

    # horizontal histo
    h0.set_title('med ± mad', color='tab:grey')
    h0.set_ylim(h0.get_ylim()[::-1])
    labels = list(df.layers.unique())
    labels = ['2/3', '4', '5/6']
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
        txt = '{} , data <-> sheet {} : {}'.format(date,
                                                 '_'.join(str(sheet).split('_')[:3]),
                                                 params.get(sheet, ''))
        fig.text(0.01, 0.01, txt, ha='left', va='bottom', alpha=0.4)
    fig.tight_layout()

    v0.text(x=1, y=0.5, s='KDE, plain <-> D0, dotted <-> D1', color='tab:gray',
            va='bottom', ha='right', transform=ax.transAxes)
    return fig


new = False
if new :
    sheet = '0'
    data_df = load_latencies(sheet)
    data_df = data_df[data_df.significancy]
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

def plot_latencies_bis(datadf, lat_mini=10, lat_maxi=80, sheet=sheet,
                       layersloc=layers_loc, xcel=True):
    """
    plot the latencies
    input :
        df : pandasDataFrame
        lat_mini : start time to use (values before are dropped)
        lat_maxi : end time to use (values after are removed)
    output :
        matplotlib figure
    """
    isi = {'0': 27.8, '1': 34.7,
           '1319_CXLEFT_TUN25_s30_csv_test.csv' : 27.8}
    isi_shift = isi.get(str(sheet), 0)
    #data filtering
    # xcel = False
    if xcel:
        df = datadf[datadf.columns[[1,3,4,5,6,8,7]]].copy()
        selection = ['on_d0_0c_25', 'on_d0_sc_25', 'on_d0_sc_150',
                     'on_d1_s0_25', 'on_d1_s0_150', 'on_d1_sc_150']
    else:
        selection = ['on_d0_0c_25', 'on_d0_sc_25', 'on_d0_sc_150',
                 'on_d1_s0_25', 'on_d1_sc_150', 'on_d1_s0_150']
    selection.insert(0, 'layers')
    df = datadf[selection].copy()
    cols = df.columns[1:]
    for col in cols:
        df.loc[df[col] > lat_maxi, col] = np.nan
        df.loc[df[col] < lat_mini, col] = np.nan
    # select columns
    # d0_cols = df.columns[:4]
    # d1_cols = df.columns[[0,4,5,6]]

    cols = ['layers', 'on_d0_sc_25', 'on_d0_sc_150', 'on_d1_sc_150']

    # layer depths limits
     # depth
    bylayer = pd.DataFrame(layersloc, index=['dmin', 'dmax']).T
    # d = 0
    # depths = []
    # for _ in df.layers.value_counts().values[:-1]:
    #     d += _
    #     depths.append(d)
    # depths.insert(0, 0)
    # depths.append(df.index.max())

    colors = ['tab:red', 'tab:orange', 'tab:green']

    fig = plt.figure()
    ax = fig.add_subplot(111)
    for i, col in enumerate(cols[1:]):
        x = df.index
        y = df[col]
        label = col
        if col == 'on_d0_sc_150':
            y += isi_shift
            label += '_shifted'
            print('shifted {} by {} msec'.format(col, isi_shift))
        ax.plot(y, x, '.',
                color=colors[i], alpha=0.7, ms=10, label=label)
        ##########
        bylayer['meds'] = df.groupby('layers')[col].median()
        bylayer['mads'] = df.groupby('layers')[col].mad()
        for j in range(len(bylayer)):
            ymin, ymax, med, mad = bylayer.iloc[j][
                ['dmin', 'dmax', 'meds', 'mads']]
            # test if values are present
            if not np.isnan(med):
                ax.vlines(med, ymin, ymax, color=colors[i],
                    alpha=0.5, linewidth=3)
        txt = 'med : {:.0f}±{:02.0f} ({:.0f}, {:.0f}, {:.0f})'.format(
            df[col].median(), df[col].mad(),
            bylayer.meds.values[0], bylayer.meds.values[1],
            bylayer.meds.values[2])
        ax.text(x=1, y=0.43 - i/8, s=txt, color=colors[i],
                va='bottom', ha='right', transform=ax.transAxes)
    # for d in depths:
    #     ax.axhline(d, color='tab:grey', alpha=0.5)
    ax.axhspan(bylayer.dmin.values[1], bylayer.dmin.values[2],
               color='tab:grey', alpha=0.4)

    ax.legend(loc='upper right')
    ax.set_xlim(0, 100)
    # ax.set_ylim(ax.get_ylim()[::-1])
    ax.set_ylim(bylayer.dmax.max(), bylayer.dmin.min())

    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)

    ax.set_ylabel('depth')
    ax.set_xlabel('time (msec)')

    if anot:
        txt = 'out of [{}, {}] msec range were excluded'.format(lat_mini, lat_maxi)
        fig.text(0.5, 0.01, txt, ha='center', va='bottom', alpha=0.4)
        date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        fig.text(0.99, 0.01, 'extra/extra_pop/plot_latencies_bis',
                 ha='right', va='bottom', alpha=0.4)
        txt = '{} , data <-> sheet {} : {}'.format(date,
                                                 '_'.join(str(sheet).split('_')[:3]),
                                                 params.get(sheet, ''))
        fig.text(0.01, 0.01, txt, ha='left', va='bottom', alpha=0.4)
    fig.tight_layout()

    return fig

plt.close('all')

fig = plot_latencies_bis(data_df, lat_mini=0, lat_maxi=80, sheet=sheet, xcel=True)

new = False
if new :
    sheet = 1
    data_df = load_latencies(sheet)
    data_df = data_df[data_df.significancy]
    data_df = clean_df(data_df, mult=4)

#sheet = file

plt.close('all')
fig = plot_latencies_bis(data_df, lat_mini=0, lat_maxi=80, sheet=sheet, xcel=True)

save = False
if save:
    sheet = str(sheet)
    file = 'latencies_bis' + str('_'.join(sheet.split('_')[:3])) + '.pdf'
    dirname = os.path.join(paths['owncFig'], 'pythonPreview', 'extra')
    filename = os.path.join(dirname, file)
    fig.savefig(filename)


#%% significancy
data_df.loc[data_df.significancy > 0, ['layers', 'significancy']]

sig = data_df.loc[data_df.significancy > 0, ['layers', 'significancy']].groupby('layers').count()
allpop = data_df.groupby('layers')['significancy'].count()


sigDf = pd.DataFrame(pd.Series(allpop))


#%%

def plot_d1_d0_low(datadf, sheet, high=False):
    """
    plot the d1 d0 relation
    input:
        datadf : pandas dataframe
        sheet : the sheet number from the related xcel file
    """
    # high speed
    isi = {0: 27.8, 1: 34.7,
           '1319_CXLEFT_TUN25_s30_csv_test.csv' : 27.8}
    isi_shift = isi.get(sheet, 0)

    # layer depths limits
    d = 0
    depths = []
    for num in datadf.layers.value_counts().values[:-1]:
        d += num
        depths.append(d)
    depths.insert(0, 0)
    depths.append(datadf.index.max())

    # plot intervals:
    def add_conf_interval(ax, data, kind='med'):
        # add med ± 2mad lines to the ax
        med = data.median()
        mad = data.mad()
        ax.axhline(med, color='tab:blue', linewidth=3, alpha=0.7)
        ax.axhline(med + 2*mad, color='tab:blue',
                   linewidth=2, linestyle=':', alpha=0.7)
        ax.axhline(med - 2*mad, color='tab:blue',
                   linewidth=2, linestyle=':', alpha=0.7)
        txt = '{:.0f}±{:.0f}'.format(med, mad)
        ax.text(ax.get_xlim()[1], med, txt,
                va='bottom', ha='right', color='tab:blue')
        ax.text(ax.get_xlim()[1], med + 2*mad, 'med+2*mad',
                va='bottom', ha='right', color='tab:blue')

    def add_regress(ax, data):
        # add linear degression to the ax
        slope, intercept = np.polyfit(data[cols[0]], data[cols[1]], 1)
        xs = (data[cols[0]].min(), data[cols[0]].max())
        fxs = (intercept + slope * data[cols[0]].min(),
               intercept + slope * subdf[cols[0]].max())
        ax.plot(xs, fxs, color='tab:red', linewidth=2, alpha=0.8)

    # latencies
    select = ['layers', 'on_d0_0c_25', 'on_d1_s0_25']
    # high speed
    if high:
        select = ['layers', 'on_d0_0c_25', 'on_d0_sc_150']

    # remove outliers
    df = datadf[select].copy()
    if high:
        df['on_d0_sc_150'] = df['on_d0_sc_150'] + isi_shift
    for col in df.columns[1:]:
        df.loc[df[col] == 100, col] = np.nan
        df.loc[df[col] < 1, col] = np.nan

    # fig = plt.figure(figsize=(10, 12))
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10, 12))
    axes = axes.flatten()
    txt = 'manip {}   (med ± mad values)'.format(sheet)
    if high:
        txt = '{}   NB high speed is shifted by one ISI ({} msec)'.format(txt, isi_shift)
    fig.suptitle(txt)

    # d1 vs do
    ax = axes[0]
    ax.set_title('scatter')
    subdf = df[[select[1], select[2]]].dropna()
    cols = subdf.columns
    ax.scatter(subdf[cols[0]], subdf[cols[1]],
               marker='o', s=65, alpha=0.8, color='tab:blue')
    med, mad = subdf[cols[0]].median(), subdf[cols[0]].mad()
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
    lims = (floor(subdf.min().min()/5)*5, ceil(subdf.max().max()/5)*5)
    ax.set_ylim(lims)
    ax.set_xlim(lims)
    ax.plot(lims, lims)
    # regress
    add_regress(ax, subdf)
    txt = '_'.join(cols[0].split('_')[1:])
    ax.set_xlabel(txt)
    txt = '_'.join(cols[1].split('_')[1:])
    if high:
        txt = '{} + {} msec'.format(txt, isi_shift)
    ax.set_ylabel(txt)

    # diff vs depth
    ax = axes[1]
    ax.set_title('diff / depth')
    subdf = (df[select[2]] - df[select[1]]).dropna()
    subdf = subdf.reset_index()
    cols = subdf.columns
    ax.plot(subdf[cols[0]], subdf[cols[1]], 'o', alpha=0.8, ms=10, color='tab:blue')
    # ax.plot(subdf, 'o', alpha=0.8, ms=10, color='tab:blue')
    add_conf_interval(ax, subdf[cols[1]])
    # layers
    ax.axvspan(depths[1], depths[2], color='tab:grey', alpha=0.3)
    txt = 'layer IV'
    ax.text(x=(depths[1] + depths[2])/2, y = ax.get_ylim()[1], s=txt,
           va='top', ha='center', color='tab:grey')
    ax.set_xlabel('depth (electrode nb)')
    # regress
    add_regress(ax, subdf)
    # labels
    txt = '{}-{}'.format(select[2].split('_')[1:][0], select[1].split('_')[1:][0])
    if high:
        txt = '{} - {}'.format(
            '_'.join(select[2].split('_')[2:]),
            '_'.join(select[1].split('_')[2:]))
    ax.set_ylabel(txt)

    # diff/ref
    ax = axes[2]
    ax.set_title('diff / ref')
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
    add_conf_interval(ax, subdf[cols[1]])
    # labels
    txt = '{}'.format(cols[1])
    if high:
        txt = '{} - {}'.format(
            '_'.join(select[2].split('_')[2:]),
            '_'.join(select[1].split('_')[2:]))
    ax.set_ylabel(txt)
    ax.set_xlabel('{}'.format(cols[0]))
     # regress
    add_regress(ax, subdf)

    # diff/mean
    ax = axes[3]
    ax.set_title('diff / mean')
    subdf = pd.DataFrame(df[select[2]] - df[select[1]]).dropna()
    txt = '{}-{}'.format(select[2].split('_')[1:][0], select[1].split('_')[1:][0])
    subdf.columns= [txt]
    subdf['moy'] = df[[select[2], select[1]]].mean(axis=1)
    subdf = subdf[reversed(subdf.columns)]
    cols = subdf.columns
    ax.plot(subdf[cols[0]], subdf[cols[1]], 'o', alpha=0.8, ms=10, color='tab:blue')
    med = subdf[cols[0]].median()
    mad = subdf[cols[0]].mad()
    ax.axvspan(med-mad, med+mad, color='tab:blue', alpha=0.3)
    txt = '{:.0f}±{:.0f}'.format(med, mad)
    ax.text(med, ax.get_ylim()[1], txt,
            va='top', ha='center', color='tab:blue')

    add_conf_interval(ax, subdf[cols[1]])
    ax.set_xlabel('{} (d0 d1)'.format(cols[0]))
    txt = cols[1]
    if high:
        txt = '{} - {}'.format(
            '_'.join(select[2].split('_')[2:]),
            '_'.join(select[1].split('_')[2:]))
    ax.set_ylabel(txt)
    # regress
    add_regress(ax, subdf)

    for ax in axes[1:]:
        ax.axhline(0, color='tab:gray')
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


# plt.close('all')
high = True
for sheet in range(2):
    # sheet = 1
    print(sheet)
    data_df = load_latencies(sheet)
    data_df = data_df[data_df.significancy]
    data_df = clean_df(data_df, mult=4)
    fig = plot_d1_d0_low(data_df, sheet, high)

    save = False
    if save:
        sheet = str(sheet)
        if high:
            file = 'latOn_d1d0_high_' + str('_'.join(sheet.split('_')[:3])) + '.pdf'
        else:
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
    select = ['layers', 'on_d0_0c_25',
              'on_d1_s0_25', 'on_d0_sc_150']
    df = datadf[select].copy()
    for col in df.columns[1:]:
        df.loc[df[col] == 100, col] = np.nan
        df.loc[df[col] < 1, col] = np.nan
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
    data_df = data_df[data_df.significancy]
    data_df = clean_df(data_df, mult=4)
    fig = plot_d1_d2_high(data_df, sheet)

    save=False
    if save:
        sheet = str(sheet)
        file = 'latOn_d2d1_high_' + str('_'.join(sheet.split('_')[:3])) + '.pdf'
        dirname = os.path.join(paths['owncFig'], 'pythonPreview', 'extra')
        filename = os.path.join(dirname, file)
        fig.savefig(filename)
