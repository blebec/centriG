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

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
# from matplotlib import markers
# from pandas.plotting import table

import config
# import fig_proposal as figp
# import general_functions as gfunc
# import load.load_data as ldat
# import load.load_traces as ltra
import extra.load_extra_pop as ld


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

csvLoad = True
if csvLoad:
    paths['data'] = os.path.join(paths['owncFig'], 'data', 'data_extra')
    files = [file for file in os.listdir(paths['data']) if file[:4].isdigit()]
    # files = ['1319_CXLEFT_TUN25_s30_csv_test_noblank.csv',
    #          '2019_CXRIGHT_TUN21_s30_csv_test_noblank.csv']
    file = files[1]
    sheet=file
    file_name = os.path.join(paths['data'], file)
    data_df, params = ld.load_csv(file_name)
else:
    # xcel
    params = {'0' : '1319_CXLEFT',
              '1' : '2019_CXRIGHT'}
    sheet = '1'
    data_df = ld.load_latencies(sheet)


layers_loc = ld.extract_layers(data_df)
# only significant part
data_df = data_df[data_df.sig_center]
data_df = data_df[data_df['sig_surround'].astype(bool)]
# clean
data_df = ld.clean_df(data_df, mult=4)
stats_df = data_df.describe()
# stats_df_sig = data_df[data_df.significancy].describe()

#%% check names
def check_names(df):
    """
    decompose the column name structure

    """
    splited = [_.split('_') for _ in df.columns]
    for i in range(8):
        names = set([_[i] for _ in splited if len(_)>i])
        if names:
            print('{}   {}'.format(i, names))

# bug with cond = 'on_d1_sc_150'
# choose file
def check_diff():
    """
    check for time shift
    """
    for num in range(2):
        paths['data'] = os.path.join(paths['owncFig'], 'data', 'data_extra')
        files = [file for file in os.listdir(paths['data']) if file[:4].isdigit()]
        # files = ['1319_CXLEFT_TUN25_s30_csv_test_noblank.csv',
        #          '2019_CXRIGHT_TUN21_s30_csv_test_noblank.csv']
        file = files[num]
        sheet=file
        file_name = os.path.join(paths['data'], file)
        data_df, params = ld.load_csv(file_name)
        csvdf = data_df.copy()

        params = {'0' : '1319_CXLEFT',
                  '1' : '2019_CXRIGHT'}
        sheet = str(num)
        data_df = ld.load_latencies(sheet)
        xceldf = data_df.copy()

        cond = 'on_d1_sc_150'
        meancsv = csvdf.describe().loc['mean'].loc[cond]
        meanxcel = xceldf.describe().loc['mean'].loc[cond]
        stdcsv = csvdf.describe().loc['std'].loc[cond]
        stdxcel = xceldf.describe().loc['std'].loc[cond]
        print('-'*20)
        print('file {}'.format('_'.join(file.split('_')[:2])))
        print('xcel {:.1f}±{:.1f}, csv {:.1f}±{:.1f}'.format(meanxcel, stdxcel, meancsv, stdcsv))
        # xcel 43.9±20.0, csv 908.4±18.9
        delta = meancsv - meanxcel
        print('delta is {:.2f}'.format(delta))
        # delta = 864.5011574074074

# check_diff()

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
    colordict = {'0c':'tab:grey', 's0':'tab:green', 'sc':'tab:red',
                 '00': 'tab:blue'}
    ons = [_ for _ in datadf. columns if _.startswith('on')]
    hhs = [_ for _ in datadf.columns if _.startswith('hh')]
    ints = [_ for _ in datadf.columns if _.startswith('int')]
# TODO sort HH order before plotting
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
    if mes is None:
        fig, axes = plt.subplots(nrows=1, ncols=3)
    else:
        fig, axes = plt.subplots(nrows=1, ncols=3, sharey=True)
    axes = axes.flatten()

    for i, dats in enumerate(conds):
        # i = 0
        # dats = conds[i]
        ax = axes[i]
        txt = dats[0].split('_')[0]
        ax.set_title(txt)
        for col in dats:
            if len(datadf[col].dropna()) == 0:
                dats.remove(col)
        bp = ax.boxplot(datadf[dats].dropna(), meanline=True, showmeans=True,
                         patch_artist=True)
        # add color code
        colors = [colordict[_.split('_')[2]] for _ in dats]
        # colors_doubled = [a for tup in zip(colors, colors) for a in tup]
        # lines
        # for i, line in enumerate(bp['boxes']):
        #     line.set_color(colors[i])
        # for i, line in enumerate(bp['caps']):
        #     line.set_color(colors_doubled[i])
        # for i, line in enumerate(bp['whiskers']):
        #     line.set_color(colors_doubled[i])
        # box
        for i, patch in enumerate(bp['boxes']):
            patch.set(facecolor=colors[i], alpha=0.3)
        # nb of cells
        y = datadf[dats].count().values
        for j, n in enumerate(y, 1):
            ax.text(x=j, y=0, s=str(n),
                    ha='center', va='center', color='tab:grey')
        labels = ['_'.join(_.split('_')[1:]) for _ in dats]
        med = datadf[dats].dropna().median()[0]
        mad = datadf[dats].dropna().mad()[0]
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
        ax.set_ylim(-4, ax.get_ylim()[1])
        ax.grid()
        for spine in ['top', 'right']:
            ax.spines[spine].set_visible(False)
    if mes is None:
        axes[1].set_ylim(axes[0].get_ylim())

    if anot:
        txt = 'file= {} ({})'.format(params.get(sheet, sheet), sheet)
        if len(txt) > 5:
            txt = '_'.join(sheet.split('_')[:3])
        fig.text(0.5, 0.01, txt,
                 ha='center', va='bottom', alpha=0.4)
        date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        fig.text(0.99, 0.01, 'extra/extra_pop/plot_boxplots',
                 ha='right', va='bottom', alpha=0.4)
        txt = '{}'.format(date)
        fig.text(0.01, 0.01, txt, ha='left', va='bottom', alpha=0.4)
    fig.tight_layout()
    return fig

# import time
# start_time = time.time()

plt.close('all')
mes = None
for mes in [None, 'on', 'hh', 'inte']:
    fig = plot_boxplots(data_df, mes=mes)
# print("--- %s seconds ---" % (time.time() - start_time))

    save=False
    if save:
        txt = 'file= {} ({})'.format(params.get(sheet, sheet), sheet)
        if len(txt) > 5:
            txt = '_'.join(sheet.split('_')[:3])
            if mes:
                txt = '_'.join([txt, mes])
            file = 'boxPlot_' + txt + '.pdf'
            dirname = os.path.join(paths['owncFig'], 'pythonPreview', 'extra')
            filename = os.path.join(dirname, file)
            fig.savefig(filename)

#%% filter process



def filter_effect(filename=file_name, removemax=True, params=params, mes='on'):
    """
    stat boxplot description significance effect
    input:
        datadf = pandas dataframe
        removemax : boolean to remove values of 100 msec
        params:  dicionary (containing the speeds used)
        measure in [on, hh and inte], default=on
    """
    measures = ['on', 'hh' ,'inte']
    if mes not in measures:
        print ('mes shoud be in {}'.format(measures))
        return
    if mes == 'inte':
        mes = mes[:-1]
# TODO change filename
    #load
    all_df, params = ld.load_csv(filename)
    all_df['nsig_surround'] = all_df.sig_surround.apply(lambda x: np.invert(x))
    #remove blank
    for col in all_df.columns:
        if col.count('_') > 2:
            if col.split('_')[2] == '00':
                all_df.drop(col, axis=1, inplace=True)
    # remove values of 100 <-> no detection
    if removemax:
        for col in all_df.columns:
            if all_df[col].dtypes == float:
                  all_df.loc[all_df[col] > 98, [col]] = np.nan
    # dispatch dataframes
    c_df = all_df[all_df.sig_center]
    s_df = all_df[all_df.sig_surround]
    sc_df = c_df[c_df.sig_surround]
    cNs_df = c_df[c_df.nsig_surround]
    Ns_df = all_df[all_df.nsig_surround]
    #dfs = [c_df, s_df, sc_df, cNs_df]
    dfs = {'all' : all_df,
            'c_sig' : c_df,
           'sc _sig' : sc_df,
           'notS_sig': Ns_df,
           's_sig' : s_df,
           'cNotS_sig' : cNs_df,
           }

    colordict = {'0c':'tab:grey', 's0':'tab:green', 'sc':'tab:red',
                 '00': 'tab:blue'}

    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(14, 16),
                                 sharey=True)
    axes = axes.flatten()
    fig.suptitle(mes)

    for i, k in enumerate(dfs):
        # i = 0
        ax = axes[i]
        ax.set_title(k)
        df = dfs[k]
        cols = [_ for _ in all_df. columns if _.startswith(mes)]
        for j, col in enumerate(cols):
            # j = 0
            # dats = conds[i]
            if len(df[col].dropna()) == 0:
                cols.remove(col)
        bp = ax.boxplot(df[cols].dropna(), meanline=True,
                        showmeans=True, patch_artist=True)
        # add color code
        colors = [colordict[_.split('_')[2]] for _ in cols]
        # colors_doubled = [a for tup in zip(colors, colors) for a in tup]
        # lines
        # for i, line in enumerate(bp['boxes']):
        #     line.set_color(colors[i])
        # for i, line in enumerate(bp['caps']):
        #     line.set_color(colors_doubled[i])
        # for i, line in enumerate(bp['whiskers']):
        #     line.set_color(colors_doubled[i])
        # box
        for i, patch in enumerate(bp['boxes']):
               patch.set(facecolor=colors[i], alpha=0.3)
        # nb of cells
        y = df[cols].count().values
        for j, n in enumerate(y, 1):
            ax.text(x=j, y=0, s=str(n),
                    ha='center', va='center', color='tab:grey')
        labels = [_.split('_')[-1] for _ in cols]
        med = df[cols].dropna().median()[0]
        mad = df[cols].dropna().mad()[0]
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
        ax.set_ylim(-4, ax.get_ylim()[1])
        # ax.grid()
        for spine in ['top', 'right']:
            ax.spines[spine].set_visible(False)
    if mes is None:
        axes[1].set_ylim(axes[0].get_ylim())

    if anot:
        txt = 'file= {} ({})'.format(params.get(sheet, sheet), sheet)
        if len(txt) > 5:
            txt = '_'.join(sheet.split('_')[:3])
        fig.text(0.5, 0.01, txt,
                 ha='center', va='bottom', alpha=0.4)
        date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        fig.text(0.99, 0.01, 'extra/extra_pop/filter_effect',
                 ha='right', va='bottom', alpha=0.4)
        txt = '{}'.format(date)
        fig.text(0.01, 0.01, txt, ha='left', va='bottom', alpha=0.4)
    fig.tight_layout()
    return fig

# import time
# start_time = time.time()

paths['data'] = os.path.join(paths['owncFig'], 'data', 'data_extra')
files = [file for file in os.listdir(paths['data']) if file[:4].isdigit()]
# files = ['1319_CXLEFT_TUN25_s30_csv_test_noblank.csv',
#          '2019_CXRIGHT_TUN21_s30_csv_test_noblank.csv']
file = files[0]
sheet=file
file_name = os.path.join(paths['data'], file)
data_df, params = ld.load_csv(file_name)
plt.close('all')
for mes in ['on', 'hh', 'inte']:
    fig = filter_effect(file_name, removemax=True, params=params, mes=mes)
# print("--- %s seconds ---" % (time.time() - start_time))

    save=False
    if save:
        txt = 'file= {} ({})'.format(params.get(sheet, sheet), sheet)
        if len(txt) > 5:
            txt = '_'.join(sheet.split('_')[:3])
        txt = '_'.join([txt, mes])
        file = 'sig_boxPlot_' + txt + '.pdf'
        dirname = os.path.join(paths['owncFig'], 'pythonPreview', 'extra')
        filename = os.path.join(dirname, file)
        fig.savefig(filename)
        print(file)


#%%
# pltconfig = config.rc_params()
# pltconfig['axes.titlesize'] = 'small'
plt.rcParams.update({'axes.titlesize': 'small'})


def plot_all_histo(df):
    """
    histograms for all the reponses
    """
    figs = []
    for mes in ['on', 'hh', 'int']:
        cols = []
        for col in df.columns:
            if mes in col:
                cols.append(col)
        # cols = [st for st in df.columns is mes in st]
        fig, axes = plt.subplots(nrows=4, ncols=5, figsize=(21, 16))
        axes = axes.flatten()
        # cols = []
        # for col in df[cols]:
        #     if df[col].dtype == 'float64':
        #         cols.append(col)
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
        if anot:
            txt = '{} ({})'.format(params.get(sheet, sheet), sheet)
            if len(txt) > 4:
                txt = '_'.join(txt.split('_')[:3])
            fig.text(0.5, 0.01, txt,
                     ha='center', va='bottom', alpha=0.4)
            date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            fig.text(0.99, 0.01, 'extra/extra_pop/plot_all_histo',
                     ha='right', va='bottom', alpha=0.4)
            txt = '{} {}'.format(date, '_'.join(str(sheet).split('_')[:3]))
            fig.text(0.01, 0.01, txt, ha='left', va='bottom', alpha=0.4)
        fig.tight_layout()
        figs.append(fig)
    return figs

plt.close('all')
fig0, fig1, fig2 = plot_all_histo(data_df)
save = False
if save:
    txt = str(sheet)
    if len(txt) > 4:
        txt = '_'.join(txt.split('_')[:3])
    for fig, mes in zip([fig0, fig1, fig2], ['on', 'hh', 'int']):
        file = '_'.join(['allHisto', txt, mes]) + '.pdf'
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
        title = 'hh latencies'
    else:
        cols = ons
        title = 'on latencies'
    dats = [_ for _ in cols if 'd0' in _]
    dats += [_ for _ in cols if 'd1' in _]

    isi = {'0': 27.8, '1': 34.7,
           '1319_CXLEFT_TUN25_s30_csv_test.csv' : 27.8,
           '1319_CXLEFT_TUN25_s30_csv_test_noblank.csv' : 27.8,
           '2019_CXRIGHT_TUN21_s30_csv_test_noblank.csv': 34.7}
    isi_shift = isi.get(sheet, 0)
    if shift:
        df['on_d0_sc_150'] += isi_shift
        if not hh:
            txt = ' (& on_d0_sc_150 shifted by {} msec)'.format(isi_shift)
            title += txt

    # remove values of 100 <-> no detection
    if removemax:
        for col in df.columns:
            if df[col].dtypes == float:
                df.loc[df[col] > 98, col] = np.nan
    # plotting
    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(15,12),
                             sharex=True, sharey=True)
    if diff:
        title = '{} ({})'.format(title, 'cond - centerOnly')
    fig.suptitle(title)

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
        # ax.hist(ser, bins=20, color=colors[i], alpha=0.7, rwidth=0.9)
        ax.hist(ser, bins=range(1, 100, 2), color=colors[i], alpha=0.7, rwidth=0.9)
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
        txt = params.get(sheet, sheet)
        if len(txt) > 4:
            txt = '_'.join(txt.split('_')[:3])
        fig.text(0.5, 0.01, txt,
                 ha='center', va='bottom', alpha=0.4)
        date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        fig.text(0.99, 0.01, 'extra/extra_pop/plot_on_histo',
                 ha='right', va='bottom', alpha=0.4)
        txt = '{}'.format(date)
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
#             save_fig(fig, diff, shift, hh, sheet, paths)

fig = plot_on_histo(data_df, diff=True, shift=False, hh=False, removemax=True)

# save_fig(fig, diff, shift, hh)

#%% histo 3 by three

def plot_histo_3conds(datadf, removemax=True, sheet=sheet,
                  diff=False, shift=False, hh=False, mes='on'):

    df = datadf.copy()
    # extract columns of interest
    ons = [_ for _ in df. columns if _.startswith('on')]
    ons = ons[:-2]  # just recordings
    hhs = [_ for _ in df.columns if 'hh' in _]
    ints = [_ for _ in df.columns if 'int' in _]
    ints = ints[:-2]
    # choose data
    dico = {'on':ons, 'hh':hhs, 'inte':ints}
    dats = dico.get(mes, None)
    if dats is None:
        return
    title = mes

    isi = {'0': 27.8, '1': 34.7,
           '1319_CXLEFT_TUN25_s30_csv_test.csv' : 27.8,
           '1319_CXLEFT_TUN25_s30_csv_test_noblank.csv' : 27.8,
           '2019_CXRIGHT_TUN21_s30_csv_test_noblank.csv': 34.7}
    isi_shift = isi.get(sheet, 0)
    if shift:
        df['on_d0_sc_150'] += isi_shift
        if not hh:
            txt = ' (& on_d0_sc_150 shifted by {} msec)'.format(isi_shift)
            title += txt

    # remove values of 100 <-> no detection
    if removemax:
        for col in df.columns:
            if df[col].dtypes == float:
                df.loc[df[col] > 98, col] = np.nan
    # plotting
    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(15,12),
                             sharex=True, sharey=True)
    if diff:
        title = '{} ({})'.format(title, 'cond - centerOnly')
    fig.suptitle(title)

    axes = axes.flatten()
    colordict = {'0c':'tab:grey', 's0':'tab:green', 'sc':'tab:red',
              '00': 'tab:blue'}
    # med = df[dats[0]].median()
    # mad = df[dats[0]].mad()
    maxi = 0

    for i, col in enumerate(dats):
        ax = axes[i//3]
        # axT = ax.twinx()
        if diff:
            ser = (df[dats[3*(i//3)]] - df[col]).dropna()
            med = ser.median()
            mad = ser.mad()
            # ax.axvspan(med - mad, med + mad,
            #            color= colordict[col.split('_')[2]], alpha=0.3)
            ax.axvline(med, color=colordict[col.split('_')[2]], alpha=0.5,
                        linewidth=2)
            print(ser)
        else :
            ser = df[col].dropna()
            # ax.axvspan(med - mad, med+mad, color='tab:blue', alpha=0.3)
            # ax.axvline(med, color='tab:blue', alpha=0.5, linewidth=2)
        # txt = '_'.join(col.split('_')[1:])
        txt = col.split('_')[-1]
        # ax.hist(ser, bins=20, color=colors[i], alpha=0.7, rwidth=0.9)
        ax.hist(ser, bins=range(1, 100, 2), color=colordict[col.split('_')[2]],
                alpha=0.7, rwidth=0.9, density=True)
        x = ser.values.tolist()
        # xvals = [_ for _ in x if not np.isnan(_)]
        if len(ser) > 0 and ser.median() > 0:
            print(col)
            # kde = stats.gaussian_kde([_ for _ in x if not np.isnan(_)])
            kde = stats.gaussian_kde(ser)
            x_kde = np.linspace(0,100, 20)
            ax.plot(x_kde, kde(x_kde), color=colordict[col.split('_')[2]],
                    alpha=1, linewidth=2, linestyle='-')
            # ax.axvline(ser.median(), color=colordict[col.split('_')[2]],
            #            alpha=0.5, linewidth=2)

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

    # for i in [0,3]:
    #     axes[i].set_ylabel('nb of electrodes')
    for i in [3,4,5]:
        axes[i].set_xlabel('time')

    if anot:
        txt = params.get(sheet, sheet)
        if len(txt) > 4:
            txt = '_'.join(txt.split('_')[:3])
        fig.text(0.5, 0.01, txt,
                 ha='center', va='bottom', alpha=0.4)
        date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        fig.text(0.99, 0.01, 'extra/extra_pop/plot_histo_3conds',
                 ha='right', va='bottom', alpha=0.4)
        txt = '{}'.format(date)
        fig.text(0.01, 0.01, txt, ha='left', va='bottom', alpha=0.4)
    fig.tight_layout()
    return fig

plt.close('all')
mes = 'inte'
fig = plot_histo_3conds(data_df, mes=mes, diff=False)

save = False
if save:
    txt = str(sheet)
    if len(sheet) > 4:
        txt = '_'.join(sheet.split('_')[:3])
    file = 'threeCond_histo_' + mes + '_' + txt + '.pdf'
    dirname = os.path.join(paths['owncFig'], 'pythonPreview', 'extra')
    filename = os.path.join(dirname, file)
    fig.savefig(filename)

#%% scatter individuals
def plot_on_scatter(datadf, removemax=True, sheet=sheet,
                    diff=False, shift=False, hh=False, layersloc=layers_loc):

    colordict = {'0c':'tab:blue', 's0':'tab:green', 'sc':'tab:red',
              '00': 'tab:grey'}

    df = datadf.copy()
    ons = [_ for _ in df. columns if _.startswith('on')]
    ons = ons[:-2]
    hhs = [_ for _ in df.columns if _.startswith('hh')]
    ints = [_ for _ in df.columns if _.startswith('int')]
    # limit for csv file
    # ons = [_ for _ in ons if _.split('_')[-1] in ('25', '150')]
    # ons.remove('on_d0_0c_150')
    # hhs = [_ for _ in hhs if _.split('_')[-1] in ('25', '150')]
    # ints = [_ for _ in ints if _.split('_')[-1] in ('25', '150')]
    # kind of plot
    if hh:
        cols = hhs
        title = 'hhs'
    else:
        cols = ons
        title = 'ons'
    # dats = [_ for _ in cols if _.startswith('on_d0')]
    # dats += ([_ for _ in cols if _.startswith('on_d1')])

    isi = {'0': 27.8, '1': 34.7,
           '1319_CXLEFT_TUN25_s30_csv_test.csv' : 27.8,
           '2019_CXRIGHT_TUN21_s30_csv_test_noblank.csv' : 34.7}
# TODO check (old = 27.8 & 34.7 new ? and 21 for 150°/sec
# ? ISI + stroke duration
    isi_shift = isi.get(sheet, 0)
    if shift:
        df['on_d0_sc_150'] += isi_shift
        if not hh:
            txt = ' (& on_d0_sc_150 shifted by {}'.format(isi_shift)
            title += txt
    # remove values of 100 <-> no detection
    if removemax:
        for col in df.columns:
            if df[col].dtypes == float:
                df.loc[df[col] > 98, col] = np.nan
    if hh:
        ons = hhs
    # plotting


    # fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(15,12),
    #                          sharex=True, sharey=True)
    fig, axes = plt.subplots(nrows=ceil(len(ons)/3), ncols=3, figsize=(15,24),
                             sharex=True, sharey=True)

    if diff:
        title = '{} ({})'.format(title, 'cond - centerOnly')
    fig.suptitle(title)

    axes = axes.flatten()
    colors = ('tab:blue', 'tab:red', 'tab:orange',
              'tab:red', 'tab:orange', 'tab:green')
    # global stat
    ref_med = df[ons[0]].median()
    ref_mad = df[ons[0]].mad()

    bylayer = pd.DataFrame(layersloc, index=['dmin', 'dmax']).T
    bylayer['ref_meds'] = df.groupby('layers')[ons[0]].median()
    bylayer['ref_mads'] = df.groupby('layers')[ons[0]].mad()

    for i, col in enumerate(ons):
        ax = axes[i]
        #layers
        ax.axhspan(bylayer.loc['4'].dmin, bylayer.loc['4'].dmax,
                   color='tab:grey', alpha=0.2)
        if diff:
            # one ref
#            df['toplot'] = df[ons[0]] - df[col]
            # one ref / speed
            df['toplot'] = df[ons[3 * (i // 3)]] - df[col]
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
                if not np.isnan(med):
                    ax.vlines(med, ymin=ymin, ymax=ymax, color='tab:blue',
                              alpha=0.5, linewidth=2)
                    ax.vlines(med-mad, ymin=ymin, ymax=ymax, color='tab:blue',
                              alpha=0.5, linewidth=2, linestyle=':')
                    ax.vlines(med+mad, ymin=ymin, ymax=ymax, color='tab:blue',
                              alpha=0.5, linewidth=2, linestyle=':')
        bylayer['meds'] = temp.groupby('layers').median()
        bylayer['mads'] = temp.groupby('layers').mad()
        for j in range(len(bylayer)):
            ymin, ymax, med, mad = bylayer.iloc[j][
                ['dmin', 'dmax', 'meds', 'mads']]
            # test if values are present
            if not np.isnan(med):
                ax.vlines(med, ymin=ymin, ymax=ymax,
                          color=colordict[col.split('_')[2]],
                           alpha=0.5, linewidth=2)
                ax.add_patch(Rectangle((med - mad, ymin), width=2*mad,
                                       height=(ymax - ymin),
                                       color=colordict[col.split('_')[2]],
                                       alpha=0.3, linewidth=0.5))
        txt = col
        ax.scatter(temp.toplot.values, temp.index,
                   color=colordict[col.split('_')[2]], alpha=0.7)
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

    for i in range(0, len(axes), 3):
        axes[i].set_ylabel('depth')
    for i in range(len(axes)-3, len(axes)):
        axes[i].set_xlabel('time')

    if anot:
        txt = '{} ({})'.format(params.get(sheet, sheet), sheet)
        if len(sheet) > 4:
            txt = '_'.join(txt.split('_')[:3])
        if hh:
            txt += ' hh '
        if shift:
            txt += ' shift '
        if diff:
            txt += ' diff '
        fig.text(0.5, 0.01, txt,
                 ha='center', va='bottom', alpha=0.4)
        date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        fig.text(0.99, 0.01, 'extra/extra_pop/plot_on_scatter',
                 ha='right', va='bottom', alpha=0.4)
        txt = '{}'.format(date)
        fig.text(0.01, 0.01, txt, ha='left', va='bottom', alpha=0.4)
    fig.tight_layout()
    fig.subplots_adjust(top=0.967, bottom=0.058,
                        left=0.038, right=0.995,
                        hspace=0.5, wspace=0.099)
    return fig

def save_scatter(fig, diff, shift, hh, sheet=sheet, paths=paths):
    txt  = str(sheet)
    if len(sheet) > 4:
        txt = '_'.join(txt.split('_')[:3])
    txt = 'on_scatter_' + txt
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
what=[True, False]
for diff in what:
    for shift in what :
        for hh in what:
            fig = plot_on_scatter(data_df, diff=diff, shift=shift, hh=hh,
                                  removemax=True)
            # save_scatter(fig, diff, shift, hh)

# TODO implement the shift for all speeds
# fig = plot_on_scatter(data_df, diff=False, shift=False,
#                       hh=True, removemax=True)


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
# TODO update this shift
    isi = {'0': 27.8, '1': 34.7,
           '1319_CXLEFT_TUN25_s30_csv_test.csv' : 27.8,
           '1319_CXLEFT_TUN25_s30_csv_test_noblank.csv' : 27.8,
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
        # diff_med = (df[d0_cols[1]] - df[col]).median()
        # diff_mad = (df[d0_cols[1]] - df[col]).mad()
        diff_mean = (df[d0_cols[1]] - df[col]).mean()
        # diff_std = (df[d0_cols[1]] - df[col]).std()
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
    file = files[1]
    sheet=file
    file_name = os.path.join(paths['data'], file)
    data_df, params = ld.load_csv(file_name)
    data_df = data_df[data_df.sig_center]
    data_df = data_df[data_df.sig_surround.astype(bool)]
    data_df = ld.clean_df(data_df, mult=4)

plt.close('all')
fig = plot_latencies(data_df, lat_mini=0, lat_maxi=80, sheet=sheet, xcel=False)

save = False
if save:
    txt = str(sheet)
    if len(sheet) > 4:
        txt = '_'.join(sheet.split('_')[:3])
    file = 'latencies_' + txt + '.pdf'
    dirname = os.path.join(paths['owncFig'], 'pythonPreview', 'extra')
    filename = os.path.join(dirname, file)
    fig.savefig(filename)

#%%  to be adpated

def plot_latencies_bis(datadf, lat_mini=10, lat_maxi=80, sheet=sheet,
                       layersloc=layers_loc, xcel=False):
    """
    plot the latencies
    input :
        df : pandasDataFrame
        lat_mini : start time to use (values before are dropped)
        lat_maxi : end time to use (values after are removed)
    output :
        matplotlib figure
    """
# TODO update this shift
    isi = {'0': 27.8, '1': 34.7,
           '1319_CXLEFT_TUN25_s30_csv_test.csv' : 27.8,
           '1319_CXLEFT_TUN25_s30_csv_test_noblank.csv' : 27.8,
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

fig = plot_latencies_bis(data_df, lat_mini=0, lat_maxi=80,
                         sheet=sheet, xcel=False)

new = False
if new :
    file = files[1]
    sheet=file
    file_name = os.path.join(paths['data'], file)
    data_df, params = ld.load_csv(file_name)
    data_df = data_df[data_df.sig_center]
    data_df = data_df[data_df.sig_surround.astype(bool)]
    data_df = ld.clean_df(data_df, mult=4)

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

def plot_d1_d0_low(datadf, sheet, param, high=False):
    """
    plot the d1 d0 relation
    input:
        datadf : pandas dataframe
        sheet : the sheet number from the related xcel file
    """
    # high speed
    isi = {0: 27.8, 1: 34.7,
           '1319_CXLEFT_TUN25_s30_csv_test.csv' : 27.8,
           '2019_CXRIGHT_TUN21_s30_csv_test_noblank.csv' : 34.7}
    isi_shift = isi.get(sheet, 0)

    # layer depths limits
    # d = 0
    # depths = []
    # for num in datadf.layers.value_counts().values[:-1]:
    #     d += num
    #     depths.append(d)
    # depths.insert(0, 0)
    # depths.append(datadf.index.max())

    depths = param.get('layerLimit', None)

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
    txt = sheet
    if len(sheet) > 4:
        txt = '_'.join(sheet.split('_')[:3])
    txt = '{} (med ± mad values)'.format(txt)
    if high:
        txt = '{}   NB high speed shifted by one ISI ({} msec)'.format(txt, isi_shift)
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

plt.close('all')

high = True
for file in files:
    # file = files[1]
    sheet=file
    file_name = os.path.join(paths['data'], file)
    data_df, params = ld.load_csv(file_name)
    data_df = data_df[data_df.sig_center]
    data_df = data_df[data_df.sig_surround.astype(bool)]
    data_df = ld.clean_df(data_df, mult=4)
    fig = plot_d1_d0_low(data_df, sheet, param=params, high=high)

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
           '1319_CXLEFT_TUN25_s30_csv_test.csv' : 27.8,
           '2019_CXRIGHT_TUN21_s30_csv_test_noblank.csv' : 34.7}
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
    txt = sheet
    if len(sheet) > 4:
        txt = '_'.join(sheet.split('_')[:3])
    fig.suptitle(txt)
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
               label=label)

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
for file in files:
    # file = files[1]
    sheet=file
    file_name = os.path.join(paths['data'], file)
    data_df, params = ld.load_csv(file_name)
    data_df = data_df[data_df.sig_center]
    data_df = data_df[data_df.sig_surround.astype(bool)]
    data_df = ld.clean_df(data_df, mult=4)
    fig = plot_d1_d2_high(data_df, sheet)

    save=False
    if save:
        sheet = str(sheet)
        file = 'latOn_d2d1_high_' + str('_'.join(sheet.split('_')[:3])) + '.pdf'
        dirname = os.path.join(paths['owncFig'], 'pythonPreview', 'extra')
        filename = os.path.join(dirname, file)
        fig.savefig(filename)
