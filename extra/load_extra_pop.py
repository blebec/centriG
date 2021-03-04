#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 26 16:33:53 2021

@author: cdesbois
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config

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
            header.append(line)
    header = [_.replace('\n', '') for _ in header]
    header[0] = header[0].split(';')
    header[0] = [_.replace('total number of stim', 'nbstim') for _ in header[0]]
    header[0] = [_.replace(' ', '') for _ in header[0]]
    header[1] = header[1].replace('speed;', 'speed:')
    temp = [tuple(_.split(':')) for _ in header[0]]
    param = {a:b for a,b in temp}
    param['nbstim'] = int(float(param['nbstim']))
    param['blank'] = param['blank'].lower() in ['true', 1]
    temp = header[1].split(':')
    temp[1] = [float(_) for _ in temp[1].split(';') if _]
    temp[1] = [int(_) for _ in temp[1] if _]
    param.update({temp[0]: temp[1]})
    del temp, header
    param['file'] = os.path.basename(filename)

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
    # speed
    speeds = param['speed']
    speeds = [str(_) for _ in speeds]
    speeds = [x for three in zip(speeds,speeds,speeds) for x in three]
    speeds.append('00')
    speeds.append('150')
    print('='*30)
    print('len(speeds)={}  len(ons)={}'.format(len(speeds), len(ons)))
    # stimulations
    stims = ['0c', 's0', 'sc'] * (len(ons)//3)
    stims.append('00')
    stims.append('sc')
    print('='*30)
    print('len(stims)={}  len(ons)={}'.format(len(stims), len(ons)))
    # reference time d0 or d1
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
    cols = [_.replace('sig[1]', 'sig_center') for _ in cols]
    cols = [_.replace('sig[2]', 'sig_surround') for _ in cols]
    # HH measures
    protocs = ['_'.join(('hh',a,b,c)) for a,b,c in zip(times, stims, speeds)]
    protocs = protocs[:len(hhs)]        # not all teh conditions are present
    for i, col in enumerate(protocs,1):
        for item in cols:
            if 'hh[' in item:
                index = cols.index(item)
                cols[index] = col
                # print(item)
                break
    df.columns = cols

    # remove the last line (error in .csv construction)
    df = df.dropna()
    # normalise sig (ie 1 or 0 instead of 10 or 0)
    for col in df.columns:
        if 'sig' in col:
            print(col)
            df[col] = df[col]/10
            df[col] = df[col].astype(bool)
        if col == 'on_d1_sc_150':
            if df[col].mean() > 100:
                df[col] = df[col] - 868
                print('removed 868 msec to {}'.format(col))
    # layers
    layers = pd.read_csv(os.path.join(os.path.dirname(filename),
                                      'layers.csv'), sep=';')
    layers = layers.set_index(layers.columns[0]).T
    # check for filename
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
    # layers limit
    depths = []
    d = 0
    for num in df.layers.value_counts().values[:-1]:
        d += num
        depths.append(d)
    depths.insert(0, 0)
    depths.append(df.index.max())
    param['layerLimit'] = depths
    return df, param


def load_latencies(xcelsheet='0'):
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
    sheet = ['EXP 1', 'EXP 2'][int(xcelsheet)]
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
    cols = [_.replace('significancy', 'sigcenter') for _ in cols]

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
    df.sigcenter = (df.sigcenter/10).astype(bool)
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
def clean_df(datadf, mult=3):
    """
    replace by nan values outside med ± mult*mad

    """
    df = datadf.copy()
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

#==============

if __name__ == '__main__':
    csvLoad = True
    if csvLoad:
        paths['data'] = os.path.join(paths['owncFig'], 'data', 'data_extra')
        files = [file for file in os.listdir(paths['data']) if file[:4].isdigit()]
        # files = ['1319_CXLEFT_TUN25_s30_csv_test_noblank.csv',
        #          '2019_CXRIGHT_TUN21_s30_csv_test_noblank.csv']
        file = files[0]
        sheet=file
        file_name = os.path.join(paths['data'], file)
        data_df, params = load_csv(file_name)
    else:
        params = {'0' : '1319_CXLEFT',
                  '1' : '2019_CXRIGHT'}
        sheet = '1'
        data_df = load_latencies(sheet)


    layers_loc = extract_layers(data_df)
    # only significant part
    data_df = data_df[data_df.sig_center]
    data_df = data_df[data_df['sig_surround'].astype(bool)]
    # clean
    data_df = clean_df(data_df, mult=4)
    stats_df = data_df.describe()
