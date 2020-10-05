#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 29 10:10:52 2020

@author: cdesbois
"""
#import platform
import os
#import getpass
import pandas as pd
import numpy as np

#import centriG.config as config
#import centriG.general_functions as gfunc

import config 
import general_functions as gfunc

paths = config.build_paths()


def load2(age='new'):
    """
    import the datafile
    return a pandasDataframe and a dictionary of contents
    """
    #____data
    if age == 'old':
        filename = 'data/old/fig2traces.xlsx'
        print('beware : old file')
    else:
        filename = 'data/data_to_use/fig2_2traces.xlsx'
        print('fig2 : new file')
        # print('file fig2traces as to be updated')
        # return None, None
    df = pd.read_excel(filename)
    #centering
    middle = (df.index.max() - df.index.min())/2
    df.index = (df.index - middle)/10
    df = df.loc[-200:150]
    # nb dico : key + [values] or key + [values, (stdUp, stdDown)]
    colsdict = {
        'indVm': ['indiVmctr', 'indiVmscpIsoStc'],
        'indSpk': ['indiSpkCtr', 'indiSpkscpIsoStc'],
        'popVm': ['popVmCtr', 'popVmscpIsoStc'],
        'popSpk': ['popSpkCtr', 'popSpkscpIsoStc'],
        'popVmSig': ['popVmCtrSig', 'popVmscpIsoStcSig',
                     ('popVmCtrSeUpSig', 'popVmCtrSeDwSig'),
                     ('popVmscpIsoStcSeUpSig', 'popVmscpIsoStcSeDwSig')],
        'popSpkSig': ['popSpkCtrSig', 'popSpkscpIsoStcSig',
                      ('popSpkCtrSeUpSig', 'popSpkCtrSeDwSig'),
                      ('popSpkscpIsoStcSeUpSig', 'popSpkscpIsoStcSeDwSig')],
        'popVmNsig': ['popVmCtrNSig', 'popVmscpIsoStcNSig',
                      ('popVmCtrSeUpNSig', 'popVmCtrSeDwNSig'),
                      ('popVmscpIsoStcSeUpNSig', 'popVmscpIsoStcSeDwNSig')],
        'popSpkNsig': ['popSpkCtrNSig', 'popSpkscpIsoStcNSig',
                       ('popSpkCtrSeUpNSig', 'popSpkCtrSeDwNSig'),
                       ('popSpkscpIsoStcSeUpNSig', 'popSpkscpIsoStcSeDwNSig')],
        'sort': ['popVmscpIsolatg', 'popVmscpIsoAmpg',
                 'lagIndiSig', 'ampIndiSig']
                }
    return df, colsdict


def load_cell_contributions(rec='vm', amp='engy', age='new'):
    """
    load the corresponding xcel file
    age in ['old', 'new'] (old <-> time50, gain50, old way)
    amp in ['gain', 'engy']
    kind in ['vm' or 'spk']
    """
    if age == 'old':
        names_dico = dict(
            vm=os.path.join('data', 'old', 'figSup34Vm.xlsx'),
            spk=os.path.join('data', 'old', 'figSup34Spk.xlsx')
            )
        filename = names_dico.get(rec)
    elif age == 'new':
#        dirname = os.path.join(paths['pg'], 'data', 'data_to_use')            
        dirname = os.path.join(paths['owncFig'], 'data', 'index')
        if rec == 'vm' and amp == 'gain':
            filename = os.path.join(dirname, 'time50gain50Vm.xlsx')
        elif rec == 'spk' and amp == 'gain':
            filename = os.path.join(dirname, 'time50gain50Spk.xlsx')
        elif rec == 'vm' and amp == 'engy':
            filename = os.path.join(dirname, 'time50engyVm.xlsx')
        elif rec == 'spk' and amp == 'engy':
            filename = os.path.join(dirname, 'time50engySpk.xlsx')
        else:
            print('check the conditions')
            return
    else:
        print('files should be updated')
        return None
    df = pd.read_excel(filename)
    df.set_index('Neuron', inplace=True)
    #rename using snake_case
    cols = gfunc.new_columns_names(df.columns)
    df.columns = cols
    #groupNames
    cols = []
    for item in df.columns:
        sp = item.split('_')
        new_name = sp[2] + sp[3] + sp[1] + '_' + sp[5]
        if len(sp) > 6:
            new_name += ('_sig')
        cols.append(new_name)
    df.columns = cols
    return df


# =============================================================================
# def load_energy_gain_index(paths, sig=True):
#     """
#     energy in files
#     pb des fichiers : les pvalues sone classées ... sans index ! dangereux !
#     """
#     def load_energy_cell(cell_name='1424M_CXG16.txt'):
#         """
#         to iterate and load sucessively all the cells
#         """
#
#         cols = ['centeronly', 'cpisosect', 'cfisosect', 'cpcrxsect', 'rdisosect',
#                 'cpisofull', 'cfisofull', 'cpcrxfull', 'rdisofull']
#
#         cols = [item + '_energy' for item in cols]
#         folder = os.path.join(paths['owncFig'], 'data',
#                               'index', 'vm', 'energyt0baseline')
#         filename = os.path.join(folder, cell_name)
#         df = pd.read_csv(filename, sep='\t', names=cols)
#         return df
#
#     # neurons & values
#     df = pd.DataFrame()
#     folder = os.path.join(paths['owncFig'], 'data',
#                           'index', 'vm', 'energyt0baseline')
#     for name in os.listdir(folder):
#         if os.path.isfile(os.path.join(folder, name)):
#             energy_df = load_energy_cell(name)
#             # nb here i choosed the median value
#             df[os.path.splitext(name)[0]] = energy_df.median()
#
#             df[os.path.splitext(name)[0]] = energy_df.mean()
#
#     # pvalues one col by condition, one line per cell
#     df = df.T
#     folder = os.path.join(paths['owncFig'], 'data',
#                           'index', 'vm', 'stats', 'pvaluesup')
#     for name in os.listdir(folder):
#         filename = os.path.join(folder, name)
#         if os.path.isfile(filename):
#             cond = os.path.basename(filename).split('.')[0]
#             cond = cond.split('indisig')[0] + '_' + cond.split('indisig')[1]
#             with open(filename, 'r') as fh:
#                 for line in fh:
#                     if '[' in line:
#                         line = line.replace('[', '')
#                     if ']' in line:
#                         line = line.replace(']', '')
#                 pvals = [np.float(item) for item in line.split(',')]
#             #rename
#             cond = cond.replace('rnd', 'rd')
#             cond = cond.replace('sec', 'sect')
#             cond = cond.replace('cross', 'crx')
#             #assign
#             df[cond + '_p'] = pvals
#     if sig:
#         # p to sig or non sig
#         cols = [col for col in df.columns if '_p' in col]
#         for col in cols:
#             df[col] = df[col] - 0.05
#             df.loc[df[col] > 0, [col]] = 0
#             df.loc[df[col] < 0, [col]] = 1
#             df[col] = df[col].astype(int)
#         # rename
#         cols = []
#         for col in df.columns:
#             if len(col.split('_')) > 2:
#                 col = col.split('_')[0] + '_' + col.split('_')[1] +  '_sig'
#             cols.append(col)
#         df.columns = cols
#     return df
# =============================================================================

#%%
if __name__ == "__main__":
    paths = config.build_paths()
    fig2_df, fig2_cols = load2('new')
    #energy_df = load_energy_gain_index(paths, sig=True)
    latGain50_v_df = load_cell_contributions(rec='vm', age='old')
    latGain50_s_df = load_cell_contributions(rec='spk', age='old')

    latGain50_v_df = load_cell_contributions(rec='vm', amp='gain', age='new')
    latGain50_s_df = load_cell_contributions(rec='spk', amp='gain', age='new')

    latEner50_v_df = load_cell_contributions(rec='vm', amp='engy', age='new')
    latEner50_s_df = load_cell_contributions(rec='spk', amp='engy', age='new')
