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

import config




def load2():
    """
    import the datafile
    return a pandasDataframe and a dictionary of contents
    """
    #____data
    filename = 'data/fig2traces.xlsx'
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

def new_columns_names(cols):
    def convert_to_snake(camel_str):
        """ camel case to snake case """
        temp_list = []
        for letter in camel_str:
            if letter.islower():
                temp_list.append(letter)
            elif letter.isdigit():
                temp_list.append(letter)
            else:
                temp_list.append('_')
                temp_list.append(letter)
        result = "".join(temp_list)
        return result.lower()
    newcols = [convert_to_snake(item) for item in cols]
    chg_dct = {'vms': 'vm_sect_', 'vmf': 'vm_full_',
               'spks': 'spk_sect_', 'spkf': 'spk_full_',
               'dlat50': 'time50', 'dgain50': 'gain50',
               'rnd': 'rd', 'cross' : 'crx'}
    for key in chg_dct:
        newcols = [item.replace(key, chg_dct[key]) for item in newcols]
    return newcols


#TODO function to developp to load energy from xcel file
def load_cell_contributions(kind='vm'):
    """
    load the corresponding xcel file
    kind = 'vm' or 'spk'
    """
    if kind == 'vm':
        filename = 'data/figSup34Vm.xlsx'
    elif kind == 'spk':
        filename = 'data/figSup34Spk.xlsx'
# TODO ajouter les fichiers energy excel
    else:
        print('kind should be vm or spk')
    df = pd.read_excel(filename)
    df.set_index('Neuron', inplace=True)
    #rename using snake_case
    cols = new_columns_names(df.columns)
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


def load_energy_gain_index(paths, sig=True):
    """
    energy in files
    pb des fichiers : les pvalues sone classées ... sans index ! dangereux !
    """
    def load_energy_cell(cell_name='1424M_CXG16.txt'):
        """
        to iterate and load sucessively all the cells
        """

        cols = ['centeronly', 'cpisosect', 'cfisosect', 'cpcrxsect', 'rdisosect',
                'cpisofull', 'cfisofull', 'cpcrxfull', 'rdisofull']

        cols = [item + '_energy' for item in cols]
        folder = os.path.join(paths['owncFig'], 'index', 'vm', 'energyt0baseline')
        filename = os.path.join(folder, cell_name)
        df = pd.read_csv(filename, sep='\t', names=cols)
        return df

    # neurons & values
    df = pd.DataFrame()
    folder = os.path.join(paths['owncFig'], 'index', 'vm', 'energyt0baseline')
    for name in os.listdir(folder):
        if os.path.isfile(os.path.join(folder, name)):
            energy_df = load_energy_cell(name)
            # nb here i choosed the median value
            df[os.path.splitext(name)[0]] = energy_df.median()

            df[os.path.splitext(name)[0]] = energy_df.mean()

    # pvalues one col by condition, one line per cell
    df = df.T
    folder = os.path.join(paths['owncFig'], 'index', 'vm', 'stats', 'pvaluesup')
    for name in os.listdir(folder):
        filename = os.path.join(folder, name)
        if os.path.isfile(filename):
            cond = os.path.basename(filename).split('.')[0]
            cond = cond.split('indisig')[0] + '_' + cond.split('indisig')[1]
            with open(filename, 'r') as fh:
                for line in fh:
                    if '[' in line:
                        line = line.replace('[', '')
                    if ']' in line:
                        line = line.replace(']', '')
                pvals = [np.float(item) for item in line.split(',')]
            #rename
            cond = cond.replace('rnd', 'rd')
            cond = cond.replace('sec', 'sect')
            cond = cond.replace('cross', 'crx')
            #assign
            df[cond + '_p'] = pvals
    if sig:
        # p to sig or non sig
        cols = [col for col in df.columns if '_p' in col]
        for col in cols:
            df[col] = df[col] - 0.05
            df.loc[df[col] > 0, [col]] = 0
            df.loc[df[col] < 0, [col]] = 1
            df[col] = df[col].astype(int)
        # rename
        cols = []
        for col in df.columns:
            if len(col.split('_')) > 2:
                col = col.split('_')[0] + '_' + col.split('_')[1] +  '_sig'
            cols.append(col)
        df.columns = cols
    return df


#%%
if __name__ == "__main__":
    paths = config.build_paths()
    fig2_df, fig2_cols = load2()
    energy_df = load_energy_gain_index(paths, sig=True)
    latGain50_v_df = load_cell_contributions('vm')
    latGain50_s_df = load_cell_contributions('spk')
