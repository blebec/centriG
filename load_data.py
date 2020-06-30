#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 29 10:10:52 2020

@author: cdesbois
"""
import platform
import os
import getpass
import pandas as pd
import numpy as np

def build_paths():
    paths = {}
    osname = platform.system()
    username = getpass.getuser()
    if osname == 'Windows'and username == 'Benoit':
        paths['pg'] = r'D:\\travail\sourcecode\developing\paper\centriG'
    elif osname == 'Linux' and username == 'benoit':
        paths['pg'] = r'/media/benoit/data/travail/sourcecode/developing/paper/centriG'
    elif osname == 'Windows'and username == 'marc':
        paths['pg'] = r'H:/pg/centriG'
    elif osname == 'Darwin' and username == 'cdesbois':
        paths['pg'] = os.path.expanduser('~/pg/chrisPg/centriG')
        paths['owncFig'] = os.path.expanduser('~/ownCloud/cgFigures')
    return paths

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
    newcols = [item.replace('vms', 'vm_s_') for item in newcols]
    newcols = [item.replace('vmf', 'vm_f_') for item in newcols]
    newcols = [item.replace('spks', 'spk_s_') for item in newcols]
    newcols = [item.replace('spkf', 'spk_f_') for item in newcols]
    return newcols

def load_cell_contributions(kind='vm'):
    """
    load the corresponding xcel file
    kind = 'vm' or 'spk'
    """
    if kind == 'vm':
        filename = 'data/figSup34Vm.xlsx'
    elif kind == 'spk':
        filename = 'data/figSup34Spk.xlsx'
    else:
        print('kind should be vm or spk')
    df = pd.read_excel(filename)
    df.set_index('Neuron', inplace=True)
    #rename using snake_case
    cols = new_columns_names(df.columns)
    df.columns = cols
    return df

# load the values50
def load_50vals(kind='vm'):
    if kind not in ['vm', 'spk']:
        print('kind should be in [vm, spk]')
        return
    df = load_cell_contributions(kind)
    trans = {'s': 'sect', 'f': 'full',
             'dlat50': 'time50', 'dgain50': 'gain50'}
    cols = []
    for item in df.columns:
        sp = item.split('_')
        new_name = sp[2] + sp[3] + trans[sp[1]] + '_' + trans[sp[5]]
        if len(sp) > 6:
            new_name += ('_sig')
        cols.append(new_name)
    df.columns = cols
    return df


#% load energy

# location : ownc/cgFigure/index/...
# energy : file = neuron, column = conditions, cells : repetition, 
# measure = mean on a defined window

# 9 conditions -> 8 stats

def load_energy_gain_index(paths, sig=True):
    """
    pb des fichiers : les pvalues sone classées ... sans index ! dangereux !
    """
    def load_energy_cell(cell_name = '1424M_CXG16.txt'):
        """
        to iterate and load sucessively all the cells        
        """

        cols = ['ctronly', 'cpisosec', 'cfisosec', 'cpcrosssec', 'rndisosec', 
                'cpisofull', 'cfisofull', 'cpcrossfull', 'rndisofull']
        folder = os.path.join(paths['owncFig'], 'index', 'energyt0baseline')
        filename = os.path.join(folder, cell_name)
        df = pd.read_csv(filename, sep='\t', names=cols)
        return df
    
    # neurons & values
    df = pd.DataFrame()
    folder = os.path.join(paths['owncFig'], 'index', 'energyt0baseline')
    for name in os.listdir(folder):
        if os.path.isfile(os.path.join(folder, name)):
            energy_df = load_energy_cell(name)
            # nb here i choosed the median value
            df[os.path.splitext(name)[0]] = energy_df.median()

            df[os.path.splitext(name)[0]] = energy_df.mean()
            
    # pvalues one col by condition, one line per cell
    df = df.T
    folder = os.path.join(paths['owncFig'], 'index', 'pvalue')
    for name in os.listdir(folder):
        filename = os.path.join(folder, name)
        if os.path.isfile(filename):
            cond = name.split('indisig')[0]
            with open(filename, 'r') as fh:
                for line in fh:
                    if '[' in line:
                        line = line.replace('[', '')
                    if ']' in line:
                        line = line.replace(']', '')
                pvals = [np.float(item) for item in line.split(',')]
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
            if len(col.split('_')) > 1:
                col = col.split('_')[0] + '_sig'
            cols.append(col)
        df.columns = cols
    return df


#%%
if __name__ == "__main__":
    paths = build_paths()
    energy_df = load_energy_gain_index(paths, sig=True)
    latGain50_v_df = load_50vals('vm')
    latGain50_s_df = load_50vals('spk')
