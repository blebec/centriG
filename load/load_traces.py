#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
load the traces
"""
import os

import pandas as pd

import centriG.config as config

paths = config.build_paths()
#os.chdir(paths['pg'])




#%%

def load_intra_mean_traces(paths, **kwargs):
#def plot_figure3(stdcolors, *args, **kwargs):
    """
    load intra mean traces
    old start of plot_figure3
    input :
        paths dico (from centriG.config.build_paths)
        kind : in ['pop', 'sig', 'nsig'] (default = new)
        rec : in ['vm', 'spk'] (default = vm)
        spread : in ['sect', 'full'] (default = sect)
        treat : in ['normAlign', 'raw', 'peak']
    """
    kind = kwargs.get('kind', 'sig')
    # substract = kwargs.get('substract', False),
    # anot = kwargs.get('anot', True),
    age = kwargs.get('age', 'new')
    rec = kwargs.get('rec', 'vm')
    spread = kwargs.get('spread' , 'sect')
    treat = kwargs.get('treat', 'normAlign')

    filename = ''
    df = ''
    if age == 'old':
        dirname = os.path.join(paths['owncFig'], 'data', 'old')
        filenames = dict(pop = os.path.join(dirname, 'fig3.xlsx'),
                         sig = os.path.join(dirname, 'fig3bis1.xlsx'),
                         nsig =  os.path.join(dirname, 'fig3bis2.xlsx'))
        # samplesize
        # cellnumbers = dict(pop = 37, sig = 10, nonsig = 27)
        # ncells = cellnumbers[kind]
        df = pd.read_excel(filenames[kind])
    elif age == 'new':
        dir_name = os.path.join(paths['owncFig'],
                                'data', 'averageTraces', 'controlsFig')
        file_list = os.listdir(dir_name)
        kind = kind.lower()
        rec = rec.lower()
        spread = spread.lower()
        treat = treat.lower()
        if kind in ['pop', 'sig', 'nsig']:
            file_list = [item for item in file_list if item.lower().startswith(kind)]
        else:
            print('kind should be in [pop, sig or nsig]')
            return
        file_list = [item for item in file_list if rec in item.lower()]
        file_list = [item for item in file_list if spread in item.lower()]
        file_list = [item for item in file_list if treat in item.lower()]

        file = file_list[0]
        filename = os.path.join(dir_name, file)
        df = pd.read_excel(filename)

# TODO rename the columns !!!

    else:
        print('non defined')

    return df, filename










#%%
if __name__ == '__main__':

    dico = dict(
        kind = 'sig',
        substract = False,
        anot = True,
        age = 'new',
        rec='vm',
        spread='sect'
        )
    dico['age'] = 'old'
    data_df = load_intra_mean_traces(paths, **dico)
    print(data_df.columns.tolist())
