#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
load the traces
"""
import os

import pandas as pd

import config
import general_functions as gfunc

#import centriG.config as config
#import centriG.general_functions as gfunc


paths = config.build_paths()
#os.chdir(paths['pg'])


# TODO update in load_data


#%%

def list_files(paths, age='new'):
    """
    list the file sin the index folders    
    
    """
    file_list = ''
    if age == 'old':
        dirname = os.path.join(paths['owncFig'], 'data', 'old')
        file_list = os.listdir(dirname)
    elif age == 'new':
        dir_name = os.path.join(paths['owncFig'],
                                'data', 'averageTraces', 'controlsFig')
        file_list = os.listdir(dir_name)
    for file in file_list:
        print(file)
    return file_list

def load_intra_mean_traces(paths, **kwargs):
#def plot_figure3(stdcolors, *args, **kwargs):
    """
    load intra mean traces
    old start of plot_figure3
    input :
        paths dico (from centriG.config.build_paths)
        kwargs :
            kind : in ['pop', 'sig', 'nsig'] (default = new)
            rec : in ['vm', 'spk'] (default = vm)
            spread : in ['sect', 'full'] (default = sect)
            align : in ['normAlign', 'raw', 'peak', 'p2p']
    """
    kind = kwargs.get('kind', 'sig')
    # substract = kwargs.get('substract', False),
    # anot = kwargs.get('anot', True),
    age = kwargs.get('age', 'new')
    rec = kwargs.get('rec', 'vm')
    spread = kwargs.get('spread' , 'sect')
    align = kwargs.get('align', 'normAlign')

    file = ''
    file_list = ''
    df = ''
    if age == 'old':
        dirname = os.path.join(paths['owncFig'], 'data', 'old')
        filenames = dict(pop = os.path.join(dirname, 'fig3.xlsx'),
                         sig = os.path.join(dirname, 'fig3bis1.xlsx'),
                         nsig =  os.path.join(dirname, 'fig3bis2.xlsx'))
        # samplesize
        # cellnumbers = dict(pop = 37, sig = 10, nonsig = 27)
        # ncells = cellnumbers[kind]
        df = pd.read_excel(filenames[kind], engine='openpyxl')
    elif age == 'new':
        dir_name = os.path.join(paths['owncFig'],
                                'data', 'averageTraces', 'controlsFig')
        file_list = os.listdir(dir_name)
        kind = kind.lower()
        rec = rec.lower()
        spread = spread.lower()
        align = align.lower()
        if kind in ['pop', 'sig', 'nsig']:
            file_list = [item for item in file_list if item.lower().startswith(kind)]
        else:
            print('kind should be in [pop, sig or nsig]')
            return df, file
        file_list = [item for item in file_list if rec in item.lower()]
        file_list = [item for item in file_list if spread in item.lower()]
        file_list = [item for item in file_list if align in item.lower()]
        if len(file_list) < 1:
            print('no file corresponding to the criteria')
            print(kwargs)
            return df, file
        else:
            print('founded')
            for st in file_list:
                print('\t {}'.format(st))
            file = file_list[0]
            print('selected \n \t {}'.format(file))
        filename = os.path.join(dir_name, file)
        df = pd.read_excel(filename, engine='openpyxl')
    else:
        print('non defined')

    # rename cols
    # name to snake case
    cols = gfunc.new_columns_names(df.columns.to_list())
    # sig in filename to sig in column name
    if file.startswith('sig'):
        cols = [item.replace('pop', 'sig') for item in cols]
    elif file.lower().startswith('nsig'):
        cols = [item.replace('pop', 'nsig') for item in cols]
    cols = [item.replace('__', '_') for item in cols]
    df.columns = cols
    return df, file


#%%
if __name__ == '__main__':
    dico = dict(
        kind = 'pop',
        substract = False,
        anot = True,
        age = 'new',
        rec='vm',
        spread='sect'
        )
    dico['kind'] = 'sig'
    dico['align'] = 'p2p'
    data_df, file = load_intra_mean_traces(paths, **dico)
    if len(data_df) > 0:
        print(f'loaded file :\n \t {file:}')
        print('columns : ')
        for item in data_df.columns.tolist():
            print(f"\t {item}")
