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


# TODO update in load_data
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
               'lat50': 'time50', 'cp': 'cp', 'cf': 'cf',
               'rnd': 'rd', 'cross' : 'cx'}
    for key in chg_dct:
        newcols = [item.replace(key, chg_dct[key]) for item in newcols]
    return newcols


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
    else:
        print('non defined')

    #rename cols    
    cols = df.columns.to_list()
    cols = new_columns_names(cols)
    if file.startswith('sig'):
        cols = [item.replace('pop', 'sig') for item in cols]
    elif file.lower().startswith('nsig'):
        cols = [item.replace('pop', 'nsig') for item in cols]
    cols = [item.replace('__', '_') for item in cols]
    df.columns = cols
    return df, filename



# f1 = '/Users/cdesbois/ownCloud/cgFigures/data/averageTraces/controlsFig/popVmSectNormAlign.xlsx'
# file = 'popVmSectNormAlign.xlsx'

# df1_columns_list = ['popVmCtr', 
#                     'popVmscpIsoStc', 
#                     'popVmscfIsoStc', 
#                     'popVmscrossStc', 
#                     'popVmfrndIsoStc', 
#                     'popVmsrndIsoStc']

# f2 = '/Users/cdesbois/ownCloud/cgFigures/data/averageTraces/controlsFig/sigVmSectRaw.xlsx'
# file = 'sigVmSectRaw.xlsx'

df2_columns_list = ['popVmCtr',
                    'popVmscpIsoStc',
                    'popVmscfIsoStc',
                    'popVmscrossStc',
                    'popVmfrndIsoStc',
                    'popVmsrndIsoStc']

# f3 = '/Users/cdesbois/ownCloud/cgFigures/data/averageTraces/controlsFig/nSigVmSectNormAlign.xlsx'
# file = 'nSigVmSectNormAlign.xlsx'

# df3_columns_list = ['popVmCtr',
#                     'popVmscpIsoStc',
#                     'popVmscfIsoStc',
#                     'popVmscrossStc',
#                     'popVmfrndIsoStc',
#                     'popVmsrndIsoStc']

# s = set()
# for l in [df1_columns_list, df2_columns_list, df3_columns_list]:
#     for item in l:
#         p = set(convert_to_snake(item).split('_'))
#         s.update(p)

# # cols = df.columns.to_list()
# cols = df1.columns.to_list()
# cols = [convert_to_snake(item) for item in l1]

# # NB from load data


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
