#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 28 13:53:38 2021

@author: cdesbois
"""

import os


import pandas as pd

import config


pd.set_option('display.max.columns', None)
#pd.set_option('display.max.rows', None)
pd.set_option('display.precision', 2)

paths = config.build_paths()
dirname = os.path.join(paths['owncFig'], 'data', 'baudot')
files = [st for st in os.listdir(dirname)
         if not os.path.isdir(os.path.join(dirname, st))]
for file in files:
    print(file)

file = 'dataGaby2005.xls'
filename = os.path.join(dirname, file)
#%%
def load_peggy():
    df = pd.read_excel(filename, header=None)
    # remove empty columns
    df = df.dropna(axis=0, how='all')
    df = df.dropna(axis=1, how='all')

    # forward row fill
    for i in range(2):
        df.iloc[i] = df.iloc[i].fillna(method='ffill')
        # nb fill should be done before str conversion because of np.nan
        df.iloc[0] = df.iloc[0].apply(lambda st: str(st).lower())
    dico = {'analysé': 'ana',
            'ant-post.': 'ant',
            'area centralis': 'ac',
            'autres': 'other',
            'barflash': 'barfl',
            'barmvt': 'barmv',
            'barre en mouvement': 'barmv',
            'barre flashee': 'barfl',
            'barres': 'bar',
            'caract. bar flashee': 'caractBarFlash',
            'caract. gaby': 'caract',
            'caract. r.c.': 'caract',
            'carateristiques de stimulation': 'caractStim',
            'cassette dat': 'dat',
            'cellule': 'cell',
            'chat': 'cat',
            'commentaire': 'comment',
            'comments': 'comment',
            'contr.': 'contr',
            'contraste': 'contr',
            'cycles': 'cycl',
            'debut': 'start',
            'debut visuel': 'visStart',
            'decharg.': 'dech',
            'dir.opt.': 'optDir',
            'direct.': 'dir',
            'directions': 'dir',
            'distancepatch': 'pathcDist',
            'dom. oc.': 'dom',
            'durée': 'dur',
            'dynamique': 'dyn',
            'electrod': 'elect',
            'electrode': 'elect',
            'electrop.': 'ephy',
            'electrophysiologie': 'ephy',
            'enrgt. dat': 'dat',
            'euil spike': 'thspike',
            'fichier': 'file',
            'file': 'file',
            'fin': 'end',
            'fin visuel': 'visEnd',
            'freqspat': 'spatFreq',
            'grating': 'grat',
            'i/v files': 'ivfile',
            'inact.': 'inact',
            'l.barre': 'lBar',
            'lateral.': 'lat',
            'localisations spatiales': 'spatialLoc',
            'masques': 'masks',
            'membrane': 'mb',
            'modulation': 'modul',
            'nbori': 'nbOri',
            'nbrep': 'nbRep',
            'nbseq': 'nbSeq',
            'nb sequence': 'nbSeq',
            'nom fichier': 'filename',
            'n°': 'n',
            'orient.': 'orient',
            'oritun': 'oriTun',
            'parametres de stimulation': 'stimParam',
            'potentiel de repos': 'Vm',
            'prof.': 'depth',
            'protocole': 'protoc',
            'pup.': 'pup',
            're': 'rem',
            'remarque': 'rem',
            'remarque generale': 'rem',
            'remarques': 'rem',
            'remarques2': 'rem2',
            'remq': 'rmq',
            'retenu': 'keep',
            'reverse correlation': 'revcor',
            'rin': 'R',
            'rmq electroph.': 'rmq',
            'répetitions': 'repet',
            'spike': 'spk',
            'type': 'kind',
            'valeur': 'value',
            'vitesse': 'speed',
            'vmco': 'vmCor.',
            'vmrest': 'vmRest',
            'w.barre': 'wBar'}

# todo = visuel -> visual, remove _°

    for i in range(4):
        df.iloc[i] = df.iloc[i].apply(lambda st: str(st).lower())
        df.iloc[i] = df.iloc[i].apply(lambda st: st.strip())
        df.iloc[i] = df.iloc[i].apply(lambda st: dico.get(st, st))
        df.iloc[i] = df.iloc[i].apply(lambda st: st.replace(' ', ''))


    # column names
    # nb df.head(3) -> header on three lines
    # df.iloc[df.index[0:7]] = df.iloc[df.index[0:7]].astype('str')
    df.iloc[df.index[0:5]] = df.iloc[df.index[0:5]].astype('str')
    cols = df.iloc[df.index[0:4]].apply(lambda st: st.str.cat(sep=', '))
    cols = [_.replace('nan,', '') for _ in cols]
    cols = [_.strip() for _ in cols]
    cols = [_.replace('nan', '') for _ in cols]
    cols = [_.replace(',', '') for _ in cols]
    cols = [_.replace('  ', ' ') for _ in cols]
    cols = [_.strip() for _ in cols]
    cols = [_.replace('NOM', '') for _ in cols]
    cols = ['nd' if len(_)<1 else _ for _ in cols]
    cols = [_.strip() for _ in cols]
    cols = [_.lower() for _ in cols]
    cols = ['_'.join(_.split(' ')) for _ in cols]
    cols = [_.replace('_nom', '') for _ in cols]
    cols = [_.replace('_(ms)', '(ms)') for _ in cols]
    cols = [_.replace('_°', '') for _ in cols]
    cols = [_.replace('visuel', 'visual') for _ in cols]
    cols = [_.replace('_mv', '(mv)') for _ in cols]
    cols = [_.replace('._mv', '(mv)') for _ in cols]
    cols = [_.replace('_(na)', '(na)') for _ in cols]
    cols = [_.replace('_(microns)', '(microns)') for _ in cols]
    cols = [_.replace('solut.', 'solut') for _ in cols]

    df.columns = cols
    df = df.drop(index=range(5))

    # drop gaby added rows
    df = df.drop(index=[193, 194, 201, 208])

    # fill names
    df.nom = df.nom.fillna(method='ffill')

    return df

data_df = load_peggy()
#%%
# NB positions = ['gaby_cr', 'visuel_ac_od', 'visuel_ac_og',
# 'revcor_cr', 'barfl_cr', 'barmv_cr', 'grat_cr(m1)'

select = ['nom', 'gaby_cr_x', 'gaby_cr_y', 'gaby_cr_l', 'gaby_cr_w', 'gaby_cr_theta'
        'revcor_cr_x', 'revcor_cr_y', 'revcor_cr_l', 'revcor_cr_w', 'revcor_cr_theta']

select = [st for st in data_df.columns if 'revcor' in st]
select = [st for st in select if '_cr' in st]
select = [st for st in select if st.endswith(('x', 'y', 'l', 'w', 'theta'))]
select.insert(0, 'nom')

df = data_df[select].copy()
cells = df.dropna(how='any')
print('{} cells'.format(len(cells.nom.unique())))
print(cells)
