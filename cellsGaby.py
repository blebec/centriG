#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 28 13:53:38 2021

@author: cdesbois
"""

import os
import sys


import pandas as pd

import config


pd.set_option('display.max.columns', None)
pd.set_option('display.max.rows', None)
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
df = pd.read_excel(filename, header=None)
# remove empty columns
df = df.dropna(axis=0, how='all')
df = df.dropna(axis=1, how='all')

#first line
df.iloc[0] = df.iloc[0].fillna(method='ffill')
df.iloc[0] = df.iloc[0].apply(lambda st: str(st).lower())

dico = {'barre flashee': 'barfl',
        'barflash': 'barfl',
         'barre en mouvement': 'barmv',
         'barmvt': 'barmv',
         'cassette dat' : 'dat',
         'chat' : 'cat',
         'electrode' : 'elect',
         'electrophysiologie': 'ephy',
         'i/v files': 'ivfile',
          'reverse correlation': 'revcor',
          'grating': 'grat'}
df.iloc[0] = df.iloc[0].apply(lambda st: dico.get(st, st))

df.iloc[1] = df.iloc[1].fillna(method='ffill')
df.iloc[1] = df.iloc[1].apply(lambda st: str(st).lower())

dico = {'analysé': 'ana',
        'ant-post.': 'ant',
        'area centralis': 'ac',
        'caract. bar flashee': 'caractBarFlash',
        'caract. gaby': 'caract',
        'caract. r.c.': 'caract',
        'carateristiques de stimulation': 'caractStim',
        'comments': 'comment',
        'debut visuel': 'visStart',
        'decharg.': 'dech',
        'direct.': 'dir',
        'dom. oc.': 'dom',
        'durée': 'dur',
        'dynamique': 'dyn',
        'electrod': 'elect',
        'electrop.': 'ephy',
        'enrgt. dat': 'dat',
        'euil spike': 'thspike',
        'file': 'file',
        'fin visuel': 'visEnd',
        'lateral.': 'lat',
        'nom fichier': 'filename',
        'n°': 'n',
        'orient.': 'orient',
        'parametres de stimulation': 'stimParam',
        'potentiel de repos': 'Vm',
        'prof.': 'depth',
        'pup.': 'pup',
        'remarque generale': 'rem',
        'remq': 'rmq',
        'retenu': 'keep',
        'rin': 'R',
        'rmq electroph.': 'rmq',
        'spike': 'spk'}
df.iloc[1] = df.iloc[1].apply(lambda st: dico.get(st, st))
df.iloc[1] = df.iloc[1].apply(lambda st: st.strip())
df.iloc[1] = df.iloc[1].apply(lambda st: st.replace(' ', ''))

df.iloc[2] = df.iloc[2].apply(lambda st: str(st).lower())
df.iloc[2] = df.iloc[2].apply(lambda st: st.strip())
df.iloc[2] = df.iloc[2].apply(lambda st: st.replace(' ', ''))
dico = { 'barres':'bar',
         'cellule':'cell',
         'commentaire': 'comment',
         'debut':'start',
         'fichier':'file',
         'fin':'end',
         'inact.':'inact',
         'l.barre':'lBar',
         'localisationsspatiales':'spatialLoc',
         'masques': 'masks',
         'membrane': 'mb',
         'modulation': 'modul',
         'nbori': 'nbOri',
         'nbrep': 'nbRep',
         'nbseq': 'nbSeq',
         'nbsequence': 'nbSeq',
         'oritun': 'oriTun',
         'protocole': 'protoc',
         're':'rem',
         'remarque': 'rem', 
         'remarques': 'rem',
         'remarques2': 'rem2',
         'type':'kind',
         'valeur': 'value',
         'vitesse': 'soeed',
         'vmco':'vmCor.',
         'vmrest':'vmRest',
         'w.barre':'wBar'}
df.iloc[2] = df.iloc[2].apply(lambda st: dico.get(st, st))

df.iloc[3] = df.iloc[3].apply(lambda st: str(st).lower())
df.iloc[3] = df.iloc[3].apply(lambda st: st.strip())
df.iloc[3] = df.iloc[3].apply(lambda st: st.replace(' ', ''))
dico = {
        'contr.': 'contr',
        'contraste': 'contr',
        'dir.opt.': 'optDir',
        'directions': 'dir',
        'distancepatch': 'pathcDist',
        'freqspat': 'spatFreq',
        'vitesse': 'speed'
        }
df.iloc[3] = df.iloc[3].apply(lambda st: dico.get(st, st))

df.iloc[4] = df.iloc[4].apply(lambda st: str(st).lower())
df.iloc[4] = df.iloc[4].apply(lambda st: st.strip())
df.iloc[4] = df.iloc[4].apply(lambda st: st.replace(' ', ''))
dico = {
        'autres': 'other',
        'cycles': 'cycl',
        'debut': 'start',
        'fin': 'end',
        'répetitions': 'repet',
        }
df.iloc[4] = df.iloc[4].apply(lambda st: dico.get(st, st))

# df.iloc[5] = df.iloc[5].apply(lambda st: str(st).lower())
# df.iloc[5] = df.iloc[5].apply(lambda st: st.strip())
# df.iloc[5] = df.iloc[5].apply(lambda st: st.replace(' ', ''))


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
cols = [_.replace('__nom', '') for _ in cols]
cols = [_.replace('_(ms)', '(ms)') for _ in cols]


df.columns = cols
df = df.drop(index=range(5))

# drop gaby added rows
df = df.drop(index=[193, 194, 201, 208])

# fill names
df.nom = df.nom.fillna(method='ffill')

