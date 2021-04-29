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

# column names
# nb df.head(3) -> header on three lines
df.iloc[df.index[0:4]] = df.iloc[df.index[0:4]].astype('str')
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
df.columns = cols
df = df.drop(index=range(4))

# drop empty
