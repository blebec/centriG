#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 13 18:08:45 2020

@author: cdesbois
"""

import os
import pandas as pd
from glob import glob

cdir = '/Users/cdesbois/pg/chrisPg/centriG/dataExtra'
os.chdir(cdir)

#%%

df = pd.DataFrame()
#ser = pd.read_csv('Top_synch_150.txt')

file_list = [y for x in os.walk(cdir) for y in glob(os.path.join(x[0], '*.txt'))]
#%%
df = pd.DataFrame()
for file in file_list:
    if os.path.basename(file)[0] not in ['r']:
        #print(os.path.basename(file)[0])
        ser = pd.read_csv(file, header=None)
        name = os.path.basename(file).split('.')[0]
        ser.columns = [name]
        if len(ser) < 300:
            df = df.join(ser, how='outer')
            print (os.path.basename(file))
            print(len(ser))
            