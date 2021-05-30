#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 13 18:08:45 2020

@author: cdesbois
"""

import os
from glob import glob

import matplotlib.pyplot as plt
import pandas as pd

cdir = "/Users/cdesbois/pg/chrisPg/centriG/dataExtra"
os.chdir(cdir)

#%%

df = pd.DataFrame()
# ser = pd.read_csv('Top_synch_150.txt')

#%%
# NB look at the readme file in the dataExtra folder
# + infos_generation_figs listing conditions of interests
#  numerotation in /dataExtra


def load_extra_data(path, bin_dur=5):
    file_list = [y for x in os.walk(path) for y in glob(os.path.join(x[0], "*.txt"))]
    adf = pd.DataFrame()
    for file in file_list:
        if os.path.basename(file)[0] not in ["r"]:
            # print(os.path.basename(file)[0])
            df = pd.read_csv(file, header=None)
            name = os.path.basename(file).split(".")[0]
            df.columns = [name]
            if len(df) < 300:
                adf = adf.join(df, how="outer")
                # print (os.path.basename(file))
                # print(len(df))
    adf.index = adf.index * bin_dur
    return adf


df = load_extra_data(cdir)
ser150 = pd.read_csv("Top_synch_150.txt")
sert25 = pd.read_csv("Top_synch_25.txt")
