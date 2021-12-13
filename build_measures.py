#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  6 15:36:27 2021

@author: cdesbois

extract the 'indexes' population traces

return:
    indexes_df : pandas.DataFrame
export as .hdf:
    measures.hdf (keys = "indexes")
"""

import os

import matplotlib.pyplot as plt
import pandas as pd

import config

# import general_functions as gfunc
import load.load_data as ldat

# ===========================
# global setup
# NB fig size : 8.5, 11.6 or 17.6 cm

anot = True  # to draw the date and name on the bottom of the plot
std_colors = config.std_colors()
speed_colors = config.speed_colors()
plt.rcParams.update(config.rc_params())
paths = config.build_paths()
os.chdir(paths["pg"])

#%% save data for fig7a
def export_measures(do_save=False):
    """save the data for the figure 7"""
    df = ldat.load_cell_contributions(rec="vm", amp="engy", age="new")
    cols = df.columns
    data_savename = os.path.join(paths["hdf"], "measures.hdf")
    key = "indexes"
    print("=" * 20, "{}(key={})".format(os.path.basename(data_savename), key))
    for item in cols:
        print(item)
    print()
    if do_save:
        df.to_hdf(data_savename, key=key)


def load_measures(display=True):
    """load the xcel file that contains the indexes values"""
    key = "indexes"
    data_loadname = os.path.join(paths["hdf"], "measures.hdf")
    df = pd.read_hdf(data_loadname, key=key)
    conds = {_.split("_")[0] for _ in df.columns}
    dico = {}
    for cond in conds:
        dico[cond] = ["_".join(_.split("_")[1:]) for _ in df.columns if cond in _]
    print("=" * 20, "{}(key={})".format(os.path.basename(data_loadname), key))
    if display:
        for k, v in dico.items():
            print(k, v)
    print()
    return df


export_measures(do_save=False)
indexes_df = load_measures()
