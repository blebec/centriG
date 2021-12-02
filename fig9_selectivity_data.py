#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 26 13:21:38 2021

@author: cdesbois
"""

import os

import matplotlib.pyplot as plt
import pandas as pd

import config
import general_functions as gfunc

# import old.old_figs as ofig

# import itertools

import load.load_data as ldat
import load.load_traces as ltra

# nb description with pandas:
pd.options.display.max_columns = 30

# ===========================
# global setup
# NB fig size : 8.5, 11.6 or 17.6 cm

anot = True  # to draw the date and name on the bottom of the plot
std_colors = config.std_colors()
speed_colors = config.speed_colors()
plt.rcParams.update(config.rc_params())
paths = config.build_paths()
paths["sup"] = os.path.join(
    paths["owncFig"], "pythonPreview", "current", "fig_data_sup"
)

os.chdir(paths["pg"])


# data used: popfill 12 cells
# /Users/cdesbois/ownCloud/cgFigures/data/averageTraces/controlsFig/
# sigVmSectNormAlign.xlsx  (predivteur lineaire cf 6)
# select = dict(age="new", rec="vm", kind="sig")
# data_df, file = ltra.load_intra_mean_traces(paths, **select)


# pop2sig : 15 cells
# neurons sig for filling in
# data/data_to_use/popfill.xlsx
# filling in sig
# pop_df = ldat.load_filldata("pop")

#%%
# popfill
def _data(display=True):
    """ load the indidf and popdf dataframe for fig 6 """
    loadfile = "fig6s.hdf"
    loaddirname = paths["figdata"]
    loadfilename = os.path.join(loaddirname, loadfile)
    indifilldf = pd.read_hdf(loadfilename, "indi")
    popfilldf = pd.read_hdf(loadfilename, "pop")
    if display:
        print("-" * 20)
        print("loaded data for figure 6 predictors")
        for key, df in zip(["indi", "pop"], [indifilldf, popfilldf]):
            print("=" * 20, "{}({})".format("loaded", key))
            for column in sorted(df.columns):
                print(column)
            print()
    return indifilldf, popfilldf


# pop2sig
def load_f8_cpIsoGain_data(display=False):
    """ load fig8_cpsisogain hdf files
    input:
        display : boolean to list the files
    return
        indidf, popdf, pop2sigdf, pop3sigdf : pandas_Dataframes
    """
    file = "fig8s.hdf"
    loaddirname = paths["figdata"]
    loadfilename = os.path.join(loaddirname, file)
    indidf = pd.read_hdf(loadfilename, key="indi")
    popdf = pd.read_hdf(loadfilename, key="pop")
    pop2sigdf = pd.read_hdf(loadfilename, key="pop2sig")
    pop3sigdf = pd.read_hdf(loadfilename, key="pop3sig")

    keys = ["indi", "pop", "pop2sig", "pop3sig"]
    dfs = [indidf, popdf, pop2sigdf, pop3sigdf]
    for key, df in zip(keys, dfs):
        print("=" * 20, "loaded {}({})".format(file, key))
        if display:
            for column in sorted(df.columns):
                print(column)
        print()
    return indidf, popdf, pop2sigdf, pop3sigdf


_, popfill_df = load_fig6_predictors_datafile()

_, _, pop2sig_df, _ = load_f8_cpIsoGain_data()

#%%


def load_fig9_supdata():
    filename = os.path.join(paths["sup"], "fig9_supData.xlsx")
    fig9supdatadf = pd.read_excel(filename, engine="openpyxl")
    # rename
    cols = fig9supdatadf.columns
    cols = [_.lower() for _ in cols]
    cols = [_.replace("fig9_", "") for _ in cols]
    res = []
    for st in cols:
        a, b, c, d = st.split("_")
        res.append("_".join(["popfill", c, a, b, d]))
    cols = res
    cols = [_.replace("_c", "_s_c") for _ in cols]
    cols = [_.replace("_f", "_f_") for _ in cols]
    cols = [_.replace("_cpcross", "_cpcx") for _ in cols]

    fig9supdatadf.columns = cols

    # centering
    middle = (fig9supdatadf.index.max() - fig9supdatadf.index.min()) / 2
    fig9supdatadf.index = (fig9supdatadf.index - middle) / 10
    fig9supdatadf = fig9supdatadf.loc[-200:200]

    return fig9supdatadf


def merge_popfil(df0, df1):
    """ compare, remove duplicated columns and merge two dataframe """
    for df in [df0, df1]:
        df.columns = [_.lower() for _ in df.columns]
    # check overlap
    diff = set(df0) & set(df1)
    if len(diff) > 0:
        print("the two dataframes have common columns")
        for col in diff:
            change = (df0[col] - df1[col]).mean()
            print("mean difference for {} is {}".format(col, change))
            if change == 0:
                print("suppressed {} in second".format(col))
                df1.drop(columns=[col], inplace=True)
    df = df0.join(df1)
    return df


blb_df = load_fig9_supdata()
popfill_df.columns = [_.lower() for _ in popfill_df.columns]
newpopfill_df = merge_popfil(blb_df, popfill_df)

# to do clean the data ... and rebuild popfill
for col in sorted(newpopfill_df.columns):
    print(col)
