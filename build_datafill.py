#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 29 15:30:40 2021

@author: cdesbois
extract filling-in significative cells data
if not present in namespace:
    build:
        - indifill_df : individual example
        - popfill_df : population data
    the functions can be used to save the data in a .hdf format
"""


import os
from datetime import datetime

from importlib import reload
from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config
import fig_proposal as figp
import general_functions as gfunc
import load.load_data as ldat
import load.load_traces as ltra

anot = True  # to draw the date and name on the bottom of the plot
std_colors = config.std_colors()
speed_colors = config.speed_colors()
plt.rcParams.update(config.rc_params())
paths = config.build_paths()
os.chdir(paths["pg"])

pd.options.display.max_columns = 30

paths = config.build_paths()
paths["save"] = os.path.join(
    paths["owncFig"], "pythonPreview", "fillinIn", "indFill_popFill"
)

paths["sup"] = os.path.join(
    paths["owncFig"], "pythonPreview", "current", "fig_data_sup"
)


def compare_df(df1, df2):
    """ compare the columns of the two dataframes
    input:
        df1, df2: pandas.DataFrame
    output:
        print columns differences
        """
    set1 = set(df1.columns)
    set2 = set(df2.columns)
    if set1 == set2:
        print("-" * 30)
        print("the two dataframes contains the same columns")
    else:
        if len(set1 - set2) > 0:
            print("-" * 30)
            print("only in first")
            for col in set1 - set2:
                print(col)
        if len(set2 - set1) > 0:
            print("-" * 30)
            print("only in second")
            for col in set2 - set1:
                print(col)


def compare_keys(df1, df2):
    """ compare the keys in the columns
    input:
        df1, df2: pandas.DataFrame
    output:
        print keys differencies
        """
    set1 = set()
    set2 = set()
    for s, df in zip([set1, set2], [df1, df2]):
        for col in df.columns:
            for _ in col.split("_"):
                s.add(_)
    if set1 == set2:
        print("-" * 30)
        print("same keys in the two datasets")
    else:
        if len(set1 - set2) > 0:
            print("-" * 30)
            print("only in first")
            for key in set1 - set2:
                print(key)
        if len(set2 - set1) > 0:
            print("-" * 30)
            print("only in second")
            for key in set2 - set1:
                print(key)


def print_keys(alist):
    """ separate the list in keys (sep = '_') and print keys"""
    print("-" * 20)
    for col in sorted(alist):
        print(col)
    print()
    keys = set()
    for it in alist:
        for _ in it.split("_"):
            keys.add(_)
    print(sorted(keys))
    print()


def test_empty_column(df):
    """ test the presence of nan """
    emptyness = {}
    for col in df:
        ser = df.loc[df[col].isna(), [col]]
        if not ser.empty:
            emptyness[col] = (ser.index.min(), ser.index.max())
            # print(col, ser)
    if emptyness:
        print("_" * 20)
        print("there are nan values")
        for k, v in emptyness.items():
            print(
                "{} : nan in {}, ref is ({}, {})".format(
                    k, v, df.index.min(), df.index.max()
                )
            )
    else:
        print("no nan values")
    print()


def merge_popfil(ddf0, ddf1):
    """ compare, remove duplicated columns and merge two dataframe """
    df0 = ddf0.copy()
    df1 = ddf1.copy()
    for df in [df0, df1]:
        df.columns = [_.lower() for _ in df.columns]
    # check overlap
    diff = set(df0) & set(df1)
    if len(diff) > 0:
        print("the two dataframes have common columns")
        for col in diff:
            change = (df0[col] - df1[col]).mean()
            print("mean difference for {} is {}".format(col, change))
            if change < 0.1:
                print("suppressed {} in second".format(col))
                df1.drop(columns=[col], inplace=True)
    df = df0.join(df1)
    return df


def center_scale_df(df, timerange=None):
    """center the dataframe and rescale it """
    if timerange == None:
        timerange = [-200, 200]
    middle = (df.index.max() - df.index.min()) / 2
    df.index = (df.index - middle) / 10
    # limit the date time range
    df = df.loc[-200:200]
    return df


def build_indi_fill_data(write=False):
    """build an unified version of individual datafill"""
    # initial filling in example
    indifilldf = pd.read_excel("data/data_to_use/indifill.xlsx", engine="openpyxl")
    indifilldf = center_scale_df(indifilldf)
    indifilldf.columns = gfunc.new_columns_names(indifilldf.columns)

    # manage supplementary data (variability)
    supfile = "fig6_supData.xlsx"
    supfilename = os.path.join(paths["sup"], supfile)
    # ci columns are shorter than se columns
    supdf = pd.read_excel(supfilename, keep_default_na=True, na_values="")
    supdf.columns = [_.lower() for _ in supdf.columns]
    # expand the shorter waves (due to computation of confidence interval)
    # all the 'cirmin' except 'lpcirmin'
    # extract cirs, remove nan, rescale index and join
    cirs = [_ for _ in supdf.columns if "cir" in _]
    cirs = [_ for _ in cirs if not "lpcir" in _]
    cir_df = supdf[cirs].copy()
    supdf = supdf.drop(columns=cirs)
    cir_df = cir_df.dropna(axis=0, how="any")
    cir_df.index = cir_df.index * 10
    supdf = supdf.join(cir_df)
    del cir_df
    # fill using a pad (not optimal)
    supdf = supdf.fillna(method="pad", axis=0)
    supdf = center_scale_df(supdf)
    # individual_only
    icols = [_ for _ in supdf.columns if _[:4].isdigit()]
    supdf = supdf[icols]
    # get cell name
    key = {"_".join(_.split("_")[:2]) for _ in icols}.pop()

    # format indifill_df
    # indifilldf = indifill_df.copy()
    indicols = indifilldf.columns
    indicols = [_.strip("s") for _ in indicols]
    indicols = [_.replace("cp_iso_", "cpiso_") for _ in indicols]
    indicols = [_.replace("_slp", "_lp") for _ in indicols]
    indicols = [key + "_" + _ for _ in indicols]
    indicols = [_.lower() for _ in indicols]
    indifilldf.columns = indicols

    # append to supdf
    supcols = supdf.columns
    supcols = [_.replace("_lp", "_lp_") for _ in supcols]
    supcols = [_.replace("_cir", "_ci_") for _ in supcols]
    supcols = [_.replace("_ctr_stc_", "_ctr_") for _ in supcols]
    supcols = [_.replace("_ci_", "_ci") for _ in supcols]
    supcols = [_.replace("_so_lp_", "_lp_") for _ in supcols]

    supcols = [_.lower() for _ in supcols]
    supdf.columns = supcols
    indifilldf = indifilldf.join(supdf)

    key = "indi"
    savefile = "individual_fillin.hdf"
    savefilename = os.path.join(paths["figdata"], savefile)
    print("=" * 20, "{}({})".format(os.path.basename(savefilename), "indi"))
    for column in sorted(indifilldf.columns):
        print(column)
    print()
    if write:
        indifilldf.to_hdf(savefilename, key)

    return indifilldf


def build_pop_fill_data(write=False):
    """ combine the xcel files to build fig6 dataframes
        input:
            write : boolean to save the data
        output:
            indifilldf = pd.dataframe for individual example
            popfilldf = pd.dataframe for population
            """
    # initial filling in population
    inifilldf = pd.read_excel("data/data_to_use/popfill.xlsx", engine="openpyxl")
    # centering
    inifilldf = center_scale_df(inifilldf)
    inifilldf.columns = gfunc.new_columns_names(inifilldf.columns)

    # manage supplementary data (variability)
    supfile = "fig6_supData.xlsx"
    supfilename = os.path.join(paths["sup"], supfile)
    supdf = pd.read_excel(supfilename, keep_default_na=True, na_values="")
    # centering
    supdf = center_scale_df(supdf)

    # pop only
    pcols = [_ for _ in supdf.columns if not _[:4].isdigit()]

    # popfill_df
    popfilldf = inifilldf.copy()

    popcols = popfilldf.columns
    popcols = [_.replace("_sect_", "_s_") for _ in popcols]
    popcols = [_.replace("_full_", "_f_") for _ in popcols]
    popcols = [_.replace("_rd_", "_rnd_") for _ in popcols]
    popcols = [_.replace("_cf_iso", "_cfiso_") for _ in popcols]
    popcols = [_.replace("_cp_iso", "_cpiso_") for _ in popcols]
    popcols = [_.replace("_cp_cx", "_cpcx_") for _ in popcols]
    popcols = [_.replace("_rnd_iso", "_rnd_") for _ in popcols]
    popcols = [_.replace("_dlp", "_lp") for _ in popcols]
    popcols = [_.replace("__", "_") for _ in popcols]
    popcols = [_.strip("_") for _ in popcols]
    popfilldf.columns = popcols

    # join fig6supdata
    vardf = supdf[pcols]
    varcols = vardf.columns
    varcols = [_.replace("pop_fillsig_", "popfill_Vm_s_") for _ in varcols]
    varcols = [_.replace("_s_ctr_stc_", "_ctr_") for _ in varcols]
    varcols = [_.replace("_slp", "_lp") for _ in varcols]
    varcols = [_.lower() for _ in varcols]
    vardf.columns = varcols
    # check duplicate
    for col in varcols:
        if col.lower() in [_.lower() for _ in popcols]:
            print("duplicated trace for {}".format(col))
            vardf = vardf.drop(columns=[col])
    popfilldf = popfilldf.join(vardf)

    # load fig9supdata
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
    # >>>>>>>>>>>
    cols = [_.replace("_s_ctr", "_ctr") for _ in cols]

    # <<<<<<<<<<<<<<<
    fig9supdatadf.columns = cols
    # centering
    middle = (fig9supdatadf.index.max() - fig9supdatadf.index.min()) / 2
    fig9supdatadf.index = (fig9supdatadf.index - middle) / 10
    fig9supdatadf = fig9supdatadf.loc[-200:200]

    # merge (call)
    popfilldf = merge_popfil(popfilldf, fig9supdatadf)
    # remove _stc_ because not present in all files (basename vs +se one)
    cols = popfilldf.columns
    cols = [_.replace("_stc", "_") for _ in cols]
    cols = [_.replace("__", "_") for _ in cols]
    cols = [_.strip("_") for _ in cols]
    popfilldf.columns = cols

    # remove duplicated
    popfilldf = popfilldf.loc[:, ~popfilldf.columns.duplicated()]

    key = "fillsig"
    savefile = "populations_traces.hdf"
    savefilename = os.path.join(paths["figdata"], savefile)
    print("=" * 20, "{}({})".format(os.path.basename(savefilename), key))
    for column in sorted(popfilldf.columns):
        print(column)
    print()
    if write:
        popfilldf.to_hdf(savefilename, key)
    return popfilldf


if not "indifill_df" in dir():
    indifill_df = build_indi_fill_data(write=False)
if not "popfill_df" in dir():
    popfill_df = build_pop_fill_data(write=True)
