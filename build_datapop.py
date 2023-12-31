#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 25 11:21:38 2021

@author: cdesbois

extract the population traces (except filling-in significative sub_population)
build
- an individual example : indi_df
- population selected data:
    pop_df : all cells traces
    pop2sig_df : index significant cells (time U reponse)
    pop3sig_df : sigificant cells (time U reponse U filling-in)
- export as .hdf:
    example_traces_vmSpk.hdf (key = "indi")
    populations_traces.hdf (keys = "pop", "pop2sig", "pop3sig")
"""


import os

import matplotlib.pyplot as plt
import pandas as pd

import config
import general_functions as gfunc

# import old.old_figs as ofig

# import itertools


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
# paths["sup"] = os.path.join(
#     paths["owncFig"], "pythonPreview", "current", "xls_data_sup"
# )

os.chdir(paths["pg"])

# general


def test_empty(dflist):
    """check column is empty (or nan) in a dataframe"""
    for df in dflist:
        for col in df.columns:
            if df[col].dropna().empty:
                print("{} is empty".format(col))


def center_scale_df(df, timerange=None, scale=None):
    """center, rescale and limit slice a dataframe
    input:
        df: pd.DataFrame
        scale: divider to rescale (default is 10)
        timerange: a list [start, stop] to slice it
    return:
        df = pandas.DataFrame
    """
    if timerange is None:
        timerange = [-200, 200]
    if scale is None:
        scale = 10
    middle = (df.index.max() - df.index.min()) / 2
    df.index = (df.index - middle) / scale
    # limit the date time range
    if timerange[0] in df.index and timerange[1] in df.index:
        df = df.loc[timerange[0] : timerange[-1]]
        print("-" * 12 + " rescaled (/{}) and sliced({})".format(scale, timerange))
    else:
        print("timerange {} not in the index, no slicing performed")
    return df


def remove_empty_columns(df, name=""):
    """remove dataframe empty columns
    input:
        df : pandas.DataFrame
        name: a given name for the dataframe variable
    return:
        df : pandas.DataFrame without the empty or nan columns
    """
    cols_toremove = []
    for col in df.columns:
        if df[col].dropna().empty:
            # print("no data for {} deleted column".format(col))
            cols_toremove.append(col)
    if cols_toremove:
        print("{:->20} empty columns deleted:".format(" " + name))
        for col in cols_toremove:
            print("{} ".format(col))
        df.drop(columns=cols_toremove, inplace=True)
    else:
        print("{:->20} no empty columns:".format(" " + name))
    return df


def print_keys(df, name=""):
    """print the keys that are used in the column names (delimiter = '_')"""
    aset = set()
    for _ in df.columns:
        l = _.split("_")
        for w in l:
            aset.add(w)
    # print("{:>20}".format(name))
    print("{} keys are : {}".format(name, sorted(aset)))
    print()


def load_fig8_cpIsoGain_initial(printTraces=False):
    """load the initial xcel file that contains the traces for fig8
    input:
        printTraces: boolean (False) print the traces names
    output:
        inidf: pandas.DataFrame contraining the population traces (& seup and sedw),
        keys of the dataframe:
            indi <-> individual example,
            pop <-> population data
            pop2sig <-> cells significant for time U significant for response
            popN2sig <-> cells notSignificant for time U notSignificant for response
    """
    filename = os.path.join(paths["pg"], "data", "data_to_use", "fig2_2traces.xlsx")
    inidf = pd.read_excel(filename, engine="openpyxl")

    print("{:=>40}".format(" load_fig8_cpIsoGain_initial"))
    dfname = "inidf"
    inidf = remove_empty_columns(inidf, "inidf")
    # centering
    middle = (inidf.index.max() - inidf.index.min()) / 2
    inidf.index = (inidf.index - middle) / 10
    inidf = inidf.loc[-200:200]
    # inidf = inidf.loc[-200:150]
    # adapt names
    cols = inidf.columns
    cols = [_.casefold() for _ in cols]
    cols = [_.replace("vm", "_vm_") for _ in cols]
    cols = [_.replace("spk", "_spk_") for _ in cols]
    cols = [_.replace("scpiso", "_s_cpiso_") for _ in cols]
    cols = [_.replace("ctr", "_ctr_") for _ in cols]
    cols = [_.replace("nsig", "_nsig_") for _ in cols]
    cols = [_.replace("sig", "_sig_") for _ in cols]
    cols = [_.replace("n_sig", "nsig") for _ in cols]
    cols = [_.replace("_stc", "_stc_") for _ in cols]
    cols = [_.replace("__", "_") for _ in cols]
    cols = [_.strip("_") for _ in cols]
    cols = [_.replace("_s_", "_") for _ in cols]
    cols = [_.replace("pop_", "popN2sig_") if "_nsig" in _ else _ for _ in cols]
    cols = [_.replace("_nsig", "") for _ in cols]
    cols = [_.replace("pop_", "pop2sig_") if "_sig" in _ else _ for _ in cols]
    cols = [_.replace("_sig", "") for _ in cols]
    inidf.columns = cols

    print(dfname)
    if printTraces:
        for col in inidf.columns:
            print(col)
        print()
    print_keys(inidf, dfname)
    return inidf


def load_fig8_cpIsoGain_variability(printTraces=False):
    """load the additional xcel file that contains the Variability for fig8
    NB additional traces are ['pop_spk_ctr', 'pop_spk_cpiso_stc']
    input:
        printTraces: boolean (False) print the traces names
    output:
        supdf: pandas.DataFrame contraining the population traces (& seup and sedw),
        keys of the dataframe:
            indi <-> individual example,
            pop <-> population data
            pop3sig <-> cells notSignificant for time U notSignificant for response
    """

    supfilename = os.path.join(paths["xlssup"], "fig8_supdata.xlsx")
    supdf = pd.read_excel(supfilename, engine="openpyxl")
    # supdf = pd.read_excel(supfilename, keep_default_na=True, na_values="")

    print("{:=>40}".format(" load_fig8_cpIsoGain_variability"))
    dfname = "supdf"
    print(dfname)
    supdf = remove_empty_columns(supdf, dfname)

    # centering
    middle = (supdf.index.max() - supdf.index.min()) / 2
    supdf.index = (supdf.index - middle) / 10
    # supdf = supdf.loc[-200:150]
    supdf = supdf.loc[-200:200]
    # adapt names
    scols = supdf.columns
    scols = [_.replace("pop_22_ctr_stc_seDW", "pop_22_ctr_stc_spk_seDW") for _ in scols]
    scols = [_.casefold() for _ in scols]
    scols = [_.replace("ind_cell_", "indi_") for _ in scols]
    scols = [_.replace("indi_", "indivm_") if "_vm_" in _ else _ for _ in scols]
    scols = [_.replace("indi_", "indispk_") if "_spk_" in _ else _ for _ in scols]

    scols = [_.replace("pop_37_", "pop_") for _ in scols]
    scols = [_.replace("pop_22_", "pop_") for _ in scols]
    scols = [_.replace("pop_15_", "pop2sig_") for _ in scols]
    scols = [_.replace("pop_6_", "pop2sig_") for _ in scols]
    scols = [_.replace("pop_20_", "pop3sig_") for _ in scols]
    scols = [_.replace("pop_10_", "pop3sig_") for _ in scols]

    for pop in ["pop", "pop2sig", "pop3sig"]:
        scols = [_.replace(pop + "_", pop + "vm_") if "_vm" in _ else _ for _ in scols]
        scols = [
            _.replace(pop + "_", pop + "spk_") if "_spk" in _ else _ for _ in scols
        ]

    scols = [_.replace("_vm", "") for _ in scols]
    scols = [_.replace("_spk", "") for _ in scols]
    scols = [_.replace("vm_", "_vm_") for _ in scols]
    scols = [_.replace("spk_", "_spk_") for _ in scols]

    scols = [_.replace("ctr_stc_", "ctr_") for _ in scols]
    scols = [_.replace("_ctr_stc", "_ctr") for _ in scols]

    scols = [_.replace("cpcross", "cpx") for _ in scols]

    supdf.columns = scols

    if printTraces:
        for col in supdf.columns:
            print(col)
        print()
    print_keys(supdf, dfname)
    return supdf


def load_fig8_cpIsoGain_pop3sig(key="sector", printTraces=False):
    """load the xcel file that contains the traces for sig3
    input:
        key : 'sector' (other not implemented)
        printTraces: boolean (False) print the traces names
    output:
        sig3df: pandas.DataFrame contraining the population traces
        (no variability present)
    """
    if key != "sector":
        print("{} should be implemented".format(key))
        return
    filename = os.path.join(
        paths["owncFig"],
        "data/averageTraces/controlsFig/union_idx_fill_sig_sector.xlsx",
    )
    sig3df = pd.read_excel(filename, engine="openpyxl")

    print("{:=>40}".format(" load_fig8_cpIsoGain_pop3sig"))
    dfname = "sig3df"
    print(dfname)
    sig3df = remove_empty_columns(sig3df, "sig3df")

    # centering already done
    middle = (sig3df.index.max() - sig3df.index.min()) / 2
    sig3df.index = (sig3df.index - middle) / 10
    sig3df = sig3df.loc[-200:200]
    # sig3df = sig3df.loc[-200:150]
    # adapt names
    cols = gfunc.new_columns_names(sig3df.columns)
    cols = [_.replace("sig_", "pop3sig_") for _ in cols]
    cols = [_.replace("full_rd", "frnd") for _ in cols]
    cols = [_.replace("frnd_iso", "frnd") for _ in cols]
    cols = [_.replace("sect_rd", "srnd") for _ in cols]
    cols = [_.replace("srnd_iso", "srnd") for _ in cols]
    cols = [_.replace("cp_iso", "cpiso") for _ in cols]
    cols = [_.replace("cf_iso", "cfiso") for _ in cols]
    cols = [_.replace("sect_cx", "cpx") for _ in cols]
    cols = [_.replace("_sect_", "_") for _ in cols]
    cols = [_.replace("__", "_") for _ in cols]
    cols = [_.replace("_.1", "") for _ in cols]
    sig3df.columns = cols

    if printTraces:
        for col in sig3df.columns:
            print(col)
        print()
    print_keys(sig3df, dfname)
    return sig3df


def load_other_sig2_data(path):
    """load extrafile builded to get all the sector traces for sig2 pop"""
    file = "sigVmSectNormAlign.xlsx"
    dir_name = os.path.join(path["owncFig"], "data", "averageTraces", "controlsFig")
    filename = os.path.join(dir_name, file)
    osig2df = pd.read_excel(filename)
    print("loading {:=>40}".format(file))
    dfname = "osig2df"
    print(dfname)
    osig2df = remove_empty_columns(osig2df, "osig2df")
    osig2df = center_scale_df(osig2df)
    cols = gfunc.new_columns_names(osig2df.columns)
    cols = [_.replace("pop_", "pop2sig_") for _ in cols]
    cols = [_.replace("_cp_iso", "_cpiso_") for _ in cols]
    cols = [_.replace("_cf_iso", "_cfiso_") for _ in cols]
    cols = [_.replace("_full_rd_iso_", "_frnd_") for _ in cols]
    cols = [_.replace("_sect_rd_iso_", "_srnd_") for _ in cols]
    cols = [_.replace("_sect_", "_") for _ in cols]
    cols = [_.replace("__", "_") for _ in cols]
    osig2df.columns = cols
    return osig2df


def extract_pop_dataframes(inidf, sig3df, supdf, osig2df):
    """takes the initial data and sorted them for figure 8
    input:
        inidf, sig3df, supdf : pandas.Dataframe
    output:
        indidf, popdf, pop2sigdf, pop3sigdf
    """
    # to build
    # inidf = ini_df
    # sig3df = sig3_df
    # supdf = sup_df
    # osig2df = osig2_df

    inicols = inidf.columns
    sigcols = sig3df.columns
    supcols = supdf.columns
    osig2cols = osig2df.columns

    # individual dataframe
    pop1 = [_ for _ in inicols if _.startswith("indi")]
    pop2 = [_ for _ in supcols if _.startswith("indi")]
    # remove overlap
    for col in list(set(pop1) & set(pop2)):
        pop1.remove(col)
    # individual df (spk and vm)
    indidf = inidf[pop1].join(supdf[pop2])

    # pop dataframe
    pop1 = [_ for _ in inicols if _.startswith("pop_")]
    pop2 = [_ for _ in supcols if _.startswith("pop_")]
    # remove overlap
    for col in list(set(pop1) & set(pop2)):
        pop1.remove(col)
    popdf = inidf[pop1].join(supdf[pop2])

    # pop2sig dataframe
    pop1 = [_ for _ in inicols if _.startswith("pop2sig_")]
    pop2 = [_ for _ in supcols if _.startswith("pop2sig_")]
    pop3 = list(osig2cols)
    # remove overlap
    for col in list(set(pop1) & set(pop2)):
        pop1.remove(col)
    pop2sigdf = inidf[pop1].join(supdf[pop2])
    for col in list(set(pop2sigdf.columns) & set(pop3)):
        pop3.remove(col)
    pop2sigdf = pop2sigdf.join(osig2df[pop3])

    # pop3sig
    pop1 = [_ for _ in sigcols if _.startswith("pop3sig_")]
    pop2 = [_ for _ in supcols if _.startswith("pop3sig_")]
    # remove overlap
    for col in list(set(pop1) & set(pop2)):
        pop1.remove(col)
    pop3sigdf = sig3df[pop1].join(supdf[pop2])

    return indidf, popdf, pop2sigdf, pop3sigdf


# =============================================================================
# def save_f8_cpIsoGain_data(indidf, popdf, pop2sigdf, pop3sigdf, write=False):
#     """ save as hdf files """
#     # datasavename = os.path.join(paths["sup"], "fig6s.hdf")
#     savefile = "fig8s.hdf"
#     keys = ["indi", "pop", "pop2sig", "pop3sig"]
#     dfs = [indidf, popdf, pop2sigdf, pop3sigdf]
#     savedirname = paths["hdf_data"]
#     savefilename = os.path.join(savedirname, savefile)
#     for key, df in zip(keys, dfs):
#         print("=" * 20, "{}({})".format(os.path.basename(savefilename), key))
#         for column in sorted(df.columns):
#             print(column)
#         print()
#         if write:
#             df.to_hdf(savefilename, key)
# =============================================================================


def save_example_ofpop(indidf, write=False):
    """save as hdf files"""
    # TODO = change 'indi' by the cellname
    savefile = "example_traces_vmSpk.hdf"
    key = "indi"
    savedirname = paths["hdf"]
    savefilename = os.path.join(savedirname, savefile)
    print("=" * 20, "{}({})".format(os.path.basename(savefilename), key))
    for column in sorted(indidf.columns):
        print(column)
    print()
    if write:
        indidf.to_hdf(savefilename, key)


def save_populations_traces(popdf, pop2sigdf, pop3sigdf, write=False):
    """save as hdf files"""
    dfs = [popdf, pop2sigdf, pop3sigdf]
    savefile = "populations_traces.hdf"
    keys = ["pop", "pop2sig", "pop3sig"]
    savedirname = paths["hdf"]
    savefilename = os.path.join(savedirname, savefile)
    for key, df in zip(keys, dfs):
        print("=" * 20, "{}({})".format(os.path.basename(savefilename), key))
        for column in sorted(df.columns):
            print(column)
        print()
        if write:
            df.to_hdf(savefilename, key)


#%%

ini_df = load_fig8_cpIsoGain_initial()
sig3_df = load_fig8_cpIsoGain_pop3sig()
sup_df = load_fig8_cpIsoGain_variability()
osig2_df = load_other_sig2_data(paths)

indi_df, pop_df, pop2sig_df, pop3sig_df = extract_pop_dataframes(
    ini_df, sig3_df, sup_df, osig2_df
)


# save_f8_cpIsoGain_data(indi_df, pop_df, pop2sig_df, pop3sig_df, write=False)

# population version
save_example_ofpop(indi_df, write=False)
save_populations_traces(pop_df, pop2sig_df, pop3sig_df, write=False)
