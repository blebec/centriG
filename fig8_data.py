#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 25 11:21:38 2021

@author: cdesbois
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
paths["sup"] = os.path.join(
    paths["owncFig"], "pythonPreview", "current", "fig_data_sup"
)

os.chdir(paths["pg"])

#%%


def load_fig8_cpIsoGain_initial(printTraces=False):
    """ load the initial xcel file that contains the traces for fig8 """
    filename = os.path.join(paths["pg"], "data", "data_to_use", "fig2_2traces.xlsx")
    inidf = pd.read_excel(filename, engine="openpyxl")
    # centering
    middle = (inidf.index.max() - inidf.index.min()) / 2
    inidf.index = (inidf.index - middle) / 10
    inidf = inidf.loc[-200:150]
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

    print("fig8_cpIsoGain_initial")
    if printTraces:
        for col in inidf.columns:
            print(col)
        print()

    aset = set()
    for _ in cols:
        l = _.split("_")
        for w in l:
            aset.add(w)
    print("keys are : {}".format(aset))
    return inidf


def load_fig8_cpIsoGain_sup(printTraces=False):
    """ load the excel sup file that contains the variability"""
    supfilename = os.path.join(paths["sup"], "fig8_supdata.xlsx")
    supdf = pd.read_excel(supfilename, keep_default_na=True, na_values="")
    # centering
    middle = (supdf.index.max() - supdf.index.min()) / 2
    supdf.index = (supdf.index - middle) / 10
    supdf = supdf.loc[-200:150]
    # adapt names
    scols = supdf.columns
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
    scols = [_.replace("cpcross", "cpx") for _ in scols]

    scols = [_.replace("pop_spk_ctr_stc", "pop_spk_ctr_sedw") for _ in scols]

    supdf.columns = scols

    print("beware : an unamed 38 column exists")

    print("fig8_cpIsoGain_sup")
    if printTraces:
        for col in supdf.columns:
            print(col)
        print()

    aset = set()
    for _ in scols:
        l = _.split("_")
        for w in l:
            aset.add(w)
    print("keys are : {}".format(aset))

    return supdf


def load_fig8_cpIsoGain_pop3sig(key="sector", printTraces=False):
    """ get sig amp U time U fill data aka sig3"""

    if key != "sector":
        print("{} should be implemented".format(key))
        return
    filename = os.path.join(
        paths["owncFig"],
        "data/averageTraces/controlsFig/union_idx_fill_sig_sector.xlsx",
    )
    sig3df = pd.read_excel(filename, engine="openpyxl")
    # centering already done
    middle = (sig3df.index.max() - sig3df.index.min()) / 2
    sig3df.index = (sig3df.index - middle) / 10
    sig3df = sig3df.loc[-200:150]
    # adapt names
    cols = gfunc.new_columns_names(sig3df.columns)
    cols = [item.replace("sig_", "pop3sig_") for item in cols]
    cols = [item.replace("full_rd", "frnd") for item in cols]
    cols = [item.replace("sect_rd", "srnd") for item in cols]
    cols = [st.replace("cp_iso", "cpiso") for st in cols]
    cols = [st.replace("cf_iso", "cfiso") for st in cols]
    cols = [st.replace("sect_cx", "cpx") for st in cols]
    cols = [st.replace("_sect_", "_") for st in cols]
    cols = [st.replace("__", "_") for st in cols]
    cols = [st.replace("_.1", "") for st in cols]
    sig3df.columns = cols

    print("fig8_cpIsoGain_pop3sig")
    if printTraces:
        for col in sig3df.columns:
            print(col)
        print()

    aset = set()
    for _ in cols:
        l = _.split("_")
        for w in l:
            aset.add(w)
    print("keys are : {}".format(aset))

    return sig3df


def extract_fig8_cpIsoGain_dataframes(inidf, sig3df, supdf):
    """ takes teh initial data and sorted them for figure 8
    input:
        inidf, sig3df, supdf : pandas.Dataframe
    output:
        indidf, popdf, pop2sigdf, pop3sigdf
        """
    inicols = inidf.columns
    sigcols = sig3df.columns
    supcols = supdf.columns

    # individual dataframe
    pop1 = [_ for _ in inicols if _.startswith("indi")]
    pop2 = [_ for _ in supcols if _.startswith("indi")]
    # remove overlap
    for col in list(set(pop1) & set(pop2)):
        pop2.remove(col)
    # individual df (spk and vm)
    indidf = inidf[pop1].join(supdf[pop2])

    # pop dataframe
    pop1 = [_ for _ in inicols if _.startswith("pop_")]
    pop2 = [_ for _ in supcols if _.startswith("pop_")]
    # remove overlap
    for col in list(set(pop1) & set(pop2)):
        pop2.remove(col)
    popdf = inidf[pop1].join(supdf[pop2])

    # pop2sig dataframe
    pop1 = [_ for _ in inicols if _.startswith("pop2sig_")]
    pop2 = [_ for _ in supcols if _.startswith("pop2sig_")]
    # remove overlap
    for col in list(set(pop1) & set(pop2)):
        pop2.remove(col)
    pop2sigdf = inidf[pop1].join(supdf[pop2])

    # pop3sig
    pop1 = [_ for _ in sigcols if _.startswith("pop3sig_")]
    pop2 = [_ for _ in supcols if _.startswith("pop3sig_")]
    # remove overlap
    for col in list(set(pop1) & set(pop2)):
        pop2.remove(col)
    pop3sigdf = sig3df[pop1].join(supdf[pop2])

    return indidf, popdf, pop2sigdf, pop3sigdf


def save_f8_cpIsoGain_data(indidf, popdf, pop2sigdf, pop3sigdf, write=False):
    """ save as hdf files """
    # datasavename = os.path.join(paths["sup"], "fig6s.hdf")
    savefile = "fig8s.hdf"
    keys = ["indi", "pop", "pop2sig", "pop3sig"]
    dfs = [indidf, popdf, pop2sigdf, pop3sigdf]
    savedirname = paths["figdata"]
    savefilename = os.path.join(savedirname, savefile)
    for key, df in zip(keys, dfs):
        print("=" * 20, "{}({})".format(os.path.basename(savefilename), key))
        for column in sorted(df.columns):
            print(column)
        print()
        if write:
            df.to_hdf(savefilename, key)


ini_df = load_fig8_cpIsoGain_initial()
sig3_df = load_fig8_cpIsoGain_pop3sig()
sup_df = load_fig8_cpIsoGain_sup()

indi_df, pop_df, pop2sig_df, pop3sig_df = extract_fig8_cpIsoGain_dataframes(
    ini_df, sig3_df, sup_df
)

save_f8_cpIsoGain_data(indi_df, pop_df, pop2sig_df, pop3sig_df, write=False)
