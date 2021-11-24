#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 24 09:29:03 2021

@author: cdesbois
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu 12 nov 2020 15:07:38 CET
@author: cdesbois
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

paths = config.build_paths()
paths["save"] = os.path.join(
    paths["owncFig"], "pythonPreview", "fillinIn", "indFill_popFill"
)

paths["sup"] = os.path.join(
    paths["owncFig"], "pythonPreview", "current", "fig_data_sup"
)

#%% figure 6

# load data
# nb treat = load | centering the index | divide index by 10 | limit -200 to 200

# change the title


def append_var_to_indidf(write=False):
    indi_df = ldat.load_filldata("indi")
    pop_df = ldat.load_filldata("pop")

    # manage supplementary data (variability)
    file = "fig6_supData.xlsx"
    filename = os.path.join(paths["sup"], file)
    supdf = pd.read_excel(filename)
    # df = sup_df.copy()
    # centering
    middle = (supdf.index.max() - supdf.index.min()) / 2
    supdf.index = (supdf.index - middle) / 10
    # limit the date time range
    supdf = supdf.loc[-200:200]

    # individual
    icols = [_ for _ in supdf.columns if _[:4].isdigit()]
    pcols = [_ for _ in supdf.columns if not _[:4].isdigit()]
    # get cell name
    key = {"_".join(_.split("_")[:2]) for _ in icols}.pop()

    # format indi_df
    indidf = indi_df.copy()
    indicols = indidf.columns
    indicols = [_.lower() for _ in indicols]
    indicols = [_.strip("s") for _ in indicols]
    indicols = [_.replace("cpiso", "cpiso_") for _ in indicols]
    indicols = [key + "_" + _ for _ in indicols]
    indidf.columns = indicols

    # join
    vardf = supdf[icols]
    vardf.columns = [_.replace("_ctr_stc_", "_ctr_") for _ in vardf.columns]
    indidf = indidf.join(vardf)
    if write:
        file = "fig6.hdf"
        filename = os.path.join(paths["sup"], file)
        indidf.tohdf(filename, key="indi")

    # pop_df
    popdf = pop_df.copy()
    popcols = popdf.columns
    popcols = [_.replace("popfill", "popfill_") for _ in popcols]
    popcols = [_.replace("_Vm", "_Vm_") for _ in popcols]
    popcols = [_.replace("_Spk", "_Spk_") for _ in popcols]
    popcols = [_.replace("_scpIso", "_s_cpiso_") for _ in popcols]
    popcols = [_.replace("_scfIso", "_s_cfiso_") for _ in popcols]
    popcols = [_.replace("_frnd", "_f_rnd_") for _ in popcols]
    popcols = [_.replace("_srnd", "_s_rnd_") for _ in popcols]
    popcols = [_.replace("_scpCross", "s_cpcx_") for _ in popcols]
    popcols = [_.replace("_Vms", "_Vm_s") for _ in popcols]
    popcols = [_.replace("_Spks", "_Spk_S") for _ in popcols]
    popcols = [_.replace("_S_", "_s_") for _ in popcols]
    popcols = [_.replace("_Ctr", "_ctr") for _ in popcols]
    popcols = [_.replace("rnd_Iso", "rnd_") for _ in popcols]
    popcols = [_.replace("_Stc", "_stc_") for _ in popcols]
    popcols = [_.replace("_So", "_so_") for _ in popcols]
    popcols = [_.replace("__", "_") for _ in popcols]
    popcols = [_.strip("_") for _ in popcols]
    popdf.columns = popcols

    # join
    vardf = supdf[pcols]
    varcols = vardf.columns
    varcols = [_.replace("pop_fillsig_", "popfill_Vm_s_") for _ in varcols]
    varcols = [_.replace("_s_ctr_stc_", "_ctr_") for _ in varcols]
    vardf.columns = varcols
    popdf = popdf.join(vardf)

    # datasavename = os.path.join(paths["sup"], "fig6s.hdf")
    # print("=" * 20, "{}({})".format(os.path.basename(datasavename), key))
    #     for item in df.columns:
    #         print(item)
    #     print()
    #     if do_save:
    #         df.to_hdf(datasavename, key)

    if write:
        file = "fig6.hdf"
        filename = os.path.join(paths["sup"], file)
        popdf.tohdf(filename, key="pop")
    return indidf


indi_df = append_var_to_indidf()

# sup_cols = sup_df.columns
# for i in range(5):
#     print("{} : {}".format(i, {_.split("_")[i] for _ in sup_cols}))
