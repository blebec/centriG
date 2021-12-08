#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  6 10:58:31 2021

@author: cdesbois

extract the 'speed' population traces
return:
    popspeed_df : pandas.DataFrame
export as .hdf:
    (in) populations_traces.hdf (keys = "speed")
"""


import os

from importlib import reload
import numpy as np
import pandas as pd

import config
import general_functions as gfunc
import load.load_data as ldat
import load.load_traces as ltra

anot = True  # to draw the date and name on the bottom of the plot
std_colors = config.std_colors()
speed_colors = config.speed_colors()
paths = config.build_paths()
os.chdir(paths["pg"])

paths = config.build_paths()
paths["save"] = os.path.join(
    paths["owncFig"], "pythonPreview", "fillinIn", "indFill_popFill"
)


def build_pop_speed_data(do_save=False):
    """ load xcel and export hdf population speed data """

    # conds, key_dico = config.std_names()

    # NB old file no more required, all is in the new
    # filename = "data/data_to_use/fig4.xlsx"
    # df = pd.read_excel(filename, engine="openpyxl")
    # # df = pd.read_excel(filename, engine="openpyxl")
    # # centering
    # middle = (df.index.max() - df.index.min()) / 2
    # df.index = df.index - middle
    # df.index = df.index / 10
    # df.columns = gfunc.new_columns_names(df.columns)
    # cols = df.columns

    supfile = "fig_speed_sup.xlsx"
    sup_filename = os.path.join(paths["sup"], supfile)
    popspeeddf = pd.read_excel(sup_filename)
    popspeeddf.columns = gfunc.new_columns_names(popspeeddf.columns)
    cols = popspeeddf.columns
    cols = [_.replace("se_", "_se_") for _ in cols]
    cols = [_.replace("_se_d_w", "_sedw") for _ in cols]
    cols = [_.replace("_se_u_p", "_seup") for _ in cols]
    cols = [_.replace("_stc", "_stc_") for _ in cols]
    cols = ["_" + _ + "_" for _ in cols]
    cols = [_.strip("_") for _ in cols]
    popspeeddf.columns = cols

    middle = (popspeeddf.index.max() - popspeeddf.index.min()) / 2
    popspeeddf.index = popspeeddf.index - middle
    popspeeddf.index = popspeeddf.index / 10

    # for k, v in conds:
    #     cols = [_.replace(k, v) for _ in cols]
    # for k, v in key_dico.items():
    #     cols = [_.replace(k, v) for _ in cols]
    # cols = [_.strip("_") for _ in cols]
    # df.columns = cols

    data_savename = os.path.join(paths["figdata"], "populations_traces.hdf")
    print("=" * 20, "{}({})".format(os.path.basename(data_savename), "speed"))
    for item in sorted(cols):
        print(item)
    print()
    if do_save:
        popspeeddf.to_hdf(data_savename, "speed")

    return popspeeddf


popspeed_df = build_pop_speed_data(do_save=False)
