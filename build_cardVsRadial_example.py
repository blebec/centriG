#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  6 14:20:23 2021

@author: cdesbois

load the two example used to compare the cardinal and radial protocol

return: (if not present in namespace)
    gaby_example_df : pandas.DataFrame of cardinal protocol
    cg_example_df : pandas.DataFrame of radial protocol
export as .hdf:
    example_cardVsRadial.hdf" (keys = 'card', 'rad')
"""

import os

# from bs4 import BeautifulSoup
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

import config

paths = config.build_paths()
anot = True  # to draw the date and name on the bottom of the plot
std_colors = config.std_colors()
speed_colors = config.speed_colors()
plt.rcParams.update(config.rc_params())
paths = config.build_paths()

dirname = os.path.join(paths["owncFig"], "data", "gabyToCg")

file_name = os.path.join(dirname, "4100gg3_vm_cp_cx_max.svg")
# filename = os.path.expanduser('~/4100gg3_vm_cp_cx_max.svg')

file_name = os.path.join(dirname, "test.svg")


def load_gaby_example(extract=False, do_save=False):
    """ load the data extracter from the gaby plots """
    directory = os.path.join(paths["owncFig"], "data", "gabyToCg")
    filename = os.path.join(directory, "gabydf.csv")
    if extract:
        os.chdir(os.path.join(directory, "numGaby"))
        files = [_ for _ in os.listdir() if os.path.isfile(_)]
        files = [_ for _ in files if _.endswith("csv")]
        x = np.arange(-40, 200, 0.1)
        datadf = pd.DataFrame(index=x)
        for file in files:
            name = file.split(".")[0]
            df = pd.read_csv(
                file, sep=";", decimal=",", dtype="float", names=["x", "y"]
            )
            df = df.sort_values(by="x")
            f = interp1d(df.x, df.y, kind="linear")
            datadf[name] = f(x)
            datadf[name] = (
                datadf[name].rolling(11, win_type="triang", center=True).mean()
            )
        if do_save:
            datadf.to_csv(filename)
            print("saved {}".format(filename))
    else:
        datadf = pd.read_csv(filename)
        datadf = pd.DataFrame(datadf)
        datadf = datadf.set_index(datadf.columns[0])
    return datadf


def load_cg_example():
    """ load cg (radial) data example to compare with gaby (cardinal) example """
    file = "cg_specificity.xlsx"
    filename = os.path.join(dirname, "sources", file)
    cgexampledf = pd.read_excel(filename)

    return cgexampledf


#%%
def export_examples_data(gabyexampledf, cgexampledf, do_save):
    """save the data used to build the figure to an hdf file"""
    df0 = gabyexampledf.copy()
    df1 = cgexampledf.copy()

    middle = (df1.index.max() - df1.index.min()) / 2
    df1.index = (df1.index - middle) / 10
    # cells = list(set([_.split("_")[0] for _ in df1.columns]))
    # cells = list({_.split("_")[0] for _ in df1.columns})
    # limit the date time range
    # df = df.loc[-200:200]
    df1 = df1.loc[-42.5:206]
    dfs = [df0, df1]

    # data_savename = os.path.join(paths["figdata"], "fig5.hdf")
    data_savename = os.path.join(paths["figdata"], "example_cardVsRadial.hdf")
    for key, df in zip(["card", "rad"], dfs):
        print("=" * 20, "{}({})".format(os.path.basename(data_savename), key))
        for item in df.columns:
            print(item)
        print()
        if do_save:
            df.to_hdf(data_savename, key)

    # pdframes = {}
    # for key in ['card', 'rad']:
    #     pdframes[key] = pd.read_hdf(data_savename, key=key)


gaby_example_df = load_gaby_example(extract=False, do_save=False)
cg_example_df = load_cg_example()

save = False
export_examples_data(gaby_example_df, cg_example_df, save)
