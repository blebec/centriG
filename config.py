#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
a module to build the config and general parameters

"""


# import inspect
import cProfile
import getpass
import io
import os
import platform
import pstats
import sys


def build_paths():
    """the basic configuration for paths"""
    paths = {}
    osname = platform.system()
    username = getpass.getuser()
    if osname == "Windows" and username == "Benoît":
        paths["pg"] = r"D:\\travail\sourcecode\developing\paper\centriG"
        sys.path.insert(0, r"D:\\travail\sourcecode\developing\paper")
        paths["owncFig"] = r"D:\\ownCloud\cgFiguresSrc"
    elif osname == "Linux" and username == "benoit":
        paths["pg"] = r"/media/benoit/data/travail/sourcecode/developing/paper/centriG"
    elif osname == "Windows" and username == "marc":
        paths["pg"] = r"H:/pg/centriG"
    elif osname == "Linux" and username == "chris":
        paths["pg"] = r"/mnt/hWin/Chris/pg/chrisPg/centriG"
        paths["owncFig"] = r"/mnt/hWin/Chris/ownCloud/cgFigures"
    elif osname == "Darwin" and username == "cdesbois":
        paths["pg"] = os.path.expanduser("~/pg/chrisPg/centriG")
        paths["owncFig"] = os.path.expanduser("~/ownCloud/cgFigures")

    # paths["xls"] = os.path.join(
    #     paths["owncFig"], "pythonPreview", "current", "xls_sup"
    # )
    paths["xlssup"] = os.path.join(
        paths["owncFig"], "pythonPreview", "current", "xls_data_sup"
    )
    paths["figsup"] = os.path.join(
        paths["owncFig"], "pythonPreview", "current", "fig_sup"
    )
    paths["hdf"] = os.path.join(
        paths["owncFig"], "pythonPreview", "current", "hdf_data"
    )
    paths["figdata"] = os.path.join(
        paths["owncFig"], "pythonPreview", "current", "figdata"
    )

    return paths


def rc_params(font_size="medium"):  # large, medium
    """
    build an rc dico param for matplotlib
    """
    params = {
        "font.sans-serif": ["Arial"],
        "font.size": 14,
        "legend.fontsize": font_size,
        "figure.figsize": (11.6, 5),
        "figure.dpi": 100,
        "axes.labelsize": "large",
        # 'axes.labelsize': font_size,
        "axes.titlesize": "large",
        # 'axes.titlesize': font_size,
        "xtick.labelsize": font_size,
        "ytick.labelsize": font_size,
        "axes.xmargin": 0,
    }
    return params


def std_colors():
    """colors choosed for centrigabor figures"""
    colors = {
        "red": tuple([_ / 256 for _ in [229, 51, 51]]),
        "green": tuple([_ / 256 for _ in [127, 204, 56]]),
        "blue": tuple([_ / 256 for _ in [0, 125, 218]]),
        "yellow": tuple([_ / 256 for _ in [238, 181, 0]]),
        "violet": tuple([_ / 256 for _ in [255, 0, 255]]),
        "dark_red": tuple([_ / 256 for _ in [115, 0, 34]]),
        "dark_green": tuple([_ / 256 for _ in [10, 146, 13]]),
        "dark_blue": tuple([_ / 256 for _ in [14, 73, 118]]),
        "dark_yellow": tuple([_ / 256 for _ in [163, 133, 16]]),
        "blue_violet": tuple([_ / 256 for _ in [138, 43, 226]]),
        "k": (0, 0, 0),
        "brown": tuple([_ / 256 for _ in [127, 51, 51]]),
        "cyan": tuple([_ / 256 for _ in [23, 190, 207]]),
        "pink": tuple([_ / 256 for _ in [255, 0, 255]]),
    }
    return colors


def speed_colors():
    """just for speed coding"""
    colors = {
        "yellow": [_ / 256 for _ in [253, 174, 74]],
        "orange": [_ / 256 for _ in [245, 124, 67]],
        "dark_orange": [_ / 256 for _ in [237, 73, 59]],
        "red": [_ / 256 for _ in [229, 51, 51]],
        "k": [0, 0, 0],
    }
    return colors


def profile(fnc):
    """a decorator that uses cProfile to profile a function"""

    def inner(*args, **kwargs):
        pr = cProfile.Profile()
        pr.enable()
        retval = fnc(*args, **kwargs)
        pr.disable()
        s = io.StringIO()
        sortby = "cumulative"
        ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
        ps.print_stats()
        print(s.getvalue())
        return retval

    return inner


def std_legends():
    """define a standard name for legends"""
    dico = dict(
        cpiso="cp-iso",
        cfiso="cf-iso",
        cpcross="cp_cross",
        rndiso="rnd-iso",
        rd="rnd",
        cx="cross",
    )
    return dico


def std_names():
    """define standard decoding names to save data
    returns:
        conds = list of tuples to separate conditions
        keydico = dico to name the conditions
    """
    conds = [
        ("_pop", "_pop_"),
        ("_fill", "_fill_"),
        ("_Speed", "_speed_"),
        ("_Vm", "_Vm_"),
        ("_Spk", "_Spk_"),
        ("_s", "_s_"),
        ("_f", "_f_"),
        ("_cp", "_cp_"),
        ("_cf", "_cf_"),
        ("rnd", "_rnd_"),
        ("_Iso", "_iso_"),
        ("_Cross", "_cross_"),
        ("_So", "_So_"),
        ("_Stc", "_Stc_"),
        ("__", "_"),
        ("f_ill", "fill"),
    ]

    keydico = {
        "_s_": "_sect_",
        "_f_": "_full_",
        "_cp_": "_centripetal_",
        "_cf_": "_centrifugal_",
        "_rnd_": "_rnd_",
        "_Stc_": "_SthenCenter_",
        "_So_": "_SurroundOnly_",
        "_Ctr_": "_CenterOnly_",
        "_Slp_": "_SlinearPredictor_",
    }
    return conds, keydico


def std_titles():
    """define a standard name for titles
    returns:
        a dico of names
    """
    stdtitles = {
        "engy": r"$\Delta$ Response",
        "time50": r"$\Delta$ Latency",
        "gain50": "Amplitude Gain",
        "gain": "Amplitude Gain",
        "sect": "Sector",
        "sec": "Sector",
        "spk": "Spikes",
        "vm": "Vm",
        "full": "Full",
    }
    return stdtitles
