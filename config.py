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
    """
    nb present also i, the load_data.py

    """
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
    """ colors choosed for centrigabor figures"""
    colors = {
        "red": tuple([x / 256 for x in [229, 51, 51]]),
        "green": tuple([x / 256 for x in [127, 204, 56]]),
        "blue": tuple([x / 256 for x in [0, 125, 218]]),
        "yellow": tuple([x / 256 for x in [238, 181, 0]]),
        "violet": tuple([x / 256 for x in [255, 0, 255]]),
        "dark_red": tuple([x / 256 for x in [115, 0, 34]]),
        "dark_green": tuple([x / 256 for x in [10, 146, 13]]),
        "dark_blue": tuple([x / 256 for x in [14, 73, 118]]),
        "dark_yellow": tuple([x / 256 for x in [163, 133, 16]]),
        "blue_violet": tuple([x / 256 for x in [138, 43, 226]]),
        "k": (0, 0, 0),
        "brown": tuple([x / 256 for x in [127, 51, 51]]),
        "cyan": tuple([x / 256 for x in [23, 190, 207]]),
        "pink": tuple([x / 256 for x in [255, 0, 255]]),
    }
    return colors


def speed_colors():
    """ just for speed coding """
    colors = {
        "yellow": [x / 256 for x in [253, 174, 74]],
        "orange": [x / 256 for x in [245, 124, 67]],
        "dark_orange": [x / 256 for x in [237, 73, 59]],
        "red": [x / 256 for x in [229, 51, 51]],
        "k": [0, 0, 0],
    }
    return colors


def profile(fnc):
    """ a decorator that uses cProfile to profile a function """

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
    """ define a standard name for legends """
    dico = dict(
        cpiso="cp-iso",
        cfiso="cf-iso",
        cpcross="cp_cross",
        rndiso="rnd-iso",
        rd="rnd",
        cx="cross",
    )
    return dico
