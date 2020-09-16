#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
a module to build the config and general parameters

"""



import platform
import os
import getpass
import sys
#import inspect


def build_paths():
    """
    nb present also i, the load_data.py

    """
    paths = {}
    osname = platform.system()
    username = getpass.getuser()
    if osname == 'Windows'and username == 'Benoît':
        paths['pg'] = r'D:\\travail\sourcecode\developing\paper\centriG'
        sys.path.insert(0, r'D:\\travail\sourcecode\developing\paper')
        paths['owncFig'] = 'D:\\ownCloud\cgFiguresSrc'
    elif osname == 'Linux' and username == 'benoit':
        paths['pg'] = r'/media/benoit/data/travail/sourcecode/developing/paper/centriG'
    elif osname == 'Windows'and username == 'marc':
        paths['pg'] = r'H:/pg/centriG'
    elif osname == 'Darwin' and username == 'cdesbois':
        paths['pg'] = os.path.expanduser('~/pg/chrisPg/centriG')
        paths['owncFig'] = os.path.expanduser('~/ownCloud/cgFigures')
    return paths

def rc_params(font_size = 'medium'):  # large, medium
    """
    build an rc dico param for matplotlib
    """    
    params = {'font.sans-serif': ['Arial'],
          'font.size': 14,
          'legend.fontsize': font_size,
          'figure.figsize': (11.6, 5),
          'figure.dpi'    : 100,
          'axes.labelsize': font_size,
          'axes.titlesize': font_size,
          'xtick.labelsize': font_size,
          'ytick.labelsize': font_size,
          'axes.xmargin': 0}
    return params

def std_colors():
    colors = {'red' : [x/256 for x in [229, 51, 51]],
              'green' : [x/256 for x in [127, 204, 56]],
              'blue' :    [x/256 for x in [0, 125, 218]],
              'yellow' :    [x/256 for x in [238, 181, 0]],
              'violet' : [x/256 for x in [255, 0, 255]],
              'dark_red': [x/256 for x in [115, 0, 34]],
              'dark_green': [x/256 for x in [10, 146, 13]],
              'dark_blue': [x/256 for x in [14, 73, 118]],
              'dark_yellow': [x/256 for x in [163, 133, 16]],
              'blue_violet': [x/256 for x in [138, 43, 226]],
              'k' : [0, 0, 0]}
    return colors

def speed_colors():
    colors = {'yellow' : [x/256 for x in [253, 174, 74]],
              'orange' : [x/256 for x in [245, 124, 67]],
              'dark_orange' : [x/256 for x in [237, 73, 59]],
              'red' : [x/256 for x in [229, 51, 51]],
              'k' : [0, 0, 0]}
    return colors
