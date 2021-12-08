#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  8 13:36:31 2021

@author: cdesbois
"""

import os

os.chdir("/Users/cdesbois/pg/commun/pythonUNIC")

import elphy_reader as er

dirname = "/Volumes/bu_equipe/_EEG_/continuous/2017/1716/intra"
filename = os.path.join(dirname, "1716G", "1716G_GAF1.DAT")


x = er.ElphyFile(filename, read_data=False)

print("{} {}".format(os.path.basename(x.file_name), x.file_size))
print("initial_objects {}".format(x.initial_objects))
print("{} episodes".format(x.n_episodes))

# episodes = x.episodes

#     - file_name, file_size   # file info
#     - initial_objects                 # array of objects that appear before
#     the first episode in the file
#     - n_episodes                         # number of episodes (=1 for continuous recording)
#     - episodes

# help(x.episodes[0].decode_episode_info)
