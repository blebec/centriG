#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  5 12:18:33 2020

@author: Benoît
"""


import getpass
import os
import platform

import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat

#import humps

def go_to_dir():
    """
    to go to the pg and file directory (spyder use)
    """
    osname = platform.system()
    username = getpass.getuser()
    if osname == 'Windows' and username == 'Benoit':
        os.chdir(r'C:\Users\Benoît\Desktop\centriExtra\data')        
    #elif osname == 'Linux' and username == 'benoit':
        #os.chdir(r'/media/benoit/data/travail/sourcecode/developing/paper/centriG')
    elif osname == 'Darwin' and username == 'cdesbois':
        os.chdir('/Users/cdesbois/ownCloud/cgFigures/elphyDataExport')
    return True
go_to_dir()

cwd = os.getcwd()
#print(cwd)

#%%
#each key is of type: PSTH_STIM_n1_ELEC_n2_TRIAL_n3
#stims of interest: 10 = 'center-only' high speed; 12: surround-then-center hs
# n1 stim = 19 (2 of interest) , n2 electrode = 64, n3 trial = 30

def load_data_file(filename,
                   dt):
    """
    returns a dictionary with keys corresponding to [conditions,electrodes]
    At the level of [conditions], each element in an array of array where
    each element is of of shape [nelec][ntrials][npoints]
    At the level of [conditions][elec], each element is an array where 
    each elecment is of shape [ntrials][npoints]
    """
    
    data = loadmat(filename)
    conditions = [10, 12]
    electrodes = np.linspace(1,64,64)
    trials = np.linspace(1,30,30)
    
    for key in data:
        if (key not in ['__header__', '__version__', '__globals__', 'DataFile']):
            reformated_data = {'t':np.arange(len(np.array(data['PSTH_STIM_1_ELEC_1_TRIAL_1']).flatten()))*dt}
            
            for cond in conditions:
                reformated_data[cond] = []
                for elec in electrodes:
                    reformated_data[cond, elec] = []
                    for trial in range(31):#trials:
                        if (cond == 10):
                            if 'PSTH_STIM_%d_ELEC_%d_TRIAL_%d' %(cond, elec, trial) in data:
                                reformated_data[cond,elec].append(np.array(data['PSTH_STIM_%d_ELEC_%d_TRIAL_%d' % (cond,elec,trial)]).flatten())

                        else:
                            if (cond == 12):
                                if 'PSTH_STIM_%d_ELEC_%d_TRIAL_%d' %(cond, elec, trial) in data: 
                                    reformated_data[cond,elec].append(np.array(data['PSTH_STIM_%d_ELEC_%d_TRIAL_%d' % (cond, elec,trial)]).flatten())
                    reformated_data[cond, elec] = np.array(reformated_data[cond,elec])        
                    reformated_data[cond].append(reformated_data[cond, elec])                
            reformated_data[cond] = np.array(reformated_data[cond])

    return reformated_data                        
#%% file specifications
filename ='PSTHS_TRIALS_2010_TUN21'
dt = 1/10E3
data = load_data_file(filename, dt)
 #%% check dimensions
#print(reformated_data.keys())
print(np.shape(data[10]))
print(np.shape(data[12]))
print(' ')

print(np.shape(data[10][63]))
print(np.shape(data[12][0]))
print(' ')

print(np.shape(data[10][63][0]))
print(np.shape(data[12][0][29]))
print(' ')
print(len(data[12][63][29]))

#print(data[10][25][29])
#print(data[12][25][29])

#%%
# 1TO DO  plot 64 electrodes, squeezed over the 30 trial dimension to replicate average MUA traces

#%%
#PSTH_STIM_n1_ELEC_n2_TRIAL_n3

def extract_name(alist):
    """ return elts = name elements """
    # built container
    elts = []
    for i in range(7):
        elts.append([])
    # extract alist elements
    for item in alist:
        item_elts = item.split('_')
        for i, elt in enumerate(item_elts):
            if elt not in elts[i]:
                elts[i].append(elt)
    for i, kind in enumerate(elts):
        print(i, ': ', kind)
    return elts

filename ='PSTHS_TRIALS_2010_TUN21'
dt = 1/10E3
data = loadmat(filename)
keys = list(data.keys())
keys = keys[3:]
elts = extract_name(keys)

#%%
name = 'PSTH' + '_' + 'STIM' + '_' + n1 + '_' + 'ELEC' + '_' + n2 + '_' + 'TRIAL' + '_' + n3

center_only_hs = [key for key in keys if key.split('_')[2] == '10']
surroundThenCenter_hs = [key for key in keys if key.split('_')[2] == '19']

def build_a_df_of_stims(keys):
    """ describe the names """
    df = pd.DataFrame(keys)
    df['stim'] = df[df.columns[0]].apply(lambda x : int(x.split('_')[2]))
    df['elect'] = df[df.columns[0]].apply(lambda x : int(x.split('_')[4]))
    df['trial'] = df[df.columns[0]].apply(lambda x : int(x.split('_')[6]))
    del df[df.columns[0]]
    return df
keys_df = build_a_df_of_stims(keys)

%%
 import xarray as xr