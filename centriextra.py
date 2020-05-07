# -*- coding: utf-8 -*-
"""
Created on Tue May  5 12:18:33 2020

@author: Benoît
"""


import platform
import os
import getpass
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat
import humps

def go_to_dir():
    """
    to go to the pg and file directory (spyder use)
    """
    osname = platform.system()
    username = getpass.getuser()
    if osname == 'Windows'and username == 'Benoit':
        os.chdir(r'C:\Users\Benoît\Desktop\centriExtra')
        
    #elif osname == 'Linux' and username == 'benoit':
        #os.chdir(r'/media/benoit/data/travail/sourcecode/developing/paper/centriG')
        
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
        if (key not in ['__header__', '__version__', '__globals__','DataFile']):
            reformated_data = {'t':np.arange(len(np.array(data['PSTH_STIM_1_ELEC_1_TRIAL_1']).flatten()))*dt}
            
            for cond in conditions:
                reformated_data[cond] = []
                for elec in electrodes:
                    reformated_data[cond,elec] = []
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
filename ='data/PSTHS_TRIALS_2010_TUN21'
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