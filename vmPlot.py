


import sys
import platform
import os
import getpass
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

 
# =============================================================================
# osname1 = sys.platform
# osname = osname1
# osname2 = platform.system()
# username = getpass.getuser()
# 
# =============================================================================
##//osname = platform.system()
##//username = getpass.getuser()
def config():
    """
    to go to the pg and file directory (spyder use)
    """
    paths = {}

    osname1 = sys.platform
    osname = osname1
    osname2 = platform.system()
    username = getpass.getuser()
    osname1 = sys.platform
    osname2 = platform.system()
    if (osname1 == 'Windows') or (osname2 =='Windows') and username == 'Benoît':
        os.chdir(r'D:\\travail\sourcecode\developing\paper\centriG')
        paths['cgFig'] = 'D:\\owncloud\cgFiguresSrc'
        paths['save'] = 'D:\\owncloud\cgFiguresSrc'
        paths['traces'] = 'D:\\owncloud\\cgFiguresSrc\\averageTraces\\'
    elif (osname1 == 'linux') or (osname2 =='Linux') and username == 'benoit':
        os.chdir(r'/media/benoit/data/travail/sourcecode/developing/paper/centriG')
        paths['cgFig'] = '/media/benoit/data/owncloud/cgFiguresSrc'
        paths['save'] = '/media/benoit/data/owncloud/cgFiguresSrc'
        paths['traces'] = '/media/benoit/data/owncloud/cgFiguresSrc/averageTraces'
    elif osname == 'Windows'and username == 'marc':
        os.chdir(r'H:/pg/centriG')
    elif osname == 'darwin' and username == 'cdesbois':
        os.chdir(r'/Users/cdesbois/pg/chrisPg/centriG')
        paths['cgFig'] = os.path.expanduser('~/ownCloud/cgFigures')
        paths['save'] = '/Users/cdesbois/ownCloud/cgFigures'
        paths['traces'] = '/Users/cdesbois/ownCloud/cgFigures/averageTraces'
    return paths


paths = config()
info_df = pd.read_excel(os.path.join(paths['traces'], 'neuron_props.xlsx'))
# =============================================================================
# if osname == 'darwin' and username == 'cdesbois':
# elif (osname1 == 'Linux') or (osname2 == 'Linux') and username == 'benoit':
#     info_df = pd.read_excel(paths['traces'] + 'neuron_props.xlsx')
# elif (osname1 == 'Windows') or (osname2 == 'Windows') and username == 'Benoît':
#     info_df = pd.read_excel(paths['traces'] + 'neuron_props.xlsx')    
# 
# =============================================================================
#print(paths['traces'])
#print(osname1)
#print('')
#print(osname2)
#print('')
#print(username)
#print('')
#print(paths.keys())
# ref = df['CENTER-ONLY'] 
# df = df.subtract(ref, axis=0) #on average traces already normalized

#%% to be checked:
sig_cells = ['1427A_CXG4',
             '1429D_CXG8',
             '1509E_CXG4',
             '1512F_CXG6',
             '1516F_CXG2',
             '1516G_CXG2',
             '1516M_CXG2',
             '1527B_CXG2',
             '1622H_CXG3',
             '1638D_CXG5']

#%%
plt.close('all')

def load_all_vm_traces(info_df):
    stims = [item for item in os.listdir(os.path.join(paths['traces'], 'vm'))
             if item[0] != '.']
    conds = {}
    for stim in stims:
        filename = os.path.join(paths['traces'], 'vm', stim)
        df = pd.read_csv(filename, sep='\t', header=None)
        df.columns = info_df.Neuron.to_list()
        # center and scale
        df.index -= df.index.max()/2
        df.index /= 10
        conds[stim] = df.copy()
    return conds

def normalize(dico):
    """ divide by centerOnly"""
    #build list
    res = dico.copy()
    conds = list(dico.keys())
    conds.remove('blank.txt')
    ref = 'ctronly.txt'
    # conds.remove(ref)  # to normalize the centerOnly
    # normalize
    data_ref = dico[ref]
    for cond in conds:
        df = dico[cond]
        res[cond] = df.divide(data_ref.max())
    return res

def align_traces(vm_dico, info_df):
    """" align traces on center only response"""
    ser = info_df.set_index('Neuron')['time3stddev']
    out_dico = {}
    for cond in vm_dico.keys():
        df = vm_dico[cond].copy()
        for cell in ser.index:
            df[cell] = df[cell].shift(-int(ser[cell]*10))
        out_dico[cond]= df
    return out_dico
        
def plot_res(vm_dico, norm_dico):
    cells = ['1424M_CXG16', '1638D_CXG5']
    cond = 'cpisosec.txt'
    fig = plt.figure()
    ax = fig.add_subplot(111)
    axT = ax.twinx()
    for cell in cells:
        ax.plot(vm_dico[cond][cell], label=cell)
    for cell in cells:
        axT.plot(norm_dico[cond][cell], 'k:', label=cell)
    ax.legend()
    axT.legend()
        
vm_dico = load_all_vm_traces(info_df)
vm_dico = normalize(vm_dico)
align_dico = align_traces(vm_dico, info_df)

plot_res(vm_dico, align_dico)


#%%
#for cond in data_cond.columns:

def testPlot(vm_dico):
    cond = 'cpisosec.txt'
    data_cond = vm_dico[cond][sig_cells]
    ref = 'ctronly.txt' 
    data_ref = vm_dico[ref][sig_cells]

    fig = plt.figure()
    fig.suptitle('several ways to look at')
    ax = fig.add_subplot(411)
    for col in data_cond.columns:
        ax.plot(data_cond[col], 'r-', alpha=0.6)#, label=col.split('_')[0])
        ax.plot(data_ref[col], 'k-', alpha=0.6)
    ax = fig.add_subplot(412)
    ax.plot(data_cond.mean(axis=1), 'r-', alpha=0.6, label=cond)
    ax.plot(data_ref.mean(axis=1), 'k-', alpha=0.6, label=ref)
    axT = ax.twinx()
    axT.plot(np.cumsum(data_cond.mean(axis=1)), 'r-', alpha=0.6, label='cumsum')
    axT.plot(np.cumsum(data_ref.mean(axis=1)), 'k-', alpha=0.6, label='cumsum')
    # axT.legend()
    # ax.legend()
    ax = fig.add_subplot(413)
    diff = data_cond.mean(axis=1) - data_ref.mean(axis=1)
    ax.plot(diff, '-g', 
            linewidth=2, alpha=0.8, label = 'diff')
    x = diff.index.values
    ax.fill_between(x=x, y1=diff,color='g', alpha=0.3)
    # ax.legend()
    ax = fig.add_subplot(414)
    ax.plot(np.cumsum(diff), label='cumsum')
    ax.plot(np.cumsum(diff -diff.mean()), label='-mean & cumsum')
    # ax.legend()
    for i, ax in enumerate(fig.get_axes()):
        ax.legend()
        for loca in ['top', 'right']:
            ax.spines[loca].set_visible(False)
        lims = ax.get_xlim()
        ax.hlines(0, lims[0], lims[1], alpha=0.3)
        lims = ax.get_ylim()
        ax.vlines(0, lims[0], lims[1], alpha=0.3)
        if i <4:
            ax.xaxis.set_visible(False)
            ax.spines['bottom'].set_visible(False)
            #fig.legend()        
    fig.tight_layout()
    
testPlot(vm_dico)
#%%