

import os
import matplotlib.pyplot as plt
import config
import load_data as ldat
import numpy as np
import  pandas as pd
from datetime import datetime

anot = True           # to draw the date and name on the bottom of the plot
std_colors = config.std_colors()
speedColors = config.speed_colors()
plt.rcParams.update(config.rc_params())
paths = config.build_paths()
os.chdir(paths['pg'])

#%%
paths['data'] = os.path.expanduser('~/pg/chrisPg/centriG/data/data_to_use')

def load_traces(paths, kind='vm', spread='sect', num=2):
    if kind == 'vm' and spread == 'sect':
        files = ['vmSectRaw.xlsx', 'vmSectNorm.xlsx', 'vmSectNormAlig.xlsx']
    elif kind == 'vm' and spread == 'full':
        files = ['vmFullRaw.xlsx', 'vmFullNorm.xlsx', 'vmFullNormAlig.xlsx']
    elif kind == 'spk' and spread == 'sect':
        files = ['spkSectRaw.xlsx', 'spkSectNorm.xlsx', 'spkSectNormAlig.xlsx']
    elif kind == 'spk' and spread == 'full':
        files = ['spkFullRaw.xlsx', 'spkFullNorm.xlsx', 'spkFullNormAlig.xlsx']
    else:
        print('load_traces: kind should be updated')
    file = files[num]
    filename = os.path.join(paths['data'], file)
    df = pd.read_excel(filename)
    # time
    middle = (df.index.max() - df.index.min())/2
    df.index = df.index - middle
    df.index = df.index/10
    
    label = file.split('.')[0]
    return label, df




def plot_align_normalize(label, data):
    """
    """
    def select_pop(df, filt='pop'):
        cols = df.columns
        if filt == 'pop':
            pop = [item for item in cols if 'n15' in item]
            df = df[pop].copy()
            df.columns = [item.replace('n15', '') for item in pop]
        elif filt == 'spk':
            spks = [item for item in cols if 'n6' in item]
            df = df[spks].copy()
            df.columns = [item.replace('n6', '') for item in spks]
        elif filt == 'spk2s':    
            spk2s = [item for item in cols if 'n5' in item]
            df = df[spk2s].copy()
            df.columns = [item.replace('n5', '') for item in spk2s]
        else:
            return
        return df

    # cols = ['CENTER-ONLY', 'CP-ISO', 'CF-ISO', 'CP-CROSS', 'RND-ISO']
    # df.columns = cols
    colors = ['k', std_colors['red'], std_colors['green'],
              std_colors['yellow'], std_colors['blue'],
              std_colors['blue']]
    alphas = [0.8, 1, 0.8, 0.8, 0.8, 0.8]
    # if substract:
    #     # subtract the centerOnly response
    #     ref = df['CENTER-ONLY']
    #     df = df.subtract(ref, axis=0)

    fig, axes = plt.subplots(ncols=1, nrows=3, figsize=(6, 18), 
                             sharex=True, sharey=True)
    # fig.suptitle(label, alpha=0.4)
    fig.text(x=0.05, y= 0.95, s=label, alpha = 0.6)
    for i, k in enumerate(['pop', 'spk', 'spk2s']):
        ax = axes[i]
        ax.set_title('pop = ' + k + ' _sig', alpha=0.6)
        df = select_pop(data, filt=k)
        cols = df.columns
        for j, col in enumerate(cols):
            ax.plot(df[col], color=colors[j], alpha=alphas[j], label=col, 
                    linewidth=2)
        for loc in ['top', 'right']:
            ax.spines[loc].set_visible(False)
        ax.set_ylabel('vm')
        if i == 2:
            ax.set_xlabel('relative time (ms)')
    ax.set_xlim(-20, 120)
    for ax in axes:
        lims = ax.get_ylim()
        ax.vlines(0, lims[0], lims[1], alpha=0.3)
        lims = ax.get_xlim()
        ax.hlines(0, lims[0], lims[1], alpha=0.3)
    fig.tight_layout()

    if anot:
        date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        fig.text(0.99, 0.01, 'alignNorm.py:plot_align_normalize',
                 ha='right', va='bottom', alpha=0.4)
        fig.text(0.01, 0.01, date, ha='left', va='bottom', alpha=0.4)
    return fig

savepath = '/Users/cdesbois/ownCloud/cgFigures/pythonPreview/proposal/3'
save = False

plt.close('all')
for kind in ['vm', 'spk']:
    for spread in ['sect', 'full']:
        for num in range(3):
            label, data = load_traces(paths, kind=kind, spread=spread, num=num)
            fig = plot_align_normalize(label, data)
            if save:
                fig.savefig(fname = os.path.join(savepath, label + '.png'))

#TODO to be checked

#%%
