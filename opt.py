

import platform
import os
import getpass

def goToDir():
    osname = platform.system()
    username = getpass.getuser()

    if osname == 'Windows'and username == 'Benoit':
        os.chdir('D:\\travail\sourcecode\developing\paper\centriG')
    elif osname == 'Linux' and username == 'benoit':
        os.chdir('/media/benoit/data/travail/sourcecode/developing/paper/centriG')
    elif osname == 'Darwin' and username == 'cdesbois':
        os.chdir('/Users/cdesbois/pg/chrisPg/centriG')
    return(True)
goToDir()

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
plt.rcParams['axes.xmargin'] = 0
plt.rcParams.update({'font.size':12})

#%%
stdColors = {
        'rouge' : [x/256 for x in [229, 51, 51]],
        'vert' : [x/256 for x in [127,	 204, 56]],
        'bleu' :	[x/256 for x in [0, 125, 218]],
        'jaune' :	[x/256 for x in [238, 181, 0]],
        'violet' : [x/256 for x in [255, 0, 255]]
            }
speedColors ={
        'orangeFonce' :     [x/256 for x in [252, 98, 48]],
        'orange' : [x/256 for x in [253, 174, 74]],
        'jaune' : [x/256 for x in [254, 226, 137]]
        }

colors = ['k', stdColors['rouge'], speedColors['orangeFonce'], 
              speedColors['orange'], speedColors['jaune']]
alpha = [0.8, 1, 0.8, 0.8, 1]

#colors = [stdColors['rouge'], speedColors['orangeFonce'], 
#              speedColors['orange'], speedColors['jaune'], 'k']
#alpha = [1, 0.8, 0.8, 1, 0.5]

df = pd.read_excel('figOpt.xlsx')
df.set_index('time', inplace=True)

# to change the order
#cols = list(df)
#cols = cols[1:] + [cols[0]]
#df = df[cols]

#%%
#plt.close('all')

#double plot with std
def plotSpeedSD(df):
    fig = plt.figure()
    ax0 = fig.add_subplot(121)
    for i, col in enumerate(df.columns):
        ax0.plot(df.loc[-400:40,[col]] + i/5, color=colors[i], label=col)
        mean = float(df.loc[-400:0, [col]].mean()) + i/5
        std = float(df.loc[-400:0, [col]].std())
        ymin = (mean - std)
        ymax = (mean + std)
        ax0.axhspan(ymin=ymin , ymax=ymax, color=colors[i], alpha=alpha[i]/3)
        #, xmin=lims[0], xmax=0,  
    ax0.annotate(xy=(-400, 0.95), s='background = mean ± std in [-400:0 ms]')
   
    ax1 = fig.add_subplot(122)
    for i, col in enumerate(df.columns):
        ax1.plot(df.loc[0:150,[col]], color=colors[i], label=col, alpha = alpha[i])
        #ref pre
        #    mean = float(df.loc[-400:0, [col]].mean())
        #    std = float(df.loc[-400:0, [col]].std())
        #    ysup = (mean + std)
        #    ax1.hlines(ysup, 0, 60, color=colors[i])
        #response max
        max = float(df.loc[30:200,[col]].max())
        ax1.hlines(max, 40, 50, color=colors[i])
        #ax1.annotate(xy=(60, 0.005), s='- : mean + std in [-400:0 ms]')


    for ax in fig.get_axes():
        lims = ax.get_ylim()
        ax.vlines(0, lims[0], lims[1], alpha=0.5)
        for loca in ['left', 'top', 'right']:
            ax.spines[loca].set_visible(False)
        ax.yaxis.set_visible(False)
        ax.set_xlabel('time (ms)')


    leg = ax1.legend(loc='center right', markerscale=None, 
             handlelength=0, framealpha=1)
    for line,text in zip(leg.get_lines(), leg.get_texts()):
        text.set_color(line.get_color())   

    fig.tight_layout()
    return fig

fig = plotSpeedSD(df)

#%% gridspec
plt.close('all')

def plotSpeedMultigraph():
    """
    plot the speed effect of centirgabor protocol
    """
    fig = plt.figure(figsize=(12,8))
    fig.suptitle('aligned on Center-Only stimulus onset (t = 0)')
    # build grid
    gs = fig.add_gridspec(5,2)
    left_axes = []
    for i in range(5):
        ax = fig.add_subplot(gs[i,0])
        left_axes.append(ax)
    right_ax = fig.add_subplot(gs[:,1])
    # to identify the plots (uncomment to use)
#    for i, ax in enumerate(left_axes):
#        st = str('ax {}'.format(i))
#        ax.annotate(st, (0.5, 0.5))
    #(manipulate the left_axes list to reorder the plots if required)

    #plot left
    cols = df.columns
    for i, ax in enumerate(left_axes):
        ax.plot(df.loc[-140:40, [cols[i]]], color= 'black', scalex =False, 
                scaley=False, label = cols[i])
        ax.fill_between(df.index, df[cols[i]], color= colors[i]) 
        ax.yaxis.set_ticks(np.arange(-0.15,0.25,0.1))    
        ax.set_xlim(-140,40)    
        ax.set_ylim(-0.15,0.25)    
    #add labels
    left_axes[2].set_ylabel('Normalized Membrane potential')
    left_axes[-1].set_xlabel('Relative time to center-only onset (ms)')
    left_axes[-1].xaxis.set_ticks(np.arange(-140,41,20))           
    #plot right    
    for i, col in enumerate(df.columns):
        right_ax.plot(df.loc[40:100,[col]], color=colors[i], label=col, alpha = alpha[i])
        max = float(df.loc[30:200,[col]].max())
        right_ax.hlines(max, 40, 50, color=colors[i])
    right_ax.set_xlabel('Relative time to center-only onset (ms)')    
    # adjust
    for ax in fig.get_axes():
        for loca in ['top', 'right']:
            ax.spines[loca].set_visible(False)
    for ax in left_axes:
        lims = ax.get_ylim()
        ax.vlines(0, lims[0], lims[1], alpha=0.5)
    # adjust spacing
    gs.update(wspace=0.2, hspace=0.05)
    # add ticks to the top
    right_ax.tick_params(axis='x', bottom =True, top = True)
    #legend
    leg = right_ax.legend(loc='lower right', markerscale=None, 
                          handlelength=0, framealpha=1)
    for line,text in zip(leg.get_lines(), leg.get_texts()):
        text.set_color(line.get_color())   

    fig.tight_layout()
    return fig

fig = plotSpeedMultigraph()
