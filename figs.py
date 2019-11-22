

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
from matplotlib.patches import Rectangle

#colors
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
#%% define the font size to be used
params = {'font.sans-serif': ['Arial'],
          'font.size': 14,
          'legend.fontsize': 'small',
          'figure.figsize': (15, 5),
         'axes.labelsize': 'small',
         'axes.titlesize':'small',
         'xtick.labelsize':'small',
         'ytick.labelsize':'small',
         'axes.xmargin': 0
         }
plt.rcParams.update(params)
plt.rcParams['axes.xmargin'] = 0            # no gap between axes and traces
#%%
# to have the string value of a variable (for work process)
import inspect
def retrieve_name(var):
    callers_local_vars = inspect.currentframe().f_back.f_locals.items()
    return [var_name for var_name, var_val in callers_local_vars if var_val is var]


#%% adjust the y scale to allign plot for a value (use zero here)

#alignement to be performed
#see https://stackoverflow.com/questions/10481990/matplotlib-axis-with-two-scales-shared-origin/10482477#10482477

def align_yaxis(ax1, v1, ax2, v2):
    """adjust ax2 ylimit so that v2 in ax2 is aligned to v1 in ax1"""
    _, y1 = ax1.transData.transform((0, v1))
    _, y2 = ax2.transData.transform((0, v2))
    inv = ax2.transData.inverted()
    _, dy = inv.transform((0, 0)) - inv.transform((0, y1-y2))
    miny, maxy = ax2.get_ylim()
    ax2.set_ylim(miny+dy, maxy+dy)
   
   
def changePlotAmplitude(ax, gain=1):
    """change the amplitude of the plot, 
    doesn't change the zero location """
    lims = ax.get_ylim()
    newLims = (lims[0]/gain, lims[1]/gain)
    ax.set_ylim(newLims)
#%% expolore size and attributes
def property(ax):
    size = ax.axes.xaxis.label.get_size()
    fontname = ax.axes.xaxis.label.get_fontname()
    print('xaxis:', fontname, size)
    size = ax.axes.yaxis.label.get_size()
    fontname = ax.axes.yaxis.label.get_fontname()
    print('yaxis:', fontname, size)
    
    
def figProp(fig):
    for ax in fig.get_axes():
        property(ax)
    
#%%
plt.close('all')

def plotFig2():
    filename = 'fig2.xlsx'
    df = pd.read_excel(filename)
    #centering
    middle = (df.index.max() - df.index.min())/2
    df.index = df.index - middle
    df.index = df.index/10
    cols = df.columns
    colors = ['k', stdColors['rouge']]
    alpha = [0.8, 0.8]

    fig = plt.figure(figsize=(8,8))
    #fig.suptitle(os.path.basename(filename))
    ax1 = fig.add_subplot(221)
    for i, col in enumerate(cols[:2]):
        ax1.plot(df.loc[-200:150,[col]], color=colors[i], alpha=alpha[i], 
               label = col)
    ax1.set_ylabel ('membrane potential (mV)')
    ax1.spines['bottom'].set_visible(False)
    ax1.axes.get_xaxis().set_visible(False)
   
    ax3 = fig.add_subplot(223, sharex = ax1)
    # NB plotting as to in the reverse order (ie red before black)
    columns = cols[2:4][::-1]
    for i, col in enumerate(columns):
        ax3.plot(df.loc[-200:150][col], color=colors[::-1][i], 
                 alpha=1, label = col, linewidth=1)
        ax3.fill_between(df.loc[-200:150].index, df.loc[-200:150][col], 
                         color=colors[::-1][i], alpha=0.5, label = col)
    ax3.set_ylabel('firing rate (spikes/s)')
    ax3.set_xlabel('time (ms)')
    
    ax2 = fig.add_subplot(222)
    for i, col in enumerate(cols[4:6]):
        ax2.plot(df.loc[-30:35,[col]], color=colors[i], alpha=alpha[i], 
               label = col)
    ax2.annotate("n=37", xy=(0.2, 0.8), 
                xycoords="axes fraction", ha='center')
    ax2.set_ylabel ('normalized membrane potential')
    ax2.spines['bottom'].set_visible(False)
    ax2.axes.get_xaxis().set_visible(False)
    ax2.set_ylim(0,1)

    ax4 = fig.add_subplot(224, sharex=ax2)
    # NB plotting as to in the reverse order (ie red before black)
    columns = list(cols[6:8])[::-1]
    for i, col in enumerate(columns):
        ax4.plot(df.loc[-30:35][col], color=colors[::-1][i], 
                 alpha=1, label = col, linewidth=1)
        ax4.fill_between(df.loc[-30:35].index, df.loc[-30:35][col], 
                         color=colors[::-1][i], alpha=0.5, label = col)
    ax4.annotate("n=20", xy=(0.2, 0.8), 
                xycoords="axes fraction", ha='center')
    ax4.set_ylabel ('normalized firing rate')
    ax4.set_xlabel('relative time (ms)')
    ax4.set_ylim(0,1)
    # stimulations
    step = 28
    xlocs = np.arange(0, -150, -28)
    names = ['D0', 'D1', 'D2', 'D3', 'D4', 'D5']
    dico = dict(zip(names, xlocs))
    #lines
    for ax in [ax1, ax3]:
        lims = ax.get_ylim()
        for dloc in xlocs :
            ax.vlines(dloc, lims[0], lims[1], linestyle=':', alpha =0.2)
    # stim location
    for key in dico.keys():
        ax3.annotate(key, xy=(dico[key]+3,-3), alpha=0.6, fontsize='x-small')
        # stim
        rect = Rectangle(xy=(dico[key],-4), width=step, height=1, fill=True,
                     alpha=0.6, edgecolor='w', facecolor='r')
        ax3.add_patch(rect)
        #center
    rect = Rectangle(xy=(0,-5), width=step, height=1, fill=True,
                     alpha=0.6, edgecolor='w', facecolor='k')
    ax3.add_patch(rect)
    # clean axes
    for ax in fig.get_axes():
#        ax.set_title(retrieve_name(ax)) # for working purposes
        for loc in ['top', 'right']:
            ax.spines[loc].set_visible(False)
        lims = ax.get_ylim()
        ax.vlines(0, lims[0], lims[1], alpha =0.2)
        lims = ax.get_xlim()
        ax.hlines(0, lims[0], lims[1], alpha =0.2)       
    # align zero between plots
    align_yaxis(ax1, 0, ax2, 0)
    align_yaxis(ax3, 0, ax4, 0)        
    fig.tight_layout()
    # remove the space between plots
    fig.subplots_adjust(hspace=0.02)
    # adjust amplitude (without moving the zero    
    changePlotAmplitude(ax1, 1.1)
    changePlotAmplitude(ax2, 0.7)
    changePlotAmplitude(ax3, 1)
    changePlotAmplitude(ax4, 0.7)    
    
    return fig        

fig = plotFig2()
#%%
    
def plotFig2B():
    filename = 'fig2.xlsx'
    df = pd.read_excel(filename)
    cols = df.columns[-2:]
    # select the rank data
    rankDf = df[df.columns[-2:]].dropna().reset_index()
    del rankDf['index']
    rankDf.index +=1 # cells = 1 to 36

    fig = plt.figure(figsize=(8,2))
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)
    axes = [ax1, ax2]

    for i, ax in enumerate(axes):
        axes[i].bar(rankDf.index, rankDf[cols[i]], color=stdColors['rouge'], label=cols[i], 
            alpha = 0.8, width=0.8)
        ax.set_xlabel('cell rank')
        for loca in ['top', 'right', 'bottom']:
            ax.spines[loca].set_visible(False)
        lims = ax.get_xlim()
        ax.hlines(0, lims[0], lims[1], alpha=0.2)
        ticks = [rankDf.index.min(), rankDf.index.max()]
        ax.set_xticks(ticks)
    ax1.set_ylabel('phase advance (ms)')
    ax2.set_ylabel('delta response')
    fig.tight_layout()
    return fig

plotFig2B()
#%%
#plt.close('all')

def plotFig3():
    filename = 'fig3.xlsx'
    df = pd.read_excel(filename)
    #centering
    middle = (df.index.max() - df.index.min())/2
    df.index = df.index - middle
    df.index = df.index/10
    cols = ['CNT-ONLY', 'CP-ISO', 'CF-ISO', 'CP_CROSS', 'RND-ISO']
    df.columns = cols
    colors = ['k', stdColors['rouge'], stdColors['vert'], 
              stdColors['jaune'], stdColors['bleu']]
    alpha = [0.5, 0.5, 0.5, 1, 0.6]

    fig = plt.figure(figsize=(8,6))       ##SUGGESTION: make y dimension much larger to see maximize visual difference between traces 
#    fig.suptitle(os.path.basename(filename))
    ax = fig.add_subplot(111)
    for i, col in enumerate(cols):
        ax.plot(df[col], color=colors[i], alpha=alpha[i], 
               label = col)
    ax.set_ylabel ('normalized membrane potential')
    ax.set_xlabel ('relative time (ms)')
    for ax in fig.get_axes():
       for loc in ['top', 'right']:
           ax.spines[loc].set_visible(False)
    ax.set_xlim(-15, 30)
    lims = ax.get_ylim()
    ax.vlines(0, lims[0], lims[1], alpha =0.2)
    lims = ax.get_xlim()
    ax.hlines(0, lims[0], lims[1], alpha =0.2)
   
    leg = ax.legend(loc='center right', markerscale=None, frameon = False, 
                   handlelength=0)
    for line,text in zip(leg.get_lines(), leg.get_texts()):
           text.set_color(line.get_color())   
    ax.annotate("n=37", xy=(0.1, 0.8), 
                xycoords="axes fraction", ha='center')
    fig.tight_layout()
    return fig              

fig = plotFig3()  

#%%
#plt.close('all')
def plotFig4():
    """ speed """
    filename = 'fig4.xlsx'
    df = pd.read_excel(filename)
    #centering
    middle = (df.index.max() - df.index.min())/2   #OBSERVATION bottom raw 0 baseline has been decentered by police and ticks size changes
    df.index = df.index - middle
    df.index = df.index/10
    cols = ['centerOnly', '100%', '70%', '80%', '50%']
    df.columns = cols
    colors = ['k', stdColors['rouge'], speedColors['orangeFonce'], 
              speedColors['orange'], speedColors['jaune']]
    alpha = [0.5, 1, 0.8, 0.8, 1]

    fig = plt.figure(figsize=(8,4))
   # fig.suptitle(os.path.basename(filename))
    ax = fig.add_subplot(111)
    for i, col in enumerate(cols):
        ax.plot(df[col], color=colors[i], alpha=alpha[i], 
                   label = col)
    ax.set_ylabel ('normalized membrane potential')
    #, fontname = 'Arial', fontsize = 14)
    ax.set_xlabel ('relative time (ms)')
    for ax in fig.get_axes():
        for loc in ['top', 'right']:
            ax.spines[loc].set_visible(False)
    fig.tight_layout()
    #fig.legend()
    ax.set_xlim(-40, 45)
    ax.set_ylim(-0.1, 1.1)
    lims = ax.get_ylim()
    ax.vlines(0, lims[0], lims[1], alpha =0.2)
    lims = ax.get_xlim()
    ax.hlines(0, lims[0], lims[1], alpha =0.2)
    leg = ax.legend(loc='lower right', markerscale=None, frameon = False, 
                    handlelength=0)
    for line,text in zip(leg.get_lines(), leg.get_texts()):
        text.set_color(line.get_color())
    ax.annotate("population average \n (n=12)", xy=(0.2, 0.8), 
                xycoords="axes fraction", ha='center')
    return fig

fig = plotFig4()                

#%%
plt.close('all')
    
def plotFig5():
    filename = 'fig5.xlsx'
    df = pd.read_excel(filename)
    #centering
    middle = (df.index.max() - df.index.min())/2
    df.index = df.index - middle
    df.index = df.index/10
    # rename columns
    cols = df.columns
    cols = ['center only', 'surround then center', 'surround only', 
                    'static linear prediction']
    dico = dict(zip(df.columns, cols))
    df.rename(columns=dico, inplace=True)
    # color parameters
    colors = ['k', stdColors['rouge'], stdColors['bleu'], stdColors['vert']]
    alpha = [0.5, 0.7, 0.8, 0.8]
    #plotting
    fig = plt.figure(figsize=(6,8))             # SUGGESTION increase a bit y dimension or subplots height
#    fig.suptitle(os.path.basename(filename))
    ax1 = fig.add_subplot(211)
    for i, col in enumerate(cols[:2]):
        ax1.plot(df.loc[-120:200,[col]], color=colors[i], alpha=alpha[i], 
               label = col)
    ax1.spines['bottom'].set_visible(False)
    ax1.axes.get_xaxis().set_visible(False)
    
    ax2 = fig.add_subplot(212, sharex=ax1, sharey=ax1)
    for i, col in enumerate(cols):
        ax2.plot(df.loc[-120:200,[col]], color=colors[i], alpha=alpha[i], 
               label = col)

    ax2.set_xlabel ('time (ms)')
    # stims
    step = 20
    hlocs = np.arange(0, -110, -step)
    names = ['D0', 'D1', 'D2', 'D3', 'D4', 'D5']
    vlocs = np.linspace(-0.7, -1.6, 4)
#    vlocs = [-0.7, -1, -1.3, -1.6]
    dico = dict(zip(names, hlocs))        
    #ax1
    for key in dico.keys():
        #name
        ax1.annotate(key, xy=(dico[key]+3,vlocs[0]), alpha=0.6, 
                     annotation_clip=False, fontsize='small')
        #stim1
        rect = Rectangle(xy=(dico[key],vlocs[1]), width=step, height=0.3, 
                         fill=True,alpha=0.6, edgecolor='w', 
                         facecolor='r')
        ax1.add_patch(rect)       
    #center
    rect = Rectangle(xy=(0,vlocs[2]), width=step, height=0.3, fill=True,
                     alpha=0.6, edgecolor='w', facecolor='k')
    ax1.add_patch(rect)
    
    st = 'surround then center'
    ax1.annotate(st, xy=(30,vlocs[1]), color=colors[1], 
                 annotation_clip = False, fontsize='small')
    st = 'center only'
    ax1.annotate(st, xy=(30,vlocs[2]), color=colors[0], 
                 annotation_clip = False, fontsize='small')
        # see annotation_clip = False
    ax1.set_ylim(-1.8, 4.5)

    #ax2
    for key in dico.keys():
        # names
        ax2.annotate(key, xy=(dico[key]+3,vlocs[0]), alpha=0.6,
                     annotation_clip = False, fontsize='small')
        # stim1
        rect = Rectangle(xy=(dico[key],vlocs[1]), width=step, height=0.3, 
                         fill=True, alpha=0.6, edgecolor='w', 
                         facecolor=colors[2])
        if key == 'D0':
            rect = Rectangle(xy=(dico[key],vlocs[1]), width=step, height=0.3, 
                             fill=True, alpha=0.6, edgecolor='w', 
                             facecolor='w')
        ax2.add_patch(rect)
        # stim2
        rect = Rectangle(xy=(dico[key],vlocs[2]), width=step, height=0.3, 
                         fill=True,alpha=0.6, edgecolor='w', 
                         facecolor=colors[1])
        ax2.add_patch(rect)
    # center
    rect = Rectangle(xy=(0,vlocs[3]), width=step, height=0.3, fill=True,
                     alpha=0.6, edgecolor='w', facecolor=colors[0])
    ax2.add_patch(rect)  
    for i,st in enumerate(['surround only', 'surround then center', 'center only']):
        ax2.annotate(st, xy=(30,vlocs[i+1]), color=colors[2-i], 
                     annotation_clip = False, fontsize='small') 
    for ax in fig.get_axes():
        leg = ax.legend(loc='upper right', markerscale=None, frameon = False, 
                        handlelength=0)
        # colored text
        for line,text in zip(leg.get_lines(), leg.get_texts()):
            text.set_color(line.get_color())
        ax.set_ylabel ('membrane potential (mV)')
        for loc in ['top', 'right']:
           ax.spines[loc].set_visible(False)
        lims = ax.get_xlim()
        ax.hlines(0, lims[0], lims[1], alpha =0.2)
        lims = ax.get_ylim()
        ax.vlines(0, lims[0], lims[1], alpha =0.2)
        for dloc in [-20, -40, -60, -80, -100]:
            ax.vlines(dloc, lims[0], lims[1], linestyle=':', alpha =0.2)    
    fig.tight_layout()
    return fig                

fig = plotFig5()

#%%
def plotFig6():
    filename = 'fig6.xlsx'
    df = pd.read_excel(filename)
    #centering
    middle = (df.index.max() - df.index.min())/2
    df.index = df.index - middle
    df.index = df.index/10
    cols = ['centerOnly', 'surroundThenCenter', 'surroundOnly', 
                   'sdUp', 'sdDown', 'linearPrediction']
    dico = dict(zip(df.columns, cols))
    df.rename(columns=dico, inplace=True)
    cols = df.columns
    colors = ['k', 'r', 'b', 'g', 'b', 'b']
    colors = ['k', stdColors['rouge'], stdColors['bleu'], 
              stdColors['violet'], stdColors['violet'], stdColors['violet']]
    alpha = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5]

    fig = plt.figure(figsize=(12,6))
   # fig.suptitle(os.path.basename(filename))
    ax1 = fig.add_subplot(121)
    for i, col in enumerate(cols[:3]):
        ax1.plot(df[col], color=colors[i], alpha=alpha[i], 
               label = col)
    ax1.set_ylim(-0.2, 1)
    ax2 = fig.add_subplot(122, sharex=ax1)
    for i in [2,5]:
        print('i=', i, colors[i]), 
        ax2.plot(df[df.columns[i]], color=colors[i], alpha=alpha[i], 
               label = df.columns[i])
    ax2.fill_between(df.index, df[df.columns[3]], df[df.columns[4]], 
                     color=colors[2],alpha=0.2)
    
    # set fontname and fontsize for y label
    ax1.set_ylabel ('normalized membrane potential (mV)')
    #, fontname = 'Arial', fontsize = 14)
    for ax in fig.get_axes():
        leg = ax.legend(loc='upper left', markerscale=None, frameon = False, 
                        handlelength=0)
        # colored text
        for line,text in zip(leg.get_lines(), leg.get_texts()):
            text.set_color(line.get_color())

        ax.set_xlim(-150, 150)
        # set fontname and fontsize for x label        
        ax.set_xlabel ('relative time (ms)')
        #, fontname = 'Arial', fontsize = 14)
        for loc in ['top', 'right']:
           ax.spines[loc].set_visible(False)
        lims = ax.get_xlim()
        ax.hlines(0, lims[0], lims[1], alpha =0.2)
        lims = ax.get_ylim()
        ax.vlines(0, lims[0], lims[1], alpha =0.2)
        
         ## set fontname and fontsize for x and yticks
#        labels = ax.get_xticklabels() + ax.get_yticklabels()
#        [label.set_fontname('Arial') for label in labels]
#        [label.set_fontsize(13) for label in labels]
    
    
    # align zero between subplots
    align_yaxis(ax1, 0, ax2, 0)
    fig.tight_layout()
    
        
    
    # add ref
    ref = (0, df.loc[0, ['centerOnly']])
    
    return fig                

fig = plotFig6()

#%% opt
colors = ['k', stdColors['rouge'], speedColors['orangeFonce'], 
              speedColors['orange'], speedColors['jaune']]
alpha = [0.8, 1, 0.8, 0.8, 1]

df = pd.read_excel('figOpt.xlsx')
df.set_index('time', inplace=True)

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
        right_ax.plot(df.loc[40:100,[col]], color=colors[i], 
                      label=col, alpha = alpha[i])
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
