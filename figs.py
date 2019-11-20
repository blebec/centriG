

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

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

#%% colors

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
#%% define the font to be used
from matplotlib import rc
#rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
#rc('font',**{'family':'serif','serif':['Times']})
rc('font',**{'family':'sans-serif','sans-serif':['Arial']})
rc('text', usetex=False)


#%% show the fonts available:
for font in plt.rcParams['font.sans-serif']:
    print (font)
for font in plt.rcParams['font.serif']:
    print (font)

#%% to have the sting value of a variable (for work process)
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
    alpha = [0.5, 0.5]

    fig = plt.figure(figsize=(8,8))
    fig.suptitle(os.path.basename(filename))

    ax1 = fig.add_subplot(221)
    for i, col in enumerate(cols[:2]):
        ax1.plot(df[col], color=colors[i], alpha=alpha[i], 
               label = col)
    #fontname and size of labels
    ax1.set_ylabel ('membrane potential (mV)', fontname = 'Arial', fontsize = 14)
    ax1.set_xlim(-200, 150)
    ax1.spines['bottom'].set_visible(False)
    ax1.axes.get_xaxis().set_visible(False)

   
    ax3 = fig.add_subplot(223, sharex = ax1)
    # NB plotting as to in the reverse order (ie red before black)
    columns = cols[2:4][::-1]
    for i, col in enumerate(columns):
        ax3.fill_between(df.index, df[col], color=colors[::-1][i], alpha=0.7, 
               label = col)
    #fontname and size of labels    
    ax3.set_ylabel('firing rate (spikes/s)', fontname = 'Arial', fontsize = 14)
    ax3.set_xlabel('time (ms)', fontname = 'Arial', fontsize = 14)
    
    ax2 = fig.add_subplot(222)
    for i, col in enumerate(cols[4:6]):
        ax2.plot(df[col], color=colors[i], alpha=alpha[i], 
               label = col)
    #fontname and size of annotations
    ax2.annotate("n=37", xy=(0.2, 0.8), 
                xycoords="axes fraction", ha='center', fontname ='Arial', fontsize = 14)
    ax2.set_ylabel ('normalized \n membrane potential', fontname = 'Arial', fontsize = 14)
    ax2.spines['bottom'].set_visible(False)
    ax2.axes.get_xaxis().set_visible(False)


    ax4 = fig.add_subplot(224, sharex=ax2)
    # NB plotting as to in the reverse order (ie red before black)
    columns = list(cols[6:8])[::-1]
    for i, col in enumerate(columns):
        ax4.fill_between(df.index, df[col], color=colors[::-1][i], alpha=0.7, 
               label = col)
    ax4.set_xlim(-30, 35)
    #fontname and size of annotations    
    ax4.annotate("n=20", xy=(0.2, 0.8), 
                xycoords="axes fraction", ha='center', fontname ='Arial', fontsize = 14)
    ax4.set_ylabel ('normalized \n firing rate', fontname = 'Arial', fontsize = 14)
    ax4.set_xlabel('relative time (ms)', fontname = 'Arial', fontsize = 14)
    
    # stimulations
    
    #step = 20
    # real stepsize 
    step = 28     
    names = ['D0', 'D1', 'D2', 'D3', 'D4', 'D5']
    #locs = [0, -20, -40, -60, -80, -100]
    #real Locs
    locs = [0, -28, -56, -84, -112, -140]    
    dico = dict(zip(names, locs))
    for key in dico.keys():
        # names + fontname and size of annotations
        ax3.annotate(key, xy=(dico[key]+3,-3), alpha=0.6, fontname = 'Arial', fontsize='small')
        # stim
        rect = Rectangle(xy=(dico[key],-4), width=step, height=1, fill=True,
                     alpha=0.6, edgecolor='w', facecolor='r')
        ax3.add_patch(rect)
        #center
    rect = Rectangle(xy=(0,-5), width=step, height=1, fill=True,
                     alpha=0.6, edgecolor='w', facecolor='k')
    ax3.add_patch(rect)

    
    
    for ax in fig.get_axes():
        ax.set_title(retrieve_name(ax)) # for working purposes
        for loc in ['top', 'right']:
            ax.spines[loc].set_visible(False)
            lims = ax.get_ylim()
            ax.vlines(0, lims[0], lims[1], alpha =0.2)
            lims = ax.get_xlim()
            ax.hlines(0, lims[0], lims[1], alpha =0.2)
    ## set fontname and fontsize for x and yticks
        labels = ax.get_xticklabels() + ax.get_yticklabels()
        [label.set_fontname('Arial') for label in labels]
        [label.set_fontsize(13) for label in labels]
    for ax in [ax1, ax3]:
        lims = ax.get_ylim()
#TODO : adjust the locations    see annotation _clip = False #DONE
#TDOD : adjust the stim bar chart locations                  #DONE
        for dloc in [0, -28, -56, -84, -112, -140]:
            ax.vlines(dloc, lims[0], lims[1], linestyle=':', alpha =0.2)
    
    
    # align zero between plots
    align_yaxis(ax1, 0, ax2, 0)
    
    fig.tight_layout()
    # remove the space between plots
    fig.subplots_adjust(hspace=0.02)
    
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
    alpha = [0.5, 0.5]

    fig = plt.figure(figsize=(8,2))
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)
    axes = [ax1, ax2]

    for i, ax in enumerate(axes):
        axes[i].bar(rankDf.index, rankDf[cols[i]], color=stdColors['rouge'], label=cols[i], 
            alpha = 0.7, width=0.8)
        ax.set_xlabel('cell rank', fontname = 'Arial', fontsize = 14)
        
        for loca in ['top', 'right', 'bottom']:
            ax.spines[loca].set_visible(False)
        lims = ax.get_xlim()
        ax.hlines(0, lims[0], lims[1], alpha=0.2)
        ticks = [rankDf.index.min(), rankDf.index.max()]
        ax.set_xticks(ticks)
    ax1.set_ylabel('phase advance (ms)', fontname ='Arial', fontsize = 14)
    ax2.set_ylabel('delta response'    , fontname ='Arial', fontsize = 14)
    
    ## set fontname and fontsize for x and yticks
    for ax in fig.get_axes():
        labels = ax.get_xticklabels() + ax.get_yticklabels()
        [label.set_fontname('Arial') for label in labels]
        [label.set_fontsize(13) for label in labels]
        
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

    fig = plt.figure(figsize=(8,4))       ##SUGGESTION: make y dimension much larger to see maximize visual difference between traces 
    fig.suptitle(os.path.basename(filename))
    ax = fig.add_subplot(111)
    for i, col in enumerate(cols):
        ax.plot(df[col], color=colors[i], alpha=alpha[i], 
               label = col)
    # fontname and size of labels    
    ax.set_ylabel ('normalized membrane potential', fontname = 'Arial', fontsize = 14)
    ax.set_xlabel ('relative time (ms)', fontname = 'Arial', fontsize = 14)

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
    
    # fontname and size of annotations
    ax.annotate("n=37", xy=(0.1, 0.8), 
                xycoords="axes fraction", ha='center', fontname ='Arial', fontsize = 14 )
    
    ## set fontname and fontsize for x and yticks
    for ax in fig.get_axes():
        labels = ax.get_xticklabels() + ax.get_yticklabels()
        [label.set_fontname('Arial') for label in labels]
        [label.set_fontsize(13) for label in labels]    
    
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
    fig.suptitle(os.path.basename(filename))
    ax = fig.add_subplot(111)
    for i, col in enumerate(cols):
        ax.plot(df[col], color=colors[i], alpha=alpha[i], 
                   label = col)
    # set fontname and fontsize for labels
    ax.set_ylabel ('normalized membrane potential', fontname = 'Arial', fontsize = 14)
    ax.set_xlabel ('relative time (ms)', fontname = 'Arial', fontsize = 14)
           
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
    
    # set fontname and fontsize for annotations
    ax.annotate("population average \n (n=12)", xy=(0.2, 0.8), 
                xycoords="axes fraction", ha='center', fontname = 'Arial', fontsize = 14)
    ## set fontname and fontsize for x and yticks
    for ax in fig.get_axes():
        labels = ax.get_xticklabels() + ax.get_yticklabels()
        [label.set_fontname('Arial') for label in labels]
        [label.set_fontsize(13) for label in labels]
   
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
    fig.suptitle(os.path.basename(filename))
    ax1 = fig.add_subplot(211)
    for i, col in enumerate(cols[:2]):
        ax1.plot(df[col], color=colors[i], alpha=alpha[i], 
               label = col)
    ax1.spines['bottom'].set_visible(False)
    ax1.axes.get_xaxis().set_visible(False)
    

    ax2 = fig.add_subplot(212, sharex=ax1, sharey = ax1)
    for i, col in enumerate(cols):
        ax2.plot(df[col], color=colors[i], alpha=alpha[i], 
               label = col)

    # set fontname and fontsize for x labels 
    ax2.set_xlabel ('time (ms)', fontname= 'Arial', fontsize = 14)
    #adjust plot    
    ax1.set_xlim(-120, 200)
    
    # stims
    step = 20
    names = ['D0', 'D1', 'D2', 'D3', 'D4', 'D5']
    #locs = [0, -20, -40, -60, -80, -100]
    #real Locs    
    locs = [0, -21, -42, -63, -84, -105]    
    vlocs = [-0.7, -1, -1.3, -1.6]
    dico = dict(zip(names, locs))
        
    #ax1
    for key in dico.keys():
        # names and fontname and size of annotations 
        ax1.annotate(key, xy=(dico[key]+3,vlocs[0]), alpha=0.6, fontname = 'Arial', fontsize='small',
                     annotation_clip = False)
        #stim1
        rect = Rectangle(xy=(dico[key],vlocs[1]), width=step, height=0.3, fill=True,
                 alpha=0.6, edgecolor='w', facecolor='r')
        ax1.add_patch(rect)       
    #center
    rect = Rectangle(xy=(0,vlocs[2]), width=step, height=0.3, fill=True,
                     alpha=0.6, edgecolor='w', facecolor='k')
    ax1.add_patch(rect)
    
    st = 'surround then center'
    ax1.annotate(st, xy=(30,vlocs[1]), color=colors[1], annotation_clip = False)
    st = 'center only'
    ax1.annotate(st, xy=(30,vlocs[2]), color=colors[0], annotation_clip = False)
        # see annotation_clip = False

    #ax2
    for key in dico.keys():
        # names
        ax2.annotate(key, xy=(dico[key]+3,vlocs[0]), alpha=0.6, fontsize='small',
                     annotation_clip = False)
        # stim1
        rect = Rectangle(xy=(dico[key],vlocs[1]), width=step, height=0.3, fill=True,
                alpha=0.6, edgecolor='w', facecolor=colors[2])
        if key == 'D0':
            rect = Rectangle(xy=(dico[key],vlocs[1]), width=step, height=0.3, fill=True,
                 alpha=0.6, edgecolor='w', facecolor='w')
        ax2.add_patch(rect)
        # stim2
        rect = Rectangle(xy=(dico[key],vlocs[2]), width=step, height=0.3, fill=True,
                 alpha=0.6, edgecolor='w', facecolor=colors[1])
        ax2.add_patch(rect)
     # center
    rect = Rectangle(xy=(0,vlocs[3]), width=step, height=0.3, fill=True,
                     alpha=0.6, edgecolor='w', facecolor=colors[0])
    ax2.add_patch(rect)
    
    # set fontname and fontsize for annotations 
    st = 'surround only'
    ax2.annotate(st, xy=(30,vlocs[1]), color=colors[2], annotation_clip = False, fontname = 'Arial', fontsize = 13)
    st = 'surround then center'
    ax2.annotate(st, xy=(30,vlocs[2]), color=colors[1], annotation_clip = False, fontname = 'Arial', fontsize = 13)
    st = 'center only'
    ax2.annotate(st, xy=(30,vlocs[3]), color=colors[0], annotation_clip = False, fontname = 'Arial', fontsize = 13)
    
    for ax in fig.get_axes():
        leg = ax.legend(loc='upper right', markerscale=None, frameon = False, 
                        handlelength=0)
        # colored text
        for line,text in zip(leg.get_lines(), leg.get_texts()):
            text.set_color(line.get_color())
        # set fontname and fontsize for x labels 
        ax.set_ylabel ('membrane potential (mV)', fontname = 'Arial', fontsize = 14)
        for loc in ['top', 'right']:
           ax.spines[loc].set_visible(False)
        lims = ax.get_xlim()
        ax.hlines(0, lims[0], lims[1], alpha =0.2)
        lims = ax.get_ylim()
        ax.vlines(0, lims[0], lims[1], alpha =0.2)
        for dloc in [-20, -40, -60, -80, -100]:
            ax.vlines(dloc, lims[0], lims[1], linestyle=':', alpha =0.2)
         ## set fontname and fontsize for x and yticks
        labels = ax.get_xticklabels() + ax.get_yticklabels()
        [label.set_fontname('Arial') for label in labels]
        [label.set_fontsize(13) for label in labels]
    
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
    fig.suptitle(os.path.basename(filename))
    ax1 = fig.add_subplot(121)
    for i, col in enumerate(cols[:3]):
        ax1.plot(df[col], color=colors[i], alpha=alpha[i], 
               label = col)
    
    ax2 = fig.add_subplot(122, sharex=ax1)
    for i in [2,5]:
        print('i=', i, colors[i]), 
        ax2.plot(df[df.columns[i]], color=colors[i], alpha=alpha[i], 
               label = df.columns[i])
    ax2.fill_between(df.index, df[df.columns[3]], df[df.columns[4]], 
                     color=colors[2],alpha=0.2)
    
    # set fontname and fontsize for y label
    ax1.set_ylabel ('normalized membrane potential (mV)', fontname = 'Arial', fontsize = 14)
    for ax in fig.get_axes():
        leg = ax.legend(loc='upper left', markerscale=None, frameon = False, 
                        handlelength=0)
        # colored text
        for line,text in zip(leg.get_lines(), leg.get_texts()):
            text.set_color(line.get_color())

        ax.set_xlim(-150, 150)
        # set fontname and fontsize for x label        
        ax.set_xlabel ('relative time (ms)', fontname = 'Arial', fontsize = 14)
        for loc in ['top', 'right']:
           ax.spines[loc].set_visible(False)
        lims = ax.get_xlim()
        ax.hlines(0, lims[0], lims[1], alpha =0.2)
        lims = ax.get_ylim()
        ax.vlines(0, lims[0], lims[1], alpha =0.2)
        
         ## set fontname and fontsize for x and yticks
        labels = ax.get_xticklabels() + ax.get_yticklabels()
        [label.set_fontname('Arial') for label in labels]
        [label.set_fontsize(13) for label in labels]
    
    
    # align zero between subplots
    align_yaxis(ax1, 0, ax2, 0)
    fig.tight_layout()
    
        
    
    # add ref
    ref = (0, df.loc[0, ['centerOnly']])
    
    return fig                

fig = plotFig6()

#%% 