

import os
os.chdir('/Users/cdesbois/pg/chrisPg/centriG')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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
alpha = [0.5, 1, 0.8, 0.8, 1]

df = pd.read_excel('figOpt.xlsx')
df.set_index('time', inplace=True)

#%%
plt.close('all')

fig, axes = plt.subplots(nrows=5, ncols=1, sharex='all', sharey='all')
for i, ax in enumerate(axes.flatten()):
    ax.plot(df.index, df[df.columns[i]], color= colors[i])

for ax in fig.get_axes():
    for loca in ['top', 'right']:
        ax.spines[loca].set_visible(False)
        
#%%
plt.close('all')
        
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
ax1.set
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


leg = ax1.legend(loc='upper right', markerscale=None, 
             handlelength=0, framealpha=1)
for line,text in zip(leg.get_lines(), leg.get_texts()):
    text.set_color(line.get_color())   

fig.tight_layout()