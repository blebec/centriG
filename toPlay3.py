import os

import matplotlib.pyplot as plt
import numpy as np

import centrifigs as cfig
import config


std_colors = config.std_colors()
plt.close("all")

#%% représentation -> scale \ controls \ diff

apath = "/Users/cdesbois/ownCloud/cgFigures/pythonPreview/proposal/3"

fig1 = cfig.plot_figure3(std_colors, "pop", age="old")
fig2 = cfig.plot_figure3(std_colors, "sig", age="old")

fname = os.path.join(apath, "1_iniPop.png")
# fig1.savefig(fname)
fname = os.path.join(apath, "1_iniSig.png")
# fig2.savefig(fname)

#%%
ax1 = fig1.get_axes()[0]
ax2 = fig2.get_axes()[0]
ax1.set_xlim(-50, 120)
ax2.set_xlim(-50, 120)
ax1.set_xticks(np.arange(-50, 120, 50))
ax2.set_xticks(np.arange(-50, 120, 50))

fname = os.path.join(apath, "2_timeExp_pop.png")
fig1.savefig(fname)
fname = os.path.join(apath, "2_timeExp_sig.png")
fig2.savefig(fname)

#%%
plt.close("all")

fig1 = cfig.plot_figure3(std_colors, "pop", age="old", substract=True)
fig2 = cfig.plot_figure3(std_colors, "sig", age="old", substract=True)

ax1 = fig1.get_axes()[0]
ax2 = fig2.get_axes()[0]
ax1.set_xlim(-50, 120)
ax2.set_xlim(-50, 120)
ax1.set_xticks(np.arange(-50, 120, 50))
ax2.set_xticks(np.arange(-50, 120, 50))

fname = os.path.join(apath, "3_subs_pop.png")
fig1.savefig(fname)
fname = os.path.join(apath, "3_subs_sig.png")
fig2.savefig(fname)

#%%
