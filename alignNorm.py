import os
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# import centriG.config as config
# import centriG.load.load_data as ldat
import config
import load.load_data as ldat

anot = True  # to draw the date and name on the bottom of the plot
std_colors = config.std_colors()
speedColors = config.speed_colors()
plt.rcParams.update(config.rc_params())
paths = config.build_paths()
os.chdir(paths.get("pg"))

#%%
paths["data"] = os.path.expanduser("~/pg/chrisPg/centriG/data/data_to_use")


def load_traces(paths, kind="vm", spread="sect", num=2):
    if kind == "vm" and spread == "sect":
        files = ["vmSectRaw.xlsx", "vmSectNorm.xlsx", "vmSectNormAlig.xlsx"]
    elif kind == "vm" and spread == "full":
        files = ["vmFullRaw.xlsx", "vmFullNorm.xlsx", "vmFullNormAlig.xlsx"]
    elif kind == "spk" and spread == "sect":
        files = ["spkSectRaw.xlsx", "spkSectNorm.xlsx", "spkSectNormAlig.xlsx"]
    elif kind == "spk" and spread == "full":
        files = ["spkFullRaw.xlsx", "spkFullNorm.xlsx", "spkFullNormAlig.xlsx"]
    else:
        print("load_traces: kind should be updated")
    file = files[num]
    filename = os.path.join(paths["data"], file)
    df = pd.read_excel(filename)
    # time
    middle = (df.index.max() - df.index.min()) / 2
    df.index = df.index - middle
    df.index = df.index / 10

    label = file.split(".")[0]

    # rename column
    cols = df.columns
    cols = [item[:-3] + ("_").join(item[-3:].split("n")) for item in cols]
    df.columns = cols

    nb_cells = list(set([item.split("_")[1] for item in df.columns]))

    return label, df


def plot_align_normalize(label, data, substract=False):
    """
    """

    def select_pop(df, filt="pop"):
        cols = df.columns
        if filt == "pop":
            pop = [item for item in cols if "n15" in item]
            df = df[pop].copy()
            df.columns = [item.replace("n15", "") for item in pop]
        elif filt == "spk":
            spks = [item for item in cols if "n6" in item]
            df = df[spks].copy()
            df.columns = [item.replace("n6", "") for item in spks]
        elif filt == "spk2s":
            spk2s = [item for item in cols if "n5" in item]
            df = df[spk2s].copy()
            df.columns = [item.replace("n5", "") for item in spk2s]
        else:
            return
        return df

    # colors = ['k', std_colors['red'], std_colors['green'],
    #           std_colors['yellow'], std_colors['blue'],
    #           std_colors['blue']]
    colors = [std_colors[color] for color in 
              "red green yellow blue blue".split()]
    colors.insert(0, (0, 0, 0))  # black as first color
    alphas = [0.8, 1, 0.8, 0.8, 0.8, 0.8]

    cell_pop = list(set([item.split("_")[1] for item in data.columns]))
    cell_pop = sorted([int(item) for item in cell_pop])[::-1]

    fig, axes = plt.subplots(
        ncols=1, nrows=len(cell_pop), figsize=(6, 18), sharex=True, sharey=True
    )
    # fig.suptitle(label, alpha=0.4)
    fig.text(x=0.05, y=0.95, s=label, alpha=0.6)
    # for i, k in enumerate(['pop', 'spk', 'spk2s']):
    for i, k in enumerate(cell_pop):
        ax = axes[i]
        ax.set_title(str(k) + " sig_cells", alpha=0.6)
        # ax.set_title('pop = ' + k + ' _sig', alpha=0.6)
        cols = [item for item in data.columns if str(k) == item.split("_")[-1]]
        df = data[cols]
        #        df = select_pop(data, filt=k)
        # remove 'rndisosect'
        #       cols = df.columns
        cols = [item for item in cols if "rndisosect" not in item]
        if substract:
            # subtract the centerOnly response
            for col in df.columns:
                if "center" in col:
                    ref = df[col]
                    df = df.subtract(ref, axis=0)
        for j, col in enumerate(cols):
            ax.plot(
                df.loc[-20:120, [col]],
                color=colors[j],
                alpha=alphas[j],
                label=col,
                linewidth=2,
            )
        # overlay of cpiso
        for j, col in enumerate(cols):
            if "cpiso" in col:
                ax.plot(
                    df.loc[-20:120, [col]],
                    color=colors[j],
                    alpha=alphas[j],
                    label=col,
                    linewidth=3,
                )

        for loc in ["top", "right"]:
            ax.spines[loc].set_visible(False)
        if "spk" in label:
            ax.set_ylabel("spk")
        else:
            ax.set_ylabel("vm")
        if i == 2:
            ax.set_xlabel("relative time (ms)")
    #    ax.set_xlim(-20, 120)
    for ax in axes:
        lims = ax.get_ylim()
        ax.vlines(0, lims[0], lims[1], alpha=0.3)
        ax.set_ylim(lims)
        lims = ax.get_xlim()
        ax.hlines(0, lims[0], lims[1], alpha=0.3)

    fig.tight_layout()

    if anot:
        date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        fig.text(
            0.99,
            0.01,
            "alignNorm.py:plot_align_normalize",
            ha="right",
            va="bottom",
            alpha=0.4,
        )
        fig.text(0.01, 0.01, date, ha="left", va="bottom", alpha=0.4)
    return fig


# savepath = "/Users/cdesbois/ownCloud/cgFigures/pythonPreview/proposal/alignNorm"
savepath = os.path.join(paths.get('owncFig'), 
                        'pythonPreview', 'old', 'proposal', '4_alignNorm')
save = False
#%%
plt.close("all")
kind = ["vm", "spk"][0]
spread = ["sect", "full"][1]
num = [0, 1, 2][0]

label, data = load_traces(paths, kind=kind, spread=spread, num=num)
fig1 = plot_align_normalize(label, data, substract=False)
fig2 = plot_align_normalize(label, data, substract=True)

#%%
plt.close("all")
for kind in ["vm", "spk"]:
    for spread in ["sect", "full"]:
        for num in range(3):
            label, data = load_traces(paths, kind=kind, spread=spread, num=num)
            fig1 = plot_align_normalize(label, data)
            fig2 = plot_align_normalize(label, data, substract=True)
            if save:
                fig1.savefig(fname=os.path.join(savepath, label + ".png"))
                fig2.savefig(fname=os.path.join(savepath, label + "Subs.png"))

# TODO to be checked

#%%
