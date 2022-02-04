#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 11 09:35:00 2021

@author: cdesbois
"""

import os
from datetime import datetime
from importlib import reload
from math import ceil, floor

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
from sklearn import linear_model


import config


# nb description with pandas:
pd.options.display.max_columns = 30

# ===========================
# global setup
# NB fig size : 8.5, 11.6 or 17.6 cm

anot = True  # to draw the date and name on the bottom of the plot
std_colors = config.std_colors()
speed_colors = config.speed_colors()
plt.rcParams.update(config.rc_params())
paths = config.build_paths()
os.chdir(paths["pg"])

paths["data"] = os.path.join(paths["owncFig"], "data")


# plt.rcParams.update(
#     {'font.sans-serif': ['Arial'],
#      'font.size': 14,
#      'legend.fontsize': 'medium',
#      'figure.figsize': (11.6, 5),
#      'figure.dpi': 100,
#      'axes.labelsize': 'medium',
#      'axes.titlesize': 'medium',
#      'xtick.labelsize': 'medium',
#      'ytick.labelsize': 'medium',
#      'axes.xmargin': 0}
# )


file_name = "/Users/cdesbois/ownCloud/cgFigures/data/baudot/scatterData/scatLat.xlsx"

# filename = '/Users/cdesbois/ownCloud/cgFigures/data/baudot/Figure latence GABY + histo.xls'

# #%% all   ... and forget
# filename = '/Users/cdesbois/ownCloud/cgFigures/data/baudot/
# traitement_latencepiero_S-C_final_papier_2.xls'
# data_dict = pd.read_excel(filename, None)
# for key in data_dict:
#     print(key)

# df = data_dict['DITRIB latence min max calcul 1']
# df = data_dict['LATENCE PIERO S-C']


def load_onsets() -> pd.DataFrame:
    """
    load the excel data and format it

    Returns
    -------
    df : pandas dataframe
    """
    filename = "/Users/cdesbois/ownCloud/cgFigures/data/baudot/dataLatenceGABYhisto.xls"
    data_dict = pd.read_excel(filename, None)
    # for k in data_dict:
    #     print(k)
    df = data_dict["DITRIB latence min max calcul 1"]
    # remove empty columns and rows
    df = df.dropna(axis=1, how="all")
    df = df.drop(0)
    df = df.dropna(axis=0, how="all")
    # format stimulation formating
    df = df.rename(columns={df.columns[0]: "stim", df.columns[-1]: "repr"})
    # fill all cells with appropriate stimulation
    df.stim = df.stim.fillna(method="ffill")
    df.repr = df.repr.fillna(method="ffill")
    # format stim ['cf_para', 'cf_iso', 'cp_para', 'cp_iso']
    df.stim = df.stim.apply(lambda st: "_".join(st.lower().split(" ")[::-1]))
    df.repr = df.repr.apply(lambda st: st.lower())

    # manage columns names
    cols = [st.lower() for st in df.columns]
    tochange = {
        "correlation": "corr",
        "airsign": "sig",
        "psth": "spk",
        "moy": "vm",
        "nom": "name",
    }
    for k, v in tochange.items():
        cols = [_.replace(k, v) for _ in cols]
    cols = ["_".join(_.split(" ")) for _ in cols]

    cols[-4] = cols[-4].replace("lat_sig_", "lat_").split(".")[0]
    cols[-4] = cols[-4].replace("_s-", "_seq-").split(".")[0]
    cols[-3] = cols[-3].replace("lat_sig_", "lat_").split(".")[0]
    cols[-3] = cols[-3].replace("_s", "_seq").split(".")[0]
    df.columns = cols
    # df['lat_vm_c-p'] *= (-1) # correction to obtain relative latency
    # for col in ['moy_c-p', 'psth_seq-c']:
    #     df[col] = df[col].astype(float)
    # cleaning
    # remove moy, ...
    to_remove = [_ for _ in df.name.unique() if not _[:3].isdigit()]
    for rem in to_remove:
        df = df.drop(df[df.name == rem].index)
    df.name = df.name.apply(lambda st: st.split()[0])
    # remove duplicated columns (names)
    df = df.T.drop_duplicates().T
    # remove col 17
    df = df.drop(labels="unnamed:_17", axis=1)

    # convert dtypes
    for col in df.columns:
        try:
            df[col] = df[col].astype(float)
        except ValueError:
            pass
            # print(f"can not convert to float '{col}'")
    # limit only to required
    cols = ["stim", "name", "lat_vm_c-p", "lat_spk_seq-c"]
    df = df[cols]
    return df


def printLenOfRecording(df: pd.DataFrame):
    """
    print the content of the dataframe : names and cells

    Parameters
    ----------
    df : pd.DataFrame
        DESCRIPTION.

    Returns
    -------
    None.

    """
    names = [_ for _ in df.name.unique() if _[:3].isdigit()]
    cells = {_[:5] for _ in names}

    print(f"{'-' * 20} {len(names)} recordings for latency {'-' * 20}")
    print(names)
    print()
    print(f"{'-' * 20} {len(cells)} of unique numId {'-' * 20}")
    print(cells)


data_df = load_onsets()

printLenOfRecording(data_df)

#%%


def get_RMSE(
    inputdf: pd.DataFrame, imini: int = None, printsummary: bool = False
) -> float:
    """
    analyse the quality of the bilinear fit

    Parameters
    ----------
    inputdf : pd.DataFrame
        the data.
    imini : int, optional (default is None)
        the index value for left and right linear fit
    plot : bool, optional
        DESCRIPTION. The default is False.

    Returns
    -------
    None.

    """
    if imini is None:
        imini = 32

    datadf = inputdf.copy()
    # datadf = data_df.copy()
    xy = ["lat_vm_c-p", "lat_spk_seq-c"]
    df = datadf[xy].copy()
    # to get advance <-> positive value
    df[xy[1]] = df[xy[1]] * (-1)
    # remove outliers
    xmin = -55
    xmax = 30
    xscales = [xmin, xmax]
    ymin = -20
    ymax = +30
    df = df.dropna().sort_values(by=df.columns[0]).reset_index(drop=True)
    # left and right OLS:
    data_left = df.iloc[:imini]
    # data = df.iloc[:i].copy()
    X = data_left[xy[0]]
    Y = data_left[xy[1]]
    # X = sm.add_constant(X)  # add a constant
    X = pd.DataFrame(X).assign(Intercept=1)

    model_left = sm.OLS(Y, X).fit()
    if printsummary:
        print(f"{'-' * 20} model_left")
        print(model_left.summary())

    data_right = df.iloc[imini:]
    # data = df.iloc[:i].copy()
    X = data_right[xy[0]]
    Y = data_right[xy[1]]
    # X = sm.add_constant(X)  # add a constant
    X = pd.DataFrame(X).assign(Intercept=1)
    model_right = sm.OLS(Y, X).fit()
    if printsummary:
        print(f"{'-' * 20} model_right")
        print(model_right.summary())

    residuals = pd.concat([model_left.resid, model_right.resid])
    RMSE = np.sqrt((residuals**2).mean())
    print(f"{RMSE=:.1f}")
    return RMSE


def filter_data(df: pd.DataFrame, params):
    cols = [_ for _ in df.columns if df[_].dtypes == float]
    xscales = params.get("xscales", [df[cols[0]].min(), df[cols[0]].max()])
    # remove ouliers
    df.loc[df[cols[0]] < xscales[0]] = np.nan
    df.loc[df[cols[0]] > xscales[1]] = np.nan
    # df = df.sort_values(by=df[cols[0]]).dropna()
    # revert the axis
    df[cols[1]] = df[cols[1]] * (-1)
    # switch = minimal residual for a bilinear fit
    # CALL
    return df


def get_switch(df: pd.DataFrame, plot: bool = False) -> (float, float):
    """
    find the switch point for bilinear fit
    Parameters
    ----------
    datadf : pd.DataFrame
        dataframe containing y ans x axis.
    plot : bool, optional (default is False)
        plot the results

    Returns
    -------
    float i switch value
    float x switch value.

    """

    df = df.dropna().sort_values(by=df.columns[0]).reset_index(drop=True)
    cols = df.columns
    residuals_list = []  # sum of squared residuals
    # perform left and right linear fit
    for i in range(2, len(df) - 1):
        residuals = []
        for data in [df.iloc[:i], df.iloc[i:]]:
            X = data[cols[0]]
            Y = data[cols[1]]
            X = pd.DataFrame(X).assign(Intercept=1)
            model = sm.OLS(Y, X).fit()
            # squared left linear fit residuals
            residuals.append(np.sum(np.array(model.resid) ** 2))

        residuals_list.append(sum(residuals))
    # find minimum residuals location
    i_mini = np.argsort(residuals_list)[0]  # index value
    x_mini = df.loc[i_mini, ["lat_vm_c-p"]][0]  # x value
    print(f"for minimal residuals location {i_mini=} {x_mini= :.1f}")

    if plot:
        fig = plt.figure()
        fig.suptitle("residuals for a double linear fit")
        ax = fig.add_subplot(111)
        ax.plot(residuals_list)
        ax.axvline(i_mini)
        txt = f"i= {i_mini} \nx= {x_mini}"
        ax.text(x=i_mini + 1, y=ax.get_ylim()[1], s=txt, va="top", ha="left")
        ax.set_ylabel("squared sum of minimals")
        ax.set_xlabel("split index location")
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        fig.tight_layout()
    return i_mini, x_mini


parameters = {"xscales": [-55, 30], "yscales": [-20, 30]}
data_df2 = filter_data(data_df, parameters)

parameters["iswitch"], parameters["xswitch"] = get_switch(
    data_df.select_dtypes("number"), plot=True
)
rmse = get_RMSE(data_df, iswitch)


#%%


def plot_phaseEffect(
    inputdf: pd.DataFrame, params, corner: bool = False
) -> (plt.Figure, pd.DataFrame):
    """
    plot the vm -> time onset transfert function

    Parameters
    ----------
    inputdf : pd.DataFrame
        the data.
    corner : bool, optional (default is False)
        display in the corner
    show_residuals : bool, optional (default is False)
        plot the residuals

    Returns
    -------
    plt.figure
    pd.DataFrame

    """
    xscales = params.get("xscales")
    yscales = params.get("yscales")
    xmin = xscales[0]
    xmax = xscales[1]
    ymin = yscales[0]
    ymax = yscales[1]
    xswitch = params.get("xswitch", None)

    datadf = inputdf.copy()
    stims = datadf.stim.dropna().unique()[::-1]
    markers = {"cf_iso": "d", "cf_para": "p", "cp_iso": "P", "cp_para": "X"}
    legends = dict(
        zip(
            ["cf_para", "cf_iso", "cp_para", "cp_iso"],
            ["CF-CROSS", "CF-ISO", "CP-CROSS", "CP-ISO"],
        )
    )
    colors = [std_colors[_] for _ in "red yellow green brown".split()]

    # plotting
    fig = plt.figure(figsize=(11.6, 8))
    fig.suptitle("Impact of the Phase between FF/Horizontal Inputs")
    gs = GridSpec(4, 5)

    v0 = fig.add_subplot(gs[0, :4])  # vertical histogram/kde
    ax0 = fig.add_subplot(gs[1:, :4], sharex=v0)  # scatter plot
    h0 = fig.add_subplot(gs[1:, 4], sharey=ax0)  # horizontal histogram
    if corner:
        c0 = fig.add_subplot((gs[0, 4]), sharex=ax0, sharey=ax0)

    ax0.axhline(0, color="tab:blue", linewidth=2, alpha=0.7)
    ax0.axvline(0, color="tab:blue", linewidth=2, alpha=0.7)

    # plot regress
    cols = [_ for _ in datadf.columns if datadf[_].dtypes == float]
    numdf = datadf[cols]
    # left part
    df = numdf[numdf[numdf.columns[0]] < xswitch].dropna()
    x = df[df.columns[0]]
    y = df[df.columns[1]]
    slope1, inter1, *_ = stats.linregress(x, y)
    f1 = lambda x: slope1 * x + inter1
    # right part
    df = numdf[datadf[numdf.columns[0]] >= xswitch].dropna()
    x = df[df.columns[0]]
    y = df[df.columns[1]]
    slope2, inter2, *_ = stats.linregress(x, y)
    f2 = lambda x: slope2 * x + inter2
    # plot
    x_intersect = (inter2 - inter1) / (slope1 - slope2)
    ax0.plot(
        [xmin, x_intersect, xmax],
        [f1(xmin), f1(x_intersect), f2(xmax)],
        linewidth=10,
        color="tab:grey",
        alpha=0.3,
    )
    print(f"{'  fit  ':-^30}")
    print(f"min residual loc {xswitch}")
    print(f"slope={slope1:.2f} inter={inter1:.0f}")
    print(f"slope={slope2:.2f} inter={inter2:.0f}")
    print(f"{' scatter ':-^30}".format(" scatter "))
    statdf = pd.DataFrame()
    removed = pd.DataFrame()
    for j, stim in enumerate(stims[::-1]):
        i = len(stims) - j - 1
        # df = datadf.loc[datadf.stim == stim, cols]
        cols_and_name = cols.copy()
        cols_and_name.append("name")
        df = datadf.loc[datadf.stim == stim, cols_and_name]
        # remove noname
        if len(df.loc[df.name.isna()]) > 0:
            print(stim, " no name \n", df.loc[df.name.isna()], "\n")
            df = df.drop(df.loc[df.name.isna()].index.tolist())
        # remove outliers
        # delay (-30)
        out = df.loc[df[cols[0]] < xscales[0]].copy()
        out["stim"] = stim
        removed = removed.append(out)
        df.loc[df[cols[0]] < xscales[0]] = np.nan
        # avance ++ (55)
        out = df.loc[df[cols[0]] > xscales[1]].copy()
        out["stim"] = stim
        removed = removed.append(out)
        df.loc[df[cols[0]] > xscales[1]] = np.nan
        # res
        num = len(df)
        navm = len(df.loc[df[cols[0]].isna()])
        naspk = len(df.loc[df[cols[1]].isna()])
        bothna = len(df.loc[df[df.columns[:2]].isnull().all(1)])
        print(f"{stim}   n={num} vmNaN={navm} spkNaN={naspk} bothNaN={bothna}")
        # cuse
        df = df.dropna()
        x = df[cols[0]].values.astype(float)
        y = df[cols[1]].values.astype(float)
        # corr
        r2 = stats.pearsonr(x.flatten(), y.flatten())[0] ** 2
        lregr = stats.linregress(x, y)
        print(
            "{}   {} cells  \t r2= {:.3f} \t stdErrFit= {:.3f}".format(
                stim, len(x), r2, lregr.stderr
            )
        )
        r2 = lregr.rvalue**2
        # label = '{} {}  r2={:.2f}'.format(len(df), stim, r2)
        label = f"{legends[stim]}"
        ax0.scatter(
            x,
            y,
            color=colors[i],
            marker=markers[stim],
            # s=150,
            s=100,
            alpha=0.8,
            label=label,
            # edgecolor="w",
            edgecolor=colors[i],
        )
        # export data
        stat = df[cols].agg(["count", "mean", "std", "median", "mad", "min", "max"])
        stat.columns = [stim + "_" + _.split("_")[1] for _ in stat.columns]
        for col in stat:
            statdf[col] = stat[col]
        # ax0.scatter(x, y, color=colors[i], marker=markers[stim.split('_')[0]],
        #            s=100, alpha=0.8, label=label, edgecolor='w')
        # kde
        kde = stats.gaussian_kde(x)
        # x_kde = np.arange(floor(min(x)), ceil(max(x)), 1)
        if stim != "cf_para":
            x_kde = np.arange(xmin, xmax, 1)
            # x_kde *= -1
            v0.plot(
                x_kde, kde(x_kde), color=colors[i], alpha=1, linewidth=2, linestyle="-"
            )
            v0.fill_between(
                x_kde,
                kde(x_kde),
                0,
                color=colors[i],
                alpha=0.2,
                linewidth=2,
                linestyle="-",
            )
            qx = np.quantile(x, q=[0.25, 0.5, 0.75])
            v0.axvline(qx[1], color=colors[i], alpha=1)
            # v0.axvspan(qx[0], qx[-1], ymin=i*.3, ymax=(i+1)*.3,
            # color=colors[i], alpha=0.3)
            kde = stats.gaussian_kde(y)
            # y_kde = np.arange(floor(min(y)), ceil(max(y)), 1)
            y_kde = np.arange(ymin, ymax, 1)
            # y_kde *= -1
            h0.plot(
                kde(y_kde), y_kde, color=colors[i], alpha=1, linewidth=2, linestyle="-"
            )
            h0.fill_betweenx(y_kde, kde(y_kde), 0, color=colors[i], alpha=0.3)
            # h0.fill_between(kde(y_kde), y_kde, 0, color=colors[i],
            #         alpha=0.3)

            qy = np.quantile(y, q=[0.25, 0.5, 0.75])
            h0.axhline(qy[1], color=colors[i], alpha=1)
            if corner:
                txt = "{}c{}r".format(len(df.name.unique()), len(df))
                c0.plot(
                    [qx[1], qx[1]],
                    [qy[0], qy[-1]],
                    linewidth=3,
                    color=colors[i],
                    alpha=0.7,
                    label=txt,
                )
                c0.plot(
                    [qy[0], qx[-1]],
                    [qy[1], qy[1]],
                    linewidth=3,
                    color=colors[i],
                    alpha=0.7,
                )

        # regress:
        # x = x.reshape(len(x), 1)
        # y = y.reshape(len(x), 1)
        # regr = linear_model.LinearRegression()
        # regr.fit(x,y)
        # if r2 > 0.01:
        #     ax0.plot(x, regr.predict(x), color=colors[i], linestyle= ':',
        #              linewidth=3, alpha=0.5)
    print(f"{' removed : ':-^20}")
    print(removed)
    print(f"{' removed  ':-^20}")

    ax0.legend(loc="upper left")

    # ax.set_ylabel('spikes onset relative latency (msec)')
    ax0.set_ylabel("Spiking Latency Advance (msec)")
    ax0.set_xlabel("FF/Horizontal Input Phase (msec)")
    # ax.set_xlabel('Vm : center - surround')
    # ax0.set_xlabel('Vm relative latency (msec)')
    for spine in ["top", "right"]:
        ax0.spines[spine].set_visible(False)
    for spine in ["left", "top", "right"]:
        v0.spines[spine].set_visible(False)
    v0.set_yticks([])
    v0.set_yticklabels([])
    for spine in ["top", "right", "bottom"]:
        h0.spines[spine].set_visible(False)
    h0.set_xticks([])
    h0.set_xticklabels([])

    if corner:
        for spine in ["top", "right"]:
            c0.spines[spine].set_visible(False)
        c0.set_xticks([])
        c0.set_xticklabels([])
        c0.set_yticks([])
        c0.set_yticklabels([])
        c0.legend(
            ncol=2,
            fontsize="small",
            loc="lower right",
            labelcolor=colors[::-1],
            frameon=False,
            markerfirst=False,
        )

    ax0.set_xlim(xmin, xmax)
    v0.set_ylim(0, v0.get_ylim()[1])
    h0.set_xlim(0, h0.get_xlim()[1])
    ax0.set_ylim(-19, 28)
    ax0.set_xlim(-52, 28)

    txt = "Horizontal Lag"
    ax0.text(
        x=ax0.get_xlim()[0] / 2,
        y=ax0.get_ylim()[0] + 2.5,
        s=txt,
        color="tab:blue",
        va="top",
        ha="center",
    )
    txt = "FF"
    ax0.text(
        x=0,
        y=ax0.get_ylim()[0] + 2.5,
        s=txt,
        color="tab:blue",
        va="top",
        ha="center",
        backgroundcolor="w",
    )
    txt = "Horizontal Advance"
    ax0.text(
        x=ax0.get_xlim()[1] / 2,
        y=ax0.get_ylim()[0] + 2.5,
        s=txt,
        color="tab:blue",
        va="top",
        ha="center",
    )

    if anot:
        date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        fig.text(
            0.99,
            0.01,
            "latencies_baudot.py:plot_phaseEffect",
            ha="right",
            va="bottom",
            alpha=0.4,
        )
        fig.text(0.01, 0.01, date, ha="left", va="bottom", alpha=0.4)
        txt = "phase effect     only {} range".format(xscales)
        fig.text(0.5, 0.01, txt, ha="right", va="bottom", alpha=0.4)
    fig.tight_layout()
    statdf = statdf.T
    statdf["count"] = statdf["count"].astype("int")

    def kf(ser):
        return ser.apply(lambda st: st.split("_")[-1])

    statdf = statdf.reset_index().sort_values(by="index", key=kf)
    statdf = statdf.set_index("index")
    return fig, statdf


plt.close("all")
figure, stat_df = plot_phaseEffect(data_df2, parameters, corner=False)
print(stat_df.round(1))

save = False
if save:
    file = "phaseEffect.png"
    dirname = os.path.join(paths["owncFig"], "pythonPreview", "baudot")
    file_name = os.path.join(dirname, file)
    figure.savefig(file_name)

    name = "f9_phaseEffect"
    paths["save"] = os.path.join(paths["owncFig"], "pythonPreview", "current", "fig")
    for ext in [".png", ".pdf", ".svg"]:
        figure.savefig(os.path.join(paths["save"], (name + ext)))


def save_fig10_data_scatter(do_save: bool = False):
    """save to hdf the data used to build the figure"""

    data_savename = os.path.join(paths["figdata"], "fig10.hdf")
    # histogram
    key = "scatter"
    df = data_df.copy()
    print()
    print("=" * 20, "{}({})".format(os.path.basename(data_savename), key))
    for item in df.columns:
        print(item)
    print()
    if do_save:
        df.to_hdf(data_savename, key)


save_fig10_data_scatter(do_save=False)


#%%
def plot_onsetTransfertFunc(inputdf: pd.DataFrame) -> plt.Figure:
    """
    plot the vm -> time onset transfert function
    """
    datadf = inputdf.copy()
    cols = ["lat_vm_c-p", "lat_spk_seq-c"]
    cols = ["lat_sig_vm_s-c.1", "lat_spk_seq-c"]
    stims = datadf.stim.unique()[::-1]
    markers = {"cf": "o", "cp": "v"}
    colors = [std_colors["red"], std_colors["yellow"], std_colors["green"], "tab:brown"]
    # xscales
    xscales = [-70, 30]

    # fig = plt.figure(figsize=(8, 6))
    # # fig.suptitle('spk Vm onset-time transfert function')
    # fig.suptitle('delta latency effect (msec)')
    # ax = fig.add_subplot(111)

    # plotting
    fig = plt.figure(figsize=(14, 12))
    fig.suptitle("baudot latencies")
    gs = GridSpec(3, 3)
    # vertical histogram/kde
    v0 = fig.add_subplot(gs[0, :2])
    # v0.set_title('v0')
    # scatter plot
    ax0 = fig.add_subplot(gs[1:, :2], sharex=v0)
    # ax0.set_title('ax0')
    # horizontal histogram
    h0 = fig.add_subplot(gs[1:, 2], sharey=ax0)
    # h0.set_title('h0')

    # stims : 'cf_para', 'cf_iso', 'cp_para', 'cp_iso'
    # colors = colors[]
    for i, stim in enumerate(stims[:-1]):
        df = datadf.loc[datadf.stim == stim, cols]
        # remove outliers
        df.loc[df[cols[0]] < xscales[0]] = np.nan
        df.loc[df[cols[0]] > xscales[1]] = np.nan
        df = df.dropna()
        # x = -1 * temp[values[0]].values
        x = df[cols[0]].values.astype(float)
        y = df[cols[1]].values.astype(float)
        # corr
        # r2 = stats.pearsonr(x.flatten(),y.flatten())[0]**2
        lregr = stats.linregress(x, y)
        r2 = lregr.rvalue**2
        print("{} \t r2= {:.3f} \t stdErrFit= {:.3f}".format(stim, r2, lregr.stderr))
        label = "{} {}  r2={:.2f}".format(len(df), stim, r2)
        # label = '{} cells, {}'.format(len(df), stim)
        ax0.scatter(
            x,
            y,
            color=colors[i],
            marker=markers[stim.split("_")[0]],
            s=100,
            alpha=0.8,
            label=label,
            edgecolor="w",
        )
        # kde
        kde = stats.gaussian_kde(x)
        x_kde = np.arange(floor(min(x)), ceil(max(x)), 1)
        v0.plot(x_kde, kde(x_kde), color=colors[i], alpha=1, linewidth=2, linestyle="-")
        q = np.quantile(x, q=[0.25, 0.5, 0.75])
        v0.axvline(q[1], color=colors[i], alpha=1)
        v0.axvspan(
            q[0], q[-1], ymin=i * 0.3, ymax=(i + 1) * 0.3, color=colors[i], alpha=0.3
        )
        kde = stats.gaussian_kde(y)
        y_kde = np.arange(floor(min(y)), ceil(max(y)), 1)
        h0.plot(kde(y_kde), y_kde, color=colors[i], alpha=1, linewidth=2, linestyle="-")
        h0.axhline(q[1], color=colors[i], alpha=1)
        h0.axhspan(
            q[0], q[-1], xmin=i * 0.3, xmax=(i + 1) * 0.3, color=colors[i], alpha=0.3
        )

        # regress:
        x = x.reshape(len(x), 1)
        y = y.reshape(len(x), 1)
        regr = linear_model.LinearRegression()
        regr.fit(x, y)
        if r2 > 0.01:
            ax0.plot(
                x,
                regr.predict(x),
                color=colors[i],
                linestyle=":",
                linewidth=3,
                alpha=0.5,
            )

    # mini = min(ax.get_xlim()[0], ax.get_ylim()[0])
    # maxi = min(ax.get_xlim()[1], ax.get_ylim()[1])
    # ax.plot([maxi, mini], [mini, maxi], '-', color='tab:grey', alpha=0.5)
    ax0.legend()
    ax0.axhline(0, color="tab:blue", linewidth=2, alpha=0.8)
    ax0.axvline(0, color="tab:blue", linewidth=2, alpha=0.8)
    # ax.set_ylabel('spikes onset relative latency (msec)')
    ax0.set_ylabel("spikes : (surround + center) - center")
    # ax.set_xlabel('Vm onset relative latency (msec)')
    # ax.set_xlabel('Vm : center - surround')
    ax0.set_xlabel("Vm : (surround + center) - center")
    for spine in ["top", "right"]:
        ax0.spines[spine].set_visible(False)
    for spine in ["left", "top", "right"]:
        v0.spines[spine].set_visible(False)
    v0.set_yticks([])
    v0.set_yticklabels([])
    for spine in ["top", "right", "bottom"]:
        h0.spines[spine].set_visible(False)
    h0.set_xticks([])
    h0.set_xticklabels([])
    # ax.set_ylim(-30, 30)
    # ax.set_xlim(xscales)
    ax0.set_xlim(-35, 15)
    if anot:
        date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        fig.text(
            0.99,
            0.01,
            "latencies_baudot.py:plot_onsetTransfertFunc",
            ha="right",
            va="bottom",
            alpha=0.4,
        )
        fig.text(0.01, 0.01, date, ha="left", va="bottom", alpha=0.4)
        txt = "only {} range".format(xscales)
        fig.text(0.5, 0.01, txt, ha="right", va="bottom", alpha=0.4)
    fig.tight_layout()
    return fig


plt.close("all")
figure = plot_onsetTransfertFunc(data_df)

save = False
if save:
    file = "m_onsetTransfertFunc.png"
    dirname = os.path.join(paths["owncFig"], "pythonPreview", "baudot")
    file_name = os.path.join(dirname, file)
    figure.savefig(file_name)

#%% histogrammes

# plt.close('all')


def histo_inAndOut(
    inputdf: pd.DataFrame, removeOutliers: bool = True, onlyCouple: bool = True
) -> plt.Figure:

    datadf = inputdf.copy()

    fig, axes = plt.subplots(nrows=4, ncols=2, figsize=(8, 6), sharex=True, sharey=True)
    axes = axes.flatten(order="F")

    cols = ["moy_c-p", "psth_seq-c"]
    cols = ["lat_sig_vm_s-c.1", "lat_spk_seq-c"]
    stims = datadf.stim.unique()
    colors = ["tab:brown", std_colors["green"], std_colors["yellow"], std_colors["red"]]
    if removeOutliers:
        # xscales
        xscales = [-30, 70]
        datadf.loc[datadf[cols[0]] < xscales[0]] = np.nan
        datadf.loc[datadf[cols[0]] > xscales[1]] = np.nan
    if onlyCouple:
        datadf[cols] = datadf[cols].dropna()
    # define bins
    maxi = max(datadf[cols[0]].quantile(q=0.95), datadf[cols[1]].quantile(q=0.95))
    mini = min(datadf[cols[0]].quantile(q=0.05), datadf[cols[1]].quantile(q=0.05))
    maxi = ceil(maxi / 5) * 5
    mini = floor(mini / 5) * 5
    # plot
    for k in range(2):  # [vm, spk]
        for i, stim in enumerate(stims):
            ax = axes[i + 4 * k]
            df = datadf.loc[datadf.stim == stim, cols[k]].dropna()
            a, b, c = df.quantile([0.25, 0.5, 0.75])
            ax.axvline(b, color=colors[i], alpha=0.6, linewidth=2)
            ax.axvspan(a, c, color=colors[i], alpha=0.3)
            txt = "med= {:.0f}".format(b)
            ax.text(
                x=1,
                y=0.8,
                s=txt,
                color="tab:grey",
                fontsize="small",
                va="bottom",
                ha="right",
                transform=ax.transAxes,
            )
            txt = "{} cells".format(len(df))
            ax.text(
                x=0,
                y=0.8,
                s=txt,
                color="tab:grey",
                fontsize="small",
                va="bottom",
                ha="left",
                transform=ax.transAxes,
            )
            # histo
            height, x = np.histogram(df.values, bins=range(mini, maxi, 5), density=True)
            x = x[:-1]
            align = "edge"  # ie right edge
            # NB ax.bar, x value = lower
            ax.bar(
                x,
                height=height,
                width=5,
                align=align,
                color=colors[i],
                edgecolor="k",
                alpha=0.6,
                label="stim",
            )
            ax.axvline(0, color="tab:blue", alpha=0.7, linewidth=2)

    for ax in axes:
        ax.set_yticks([])
        for spine in ["left", "top", "right"]:
            ax.spines[spine].set_visible(False)
    axes[0].set_title("Input (Vm)")
    axes[4].set_title("Output (Spk)")
    axes[3].set_xlabel("Onset Relative Latency (msec)")
    axes[7].set_xlabel("Onset Relative Latency (msec)")

    if anot:
        date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        fig.text(
            0.99,
            0.01,
            "latencies_baudot.py:histo_inOut",
            ha="right",
            va="bottom",
            alpha=0.4,
        )
        fig.text(0.01, 0.01, date, ha="left", va="bottom", alpha=0.4)
        if removeOutliers:
            txt = "only {} range ".format(xscales)
            fig.text(0.5, 0.01, txt, ha="center", va="bottom", alpha=0.4)
    fig.tight_layout()
    return fig


plt.close("all")
figure = histo_inAndOut(data_df, removeOutliers=True, onlyCouple=True)
save = False
if save:
    file = "histo_inAndOut.png"
    dirname = os.path.join(paths["owncFig"], "pythonPreview", "baudot")
    file_name = os.path.join(dirname, file)
    figure.savefig(file_name)

#%% diff /ref
plt.close("all")


def plot_diffMean(
    inputdf: pd.DataFrame, removeOutliers: bool = True, refMean: bool = True
) -> plt.Figure:
    # datadf = data_df.copy()
    datadf = inputdf.copy()
    #    cols = ['moy_c-p', 'psth_seq-c']
    cols = ["lat_vm_c-p", "lat_spk_seq-c"]
    stims = datadf.stim.unique()
    markers = {"cf": "o", "cp": "v"}
    colors = ["tab:brown", std_colors["green"], std_colors["yellow"], std_colors["red"]]
    if removeOutliers:
        # xscales
        xscales = [-30, 70]
        datadf.loc[datadf[cols[0]] < xscales[0]] = np.nan
        datadf.loc[datadf[cols[0]] > xscales[1]] = np.nan

    fig, axes = plt.subplots(
        nrows=2, ncols=2, figsize=(12, 12), sharex=True, sharey=True
    )
    axes = axes.flatten()

    for i, stim in enumerate(stims):
        ax = axes[i]
        df = datadf.loc[datadf.stim == stim, cols].dropna().copy()
        if refMean:
            x = df.mean(axis=1)
        else:
            x = df[cols[0]]
        y = df.diff(axis=1)[cols[1]]
        # y = (df[cols[1]] - df[cols[0]])
        # quantiles
        a, b, c = x.quantile([0.25, 0.5, 0.75])
        ax.axvline(b, color=colors[i], alpha=0.6, linewidth=2)
        ax.axvspan(a, c, color=colors[i], alpha=0.3)
        a, b, c = y.quantile([0.25, 0.5, 0.75])
        ax.axhline(b, color=colors[i], alpha=0.6, linewidth=2)
        ax.axhspan(a, c, color=colors[i], alpha=0.3)
        # plot
        x = x.values
        y = y.values
        ax.scatter(
            x,
            y,
            color=colors[i],
            marker=markers[stim.split("_")[0]],
            s=100,
            alpha=0.8,
            label=stim,
            edgecolor="w",
        )
        # regress:
        x = x.reshape(len(x), 1)
        y = y.reshape(len(x), 1)
        regr = linear_model.LinearRegression()
        regr.fit(x, y)
        ax.plot(
            x, regr.predict(x), color=colors[i], linestyle=":", linewidth=3, alpha=0.7
        )
        ax.legend()
        txt = "{} cells".format(len(df))
        ax.text(
            x=0.05,
            y=0.7,
            s=txt,
            color="tab:grey",
            fontsize="small",
            va="bottom",
            ha="left",
            transform=ax.transAxes,
        )
        ax.axhline(0, color="tab:blue", linewidth=2, alpha=0.8)
        ax.axvline(0, color="tab:blue", linewidth=2, alpha=0.8)
        ax.set_ylabel("spikes - vm (msec)")
        if refMean:
            ax.set_xlabel("mean(Vm, Spk) onset relative latency (msec)")
        else:
            ax.set_xlabel("Vm onset relative latency (msec)")
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)
    ax.set_ylim(-60, 60)
    # ax.set_xlim(-25, 60)
    if anot:
        date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        fig.text(
            0.99,
            0.01,
            "latencies_baudot.py:plot_diffMean",
            ha="right",
            va="bottom",
            alpha=0.4,
        )
        fig.text(0.01, 0.01, date, ha="left", va="bottom", alpha=0.4)
    fig.tight_layout()
    return fig


plt.close("all")
ref_mean = True
figure = plot_diffMean(data_df, refMean=ref_mean)
save = False
if save:
    file = "diffMean.png"
    dirname = os.path.join(paths["owncFig"], "pythonPreview", "baudot")
    file_name = os.path.join(dirname, file)
    figure.savefig(file_name)


#%%  test with centriG cells
from load import load_data as ldat

recs = ["vm", "spk"]
amps = ["gain", "engy"]


def load_cg_vmSpk(param: str = "time") -> plt.Figure:
    """
    load latencies values for vm and spikes from centrigabor protocols
    return:
        pandas dataframe
    """
    if param not in ["time", "engy"]:
        print("param should be in [time, engy]")
        return
    # param = 'engy'
    vm_df = ldat.load_cell_contributions("vm")
    # filter
    cols = vm_df.columns
    cols = [_ for _ in cols if param in _]
    cols = [_ for _ in cols if "sect" in _]
    cols = [_ for _ in cols if "sig" not in _]
    vm_df = vm_df[cols]
    # renmae
    cols = ["_".join([_.split("sect")[0], "vm"]) for _ in cols]
    vm_df.columns = cols
    # reorder
    vm_df = vm_df.iloc[:, [3, 0, 2, 1]]

    spk_df = ldat.load_cell_contributions("spk")
    # filter
    cols = spk_df.columns
    cols = [_ for _ in cols if param in _]
    cols = [_ for _ in cols if "sect" in _]
    cols = [_ for _ in cols if "sig" not in _]
    spk_df = spk_df[cols]
    # renmae
    cols = ["_".join([_.split("sect")[0], "spk"]) for _ in cols]
    spk_df.columns = cols
    # reorder
    spk_df = spk_df.iloc[:, [3, 0, 1, 2]]

    # combine
    df = pd.concat([vm_df, spk_df], axis=1)
    df = df.dropna()

    return df


def plot_cg_onsetTransfertFunc(param: str = "time") -> plt.Figure:
    """
    plot the vm -> time onset transfert function
    """
    # datadf = cgdf.copy()
    datadf = load_cg_vmSpk(param)
    labeldict = {"time": "latency 50", "engy": "response integral"}
    # cols = ['lat_vm_c-p', 'lat_spk_seq-c']
    # cols = ['lat_sig_vm_s-c.1', 'lat_spk_seq-c']
    stims = [_.split("_")[0] for _ in datadf.columns if "vm" in _]
    markers = {"cf": "o", "cp": "v", "rd": "+"}
    colors = [std_colors["red"], std_colors["yellow"], std_colors["green"]]
    # xscales
    xscales = [-70, 30]

    # fig = plt.figure(figsize=(8, 6))
    # # fig.suptitle('spk Vm onset-time transfert function')
    # fig.suptitle('delta latency effect (msec)')
    # ax = fig.add_subplot(111)

    # plotting
    fig = plt.figure(figsize=(14, 12))
    txt = "centrigabor {} (to = center only)".format(labeldict[param])
    fig.suptitle(txt)
    gs = GridSpec(3, 3)
    # vertical histogram/kde
    v0 = fig.add_subplot(gs[0, :2])
    # v0.set_title('v0')
    # scatter plot
    ax0 = fig.add_subplot(gs[1:, :2], sharex=v0)
    # ax0.set_title('ax0')
    # horizontal histogram
    h0 = fig.add_subplot(gs[1:, 2], sharey=ax0)
    # h0.set_title('h0')

    # stims : 'cf_para', 'cf_iso', 'cp_para', 'cp_iso'
    for i, stim in enumerate(stims[1:]):
        cols = ["_".join([stim, "vm"]), "_".join([stim, "spk"])]
        if param == "time":
            df = datadf[cols] * -1
        else:
            df = datadf[cols]
        # remove outliers
        # df.loc[df[cols[0]] < xscales[0]] = np.nan
        # df.loc[df[cols[0]] > xscales[1]] = np.nan
        df.loc[df[cols[0]] < xscales[0]].apply(lambda x: np.nan)
        df.loc[df[cols[0]] > xscales[1]].apply(lambda x: np.nan)
        df = df.dropna()
        # x = -1 * temp[values[0]].values
        x = df[cols[0]].values.astype(float)
        y = df[cols[1]].values.astype(float)
        # corr
        # r2 = stats.pearsonr(x.flatten(),y.flatten())[0]**2
        lregr = stats.linregress(x, y)
        r2 = lregr.rvalue**2
        st = "_".join([stim[:2], stim[2:]])
        print("{} \t r2= {:.3f} \t stdErrFit= {:.3f}".format(st, r2, lregr.stderr))
        label = "{} {}  r2={:.2f}".format(len(df), st, r2)
        # label = '{} cells, {}'.format(len(df), stim)
        ax0.scatter(
            x,
            y,
            color=colors[i],
            marker=markers[stim[:2]],
            s=100,
            alpha=0.8,
            label=label,
            edgecolor="w",
        )
        # kde
        kde = stats.gaussian_kde(x)
        step = (max(x) - min(x)) / 100
        x_kde = np.arange(floor(min(x)), ceil(max(x)), step)
        v0.plot(x_kde, kde(x_kde), color=colors[i], alpha=1, linewidth=2, linestyle="-")
        q = np.quantile(x, q=[0.25, 0.5, 0.75])
        v0.axvline(q[1], color=colors[i], alpha=1)
        v0.axvspan(
            q[0], q[-1], ymin=i * 0.3, ymax=(i + 1) * 0.3, color=colors[i], alpha=0.3
        )
        kde = stats.gaussian_kde(y)
        step = (max(x) - min(x)) / 100
        y_kde = np.arange(floor(min(y)), ceil(max(y)), step)
        h0.plot(kde(y_kde), y_kde, color=colors[i], alpha=1, linewidth=2, linestyle="-")
        h0.axhline(q[1], color=colors[i], alpha=1)
        h0.axhspan(
            q[0], q[-1], xmin=i * 0.3, xmax=(i + 1) * 0.3, color=colors[i], alpha=0.3
        )

        # regress:
        x = x.reshape(len(x), 1)
        y = y.reshape(len(x), 1)
        regr = linear_model.LinearRegression()
        regr.fit(x, y)
        if r2 > 0.01:
            ax0.plot(
                x,
                regr.predict(x),
                color=colors[i],
                linestyle=":",
                linewidth=3,
                alpha=0.5,
            )

    # mini = min(ax.get_xlim()[0], ax.get_ylim()[0])
    # maxi = min(ax.get_xlim()[1], ax.get_ylim()[1])
    # ax.plot([maxi, mini], [mini, maxi], '-', color='tab:grey', alpha=0.5)
    # for ax in [ax0, v0, h0]:
    ax0.legend()
    ax0.axvline(0, color="tab:blue", linewidth=2, alpha=0.8)
    ax0.axhline(0, color="tab:blue", linewidth=2, alpha=0.8)
    # v0.axvline(0, color='tab:blue', linewidth=2, alpha=0.8)
    # h0.axhline(0, color='tab:blue', linewidth=2, alpha=0.8)
    # ax.set_ylabel('spikes onset relative latency (msec)')
    txt = "spikes : {} ".format(labeldict[param])
    ax0.set_ylabel(txt)
    # ax.set_xlabel('Vm onset relative latency (msec)')
    # ax.set_xlabel('Vm : center - surround')
    txt = "Vm : {} ".format(labeldict[param])
    ax0.set_xlabel(txt)
    for spine in ["top", "right"]:
        ax0.spines[spine].set_visible(False)
    for spine in ["left", "top", "right"]:
        v0.spines[spine].set_visible(False)
    v0.set_yticks([])
    v0.set_yticklabels([])
    for spine in ["top", "right", "bottom"]:
        h0.spines[spine].set_visible(False)
    h0.set_xticks([])
    h0.set_xticklabels([])
    # ax.set_ylim(-30, 30)
    # ax.set_xlim(xscales)
    # ax0.set_xlim(-35, 15)
    if anot:
        date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        fig.text(
            0.99,
            0.01,
            "latencies_baudot.py:plot_cg_onsetTransfertFunc",
            ha="right",
            va="bottom",
            alpha=0.4,
        )
        fig.text(0.01, 0.01, date, ha="left", va="bottom", alpha=0.4)
        txt = "only {} range".format(xscales)
        fig.text(0.5, 0.01, txt, ha="right", va="bottom", alpha=0.4)
    fig.tight_layout()
    return fig


plt.close("all")

# cg_df = load_cgLat_vmSpk()
figure = plot_cg_onsetTransfertFunc(param="time")

save = False
if save:
    file = "mcg_onsetTransfertFunc.png"
    dirname = os.path.join(paths["owncFig"], "pythonPreview", "baudot")
    file_name = os.path.join(dirname, file)
    figure.savefig(file_name)

#%%


def hist_diff_lat(datadf: pd.DataFrame) -> plt.Figure:
    cp_iso = datadf.loc[
        datadf.stim == "cp_iso", ["name", "lat_sig_vm_s-c.1"]
    ].set_index("name")
    cp_iso.columns = ["cp_iso_lat_s-c"]
    cp_para = datadf.loc[
        datadf.stim == "cp_para", ["name", "lat_sig_vm_s-c.1"]
    ].set_index("name")
    cp_para.columns = ["cp_para_lat_s-c"]

    cf_iso = datadf.loc[
        datadf.stim == "cf_iso", ["name", "lat_sig_vm_s-c.1"]
    ].set_index("name")
    cf_iso.columns = ["cf_iso_lat_s-c"]
    cf_para = datadf.loc[
        datadf.stim == "cf_para", ["name", "lat_sig_vm_s-c.1"]
    ].set_index("name")
    cf_para.columns = ["cf_para_lat_s-c"]

    df1 = pd.concat([cp_iso, cf_iso], axis=1).dropna()
    df1["diffe"] = df1["cp_iso_lat_s-c"] - df1["cf_iso_lat_s-c"]

    df2 = pd.concat([cp_para, cf_para], axis=1).dropna()
    df2["diffe"] = df2["cp_para_lat_s-c"] - df2["cf_para_lat_s-c"]

    # fig, axes = plt.subplots(nrows=1, ncols=2, sharex=True, sharey=True)
    # axes = axes.flatten()
    fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(8, 6), sharex=True, sharey=True)

    fig.suptitle("delta latency")
    axtitles = ["cp_iso minus cf_iso", "cp_para minus cf_para"]
    # for i, df in enumerate([df1, df2][0]):
    # ax = axes[i]
    i = 0
    df = df1
    ax = axes
    ax.set_title(axtitles[i])
    color = ["tab:red", "tab:green"][i]
    txt = "{} cells".format(len(df))
    ax.hist(df.diffe, bins=15, color=color, alpha=0.8, edgecolor="k", label=txt)
    for spine in ["left", "top", "right"]:
        ax.spines[spine].set_visible(False)
    ax.set_yticks([])
    ax.set_xlabel("(sequence-center lat) minus (sequence-center lat) (msec)")
    ax.axvline(x=0)
    ax.legend(loc="upper left")
    if anot:
        date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        fig.text(
            0.99,
            0.01,
            "latencies_baudot.py:hist_diff_lat",
            ha="right",
            va="bottom",
            alpha=0.4,
        )
        fig.text(0.01, 0.01, date, ha="left", va="bottom", alpha=0.4)
    fig.tight_layout()
    return fig


plt.close("all")
fig = hist_diff_lat(data_df)
save = False
if save:
    file = "histDiffLat.pdf"
    dirname = os.path.join(paths["owncFig"], "pythonPreview", "baudot")
    file_name = os.path.join(dirname, file)
    figure.savefig(file_name)
