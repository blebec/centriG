


from datetime import datetime
import os
from more_itertools import sort_together

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



import config
anot = True           # to draw the date and name on the bottom of the plot
std_colors = config.std_colors()
speedColors = config.speed_colors()
plt.rcParams.update(config.rc_params())
paths = config.build_paths()
os.chdir(paths['pg'])
paths['traces'] = os.path.join(paths['owncFig'], 'averageTraces')

vm_info_df = pd.read_excel(os.path.join(paths['traces'], 'neuron_props.xlsx'))
speed_info_df = pd.read_excel(
    os.path.join(paths['traces'], 'neuron_props_speed.xlsx'))

#%% to be checked:
vm_sig_cells = ['1427A_CXG4',
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

def load_all_traces(info_df, folder='vm_all'):
    stims = [item for item in os.listdir(os.path.join(paths['traces'], folder))
             if item[0] != '.']
    # remove dir
    for stim in stims:
        if not os.path.isfile(os.path.join(paths['traces'], folder, stim)):
            stims.remove(stim)
    dico = {}
    for stim in stims:
        filename = os.path.join(paths['traces'], folder, stim)
        df = pd.read_csv(filename, sep='\t', header=None)
        df.columns = info_df.Neuron.to_list()
        # center and scale
        df.index -= df.index.max()/2
        df.index /= 10
        dico[stim] = df.copy()
    return dico

def normalize(dico):
    """ divide by centerOnly"""
    #build list
    res = dico.copy()
    conds = list(dico.keys())
    if 'blank.txt' in conds:
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
        out_dico[cond] = df
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


vm_dico = load_all_traces(vm_info_df, 'vm_all')
vm_dico = normalize(vm_dico)
# vm_align_dico = align_traces(vm_dico, vm_info_df)
#plot_res(vm_dico, vm_align_dico)
vm_dico = align_traces(vm_dico, vm_info_df)

speed_dico = load_all_traces(speed_info_df, 'vm_speed')
speed_dico = normalize(speed_dico)
speed_dico = align_traces(speed_dico, speed_info_df)

#%%
#for cond in data_cond.columns:
plt.close('all')


def testPlot(dico, sig_cells):
    cond = 'cpisosec.txt'
    data_cond = dico[cond][sig_cells]
    ref = 'ctronly.txt'
    data_ref = dico[ref][sig_cells]

    fig, axes = plt.subplots(ncols=2, nrows=5, sharex=True)
    axes = axes.flatten()
    # fig.suptitle('several ways to look at')
    # recordings
    ax = axes[0]
    ax.set_title('condition')
    for col in data_cond.columns:
        ax.plot(data_cond[col], 'r-', alpha=0.8)#, label=col.split('_')[0])
    ax = axes[1]
    ax.set_title('reference')
    for col in data_cond.columns:
        ax.plot(data_ref[col], 'k-', alpha=0.6)
    # mean or med
    ax = axes[2]
    ax.set_title('mean & std')
    middle = data_cond.mean(axis=1)
    errs = data_cond.std(axis=1)
    ax.plot(middle, 'r-', alpha=0.6, label=cond)
    ax.fill_between(x=data_cond.index, y1=(middle - errs), y2=(middle + errs),
                    color='r', alpha=0.3)
    middle = data_ref.mean(axis=1)
    errs = data_ref.std(axis=1)
    ax.plot(middle, 'k-', alpha=0.6, label=cond)
    ax.fill_between(x=data_ref.index, y1=(middle - errs), y2=(middle + errs),
                    color='k', alpha=0.2)
    # med & mad
    ax = axes[3]
    ax.set_title('med & mad')
    middle = data_cond.median(axis=1)
    errs = data_cond.mad(axis=1)
    ax.plot(middle, 'r-', alpha=0.6, label=cond)
    ax.fill_between(x=data_cond.index, y1=(middle - errs), y2=(middle + errs),
                    color='r', alpha=0.3)
    middle = data_ref.median(axis=1)
    errs = data_ref.mad(axis=1)
    ax.plot(middle, 'k-', alpha=0.6, label=cond)
    ax.fill_between(x=data_ref.index, y1=(middle - errs), y2=(middle + errs),
                    color='k', alpha=0.2)
    # difference
    ax = axes[4]
    ax.set_title('difference')
    diff = data_cond.mean(axis=1) - data_ref.mean(axis=1)
    ax.plot(diff, '-g',
            linewidth=2, alpha=0.8, label='diff')
    x = diff.index.values
    ax.fill_between(x=x, y1=diff, color='g', alpha=0.3)

    #derive
    ax = axes[5]
    ax.plot(diff.diff())

    # cum sum
    ax = axes[6]
    ax.set_title('cum sum')
    ax.plot(np.cumsum(diff), label='cumsum')
    ax.plot(np.cumsum(diff -diff.mean()), label='-mean & cumsum')
    #
    ax = axes[7]


    for i, ax in enumerate(fig.get_axes()):
        ax.set_xlim(-200, 200)
        for loca in ['left', 'top', 'right', 'bottom']:
            ax.spines[loca].set_visible(False)
            ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
        lims = ax.get_xlim()
        ax.hlines(0, lims[0], lims[1], alpha=0.3)
        lims = ax.get_ylim()
        ax.vlines(0, lims[0], lims[1], alpha=0.3)

    fig.tight_layout()

testPlot(vm_dico, vm_sig_cells)
#%% no center

def extract_cell_significativity():
    filename = 'surroundonly_sector_sig.csv'
    sigcell_df = pd.read_csv(os.path.join('data', filename), sep=';')
    dico = {}
    for cond in sigcell_df.columns[1:]:
        alist = sigcell_df.loc[sigcell_df[cond] > 0, 'Neuron'].to_list()
        dico[cond] = alist
    return dico

all_cells = vm_dico['ctronly.txt'].columns.to_list()
sigcell_dico = extract_cell_significativity()

no_center_list = [item for item in vm_dico.keys() if 'cp' in item and 'woct' in item]
center_only = ['ctronly.txt']

column_list_fig6 = ['Center_Only', 'Surround_then_Center', 'Surround_Only',
                    'Static_linear_prediction']
column_list = ['center_only', 'surround_then_center', 'surround_only',
               'static_linear_prediction']

#%%
def extract_cell_data(cell, cond='rndisofull'):
    df = pd.DataFrame()
    df['center_only'] = vm_dico['ctronly.txt'][cell]
    df['surround_then_center'] = vm_dico[cond + '.txt'][cell]
    df['surround_only'] = vm_dico[cond + 'woctr.txt'][cell]
    return df

def simple_plot(cond, cell_list, df_list, lag_list, traces=True):
    """

    """
    # std_colors = {'red' : [x/256 for x in [229, 51, 51]],
    #               'green' : [x/256 for x in [127, 204, 56]],
    #               'blue' :	[x/256 for x in [0, 125, 218]],
    #               'yellow' :	[x/256 for x in [238, 181, 0]]}

    if 'cpiso' in cond:
        color = std_colors['red']
    elif 'cfiso'in cond:
        color = std_colors['green']
    elif 'cross' in cond:
        color = std_colors['yellow']
    else:
        color = std_colors['blue']

    if len(cell_list) > 12:
        nrows = 6
        ncols = 7
#        fig, axes = plt.subplots(nrows=6 , ncols=7, sharex=True)#, sharey=True)
    else:
        nrows = 3
        ncols = 4
        # fig, axes = plt.subplots(nrows=3 , ncols=4, sharex=True, sharey=True,
        #                          figsize=(16, 9))
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols,
                             sharex=True, sharey=True, figsize=(16, 9))
    axes = axes.T.flatten()
    title = 'no center & diff (' + cond + ')'
    fig.suptitle(title)
    for i, df in enumerate(df_list):
        ax = axes[i]
        ax.set_title(cell_list[i].split('_')[0])
        if traces:
            ax.plot(df.center_only, '-k', alpha=0.5, label='ref')
            ax.plot(df.surround_then_center, '-', color=color,
                    alpha=0.8, label='cp')
        ax.plot(df.surround_only, '-g', alpha=0.6, label='no_cent')
        diff = df.surround_then_center - df.center_only
        addition = False
        if addition:
            add = df.surround_then_center + df.center_only
            ax.plot(add, ':g', alpha=0.8, linewidth=2, label='add')
            ax.plot(df.surround_then_center, '-r', alpha=0.6, label='cp')

        ax.plot(diff, ':', color=color, linewidth=2, alpha=1, label='diff')
        ax.set_xlim(-50, 200)
        lims = ax.get_xlim()
        ax.hlines(0, lims[0], lims[1], alpha=0.4)
        lims = ax.get_ylim()
        ax.vlines(0, 0, lims[1], alpha=0.4)
        limx = ax.get_xlim()
        limy = ax.get_ylim()
        ax.fill_between(df.index, df.surround_only, color='g', alpha=0.3)
        ax.plot(lag_list[i], df.loc[lag_list[i], ['surround_only']], '+g',
                alpha=0.8)
        # ax.vlines(lag_list[i], limy[0], 0.5, color='g')
    for i, ax in enumerate(axes):
        for spine in ['top', 'right']:
            ax.spines[spine].set_visible(False)
        if traces:
            ax.set_ylim(-0.3, 1.5)
            custom_ticks = np.arange(0, 1.1, 0.5)
            ax.set_yticks(custom_ticks)
        else:
            ax.set_ylim(-0.3, 0.9)
            custom_ticks = np.linspace(0, 0.5, 2)
            ax.set_yticks(custom_ticks)

        # ax.legend()
        if i > nrows - 1:
            ax.spines['left'].set_visible(False)
            ax.yaxis.set_visible(False)
    fig.tight_layout()
    if anot:
        date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        fig.text(0.99, 0.01, 'vmPlot.py:simple_plot',
                 ha='right', va='bottom', alpha=0.4)
        fig.text(0.01, 0.01, date, ha='left', va='bottom', alpha=0.4)
    return fig


plt.close('all')

cell = all_cells[0]
cell_df = extract_cell_data(cell)
#align_center(cell_df)


def extract_data(cells, cond='cpisosec', sort_by='lag'):
    dfs = []
    lags = []
    amps = []
    for cell in cells:
        df = extract_cell_data(cell, cond)
        dfs.append(df)
        lags.append(df.loc[0:120, ['surround_only']].idxmax()[0])
        amps.append(df.loc[0:120, ['surround_only']].max()[0])
    if sort_by == 'amp':
        # sort by amp
        s_amps, s_lags, s_cells, s_dfs = sort_together([amps, lags, cells, dfs],
                                                       reverse=True)
    else:
        # sort by lag
        s_lags, s_cells, s_dfs = sort_together([lags, cells, dfs])
    return cond, s_cells, s_dfs, s_lags
#
# =============================================================================
# dfs = []
# lags = []
# amps = []
# cond = 'cpisosec'
#
# #for cell in cells:
# cells = sigcell_dico[cond]
# for cell in cells:
#     df = extract_cell_data(cell)
#     dfs.append(df)
#     lags.append(df.loc[0:120, ['surround_only']].idxmax()[0])
#     amps.append(df.loc[0:120, ['surround_only']].max()[0])
#
# =============================================================================
# cond='cpisosec'
# cells = sigcell_dico[cond]
# _, s_cells, s_dfs, s_lags = extract_data(cells, cond='cpisosec', sort_by='lag')

# fig = simple_plot(cond, s_cells, s_dfs, s_lags, traces=False)

for cond in sigcell_dico.keys():
    cells = sigcell_dico[cond]
    # cells = all_cells
    _, s_cells, s_dfs, s_lags = extract_data(cells, cond=cond, sort_by='lag')
    fig = simple_plot(cond, s_cells, s_dfs, s_lags, traces=True)
    # filename =  os.path.join(paths['cgFig'], 'pythonPreview',
    #                           'fillingIn', 'sig', cond + '.png')
    # fig.savefig(filename, format='png')



# max location of no center:
#     df.loc[-50:80, ['surround_only']].idxmax()
