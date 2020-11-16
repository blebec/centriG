# -*- coding: utf-8 -*-

import inspect


# general functions
def retrieve_name(var):
    """
    to retrieve the string value of a variable
    """
    callers_local_vars = inspect.currentframe().f_back.f_locals.items()
    return [var_name for var_name, var_val in callers_local_vars if var_val is var]


# adjust the y scale to allign plot for a value (use zero here)

#alignement to be performed
#see https://stackoverflow.com/questions/10481990/
#matplotlib-axis-with-two-scales-shared-origin/10482477#10482477

def align_yaxis(ax1, v1, ax2, v2):
    """adjust ax2 ylimit so that v2 in ax2 is aligned to v1 in ax1"""
    _, y1 = ax1.transData.transform((0, v1))
    _, y2 = ax2.transData.transform((0, v2))
    inv = ax2.transData.inverted()
    _, dy = inv.transform((0, 0)) - inv.transform((0, y1-y2))
    miny, maxy = ax2.get_ylim()
    ax2.set_ylim(miny+dy, maxy+dy)


def change_plot_trace_amplitude(ax, gain=1):
    """change the amplitude of the plot,
    doesn't change the zero location """
    lims = ax.get_ylim()
    new_lims = (lims[0]/gain, lims[1]/gain)
    ax.set_ylim(new_lims)


def properties(ax):
    """
    print size and attributes of an axe
    """
    size = ax.axes.xaxis.label.get_size()
    fontname = ax.axes.xaxis.label.get_fontname()
    print('xaxis:', fontname, size)
    size = ax.axes.yaxis.label.get_size()
    fontname = ax.axes.yaxis.label.get_fontname()
    print('yaxis:', fontname, size)


def fig_properties(afig):
    """
    exoplore figure properties
    """
    for ax in afig.get_axes():
        properties(ax)


def inch_to_cm(value):
    return value/2.54


def axis_data_coords_sys_transform(axis_obj_in,xin,yin,inverse=False):
    """ inverse = False : Axis => Data
                = True  : Data => Axis
    """
    xlim = axis_obj_in.get_xlim()
    ylim = axis_obj_in.get_ylim()

    xdelta = xlim[1] - xlim[0]
    ydelta = ylim[1] - ylim[0]
    if not inverse:
        xout =  xlim[0] + xin * xdelta
        yout =  ylim[0] + yin * ydelta
    else:
        xdelta2 = xin - xlim[0]
        ydelta2 = yin - ylim[0]
        xout = xdelta2 / xdelta
        yout = ydelta2 / ydelta
    return xout,yout


def new_columns_names(cols):
    """
    change the columns names -> snake_case explicit
    input : list of column names
    output: list of column names
    """
    def convert_to_snake(camel_str):
        """ camel case to snake case """
        temp_list = []
        for letter in camel_str:
            if letter.islower():
                temp_list.append(letter)
            elif letter.isdigit():
                temp_list.append(letter)
            else:
                temp_list.append('_')
                temp_list.append(letter)
        result = "".join(temp_list)
        return result.lower()
    newcols = [convert_to_snake(item) for item in cols]
    # nb added an '_i' to cross to have the same lengt in all stim names
    chg_dct = dict(vms = 'vm_sect_',
                   vmf = 'vm_full_',
                   spks = 'spk_sect_',
                   spkf = 'spk_full_',
                   dlat50 = 'time50',
                   dgain50 = 'gain50',
                   lat50 = 'time50',
                   cp =  'cp',
                   cf = 'cf',
                   rnd = 'rd',
                   cross = 'cx',
                   denergy = 'engy')
    for key in chg_dct:
        newcols = [item.replace(key, chg_dct[key]) for item in newcols]
    return newcols
