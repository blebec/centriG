# -*- coding: utf-8 -*-

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
