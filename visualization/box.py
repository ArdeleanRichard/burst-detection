import os

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Polygon
import pandas as pd
from scipy.stats import stats


from visualization.label_map import LABEL_COLOR_MAP_SMALLER


def plot_box(data, METHODS, conditions, title='', xlabel='Methods', ylabel='Performance', outliers=False, save=False, savefile=""):
    fig, ax1 = plt.subplots(figsize=(12, 6), dpi=300)
    # fig.canvas.manager.set_window_title('A Boxplot Example')
    fig.subplots_adjust(left=0.075, right=0.95, top=0.9, bottom=0.25)

    c = 'k'
    black_dict = {
        # 'patch_artist': True,
        # 'boxprops': dict(color=c, facecolor=c),
        'capprops': dict(color=c, linewidth=2),
        'flierprops': dict(color=c, markeredgecolor=c, linewidth=2),
        'medianprops': dict(color=c, linewidth=2),
        'whiskerprops': dict(color=c, linewidth=2)
    }

    # pentru outliers, showfliers = True
    bp = ax1.boxplot(data, notch=False, sym='+', vert=True, whis=1.5, showfliers=outliers, **black_dict)
    plt.setp(bp['boxes'], color='black')
    plt.setp(bp['whiskers'], color='black')
    plt.setp(bp['fliers'], color='red', marker='+')

    # Add a horizontal grid to the plot, but make it very light in color
    # so we can use it for reading data values but not be distracting
    ax1.yaxis.grid(True, linestyle='-', which='major', color='lightgrey', alpha=0.5)

    ax1.set(
        axisbelow=True,  # Hide the grid behind plot objects
        title=f'{title}',
        xlabel=xlabel,
        ylabel=ylabel,
    )
    ax1.set_title(f'{title}', fontsize=24, fontweight='bold')
    ax1.set_xlabel(xlabel, fontsize=24)
    ax1.set_ylabel(ylabel, fontsize=24)

    # Now fill the boxes with desired colors
    num_boxes = len(data)

    for i in range(num_boxes):

        box = bp['boxes'][i]
        box_x = []
        box_y = []
        for j in range(5):
            box_x.append(box.get_xdata()[j])
            box_y.append(box.get_ydata()[j])
        box_coords = np.column_stack([box_x, box_y])

        med = bp['medians'][i]

        # Alternate among colors
        ax1.add_patch(Polygon(box_coords, facecolor=LABEL_COLOR_MAP_SMALLER[i % len(METHODS)], linewidth=5))

        ax1.plot(np.average(med.get_xdata()), np.average(data[i]), color='w', marker='*', markeredgecolor='k', markersize=6)

    # Set the axes ranges and axes labels
    ax1.set_xlim(0.5, num_boxes + 0.5)
    # top = 1.1
    # bottom = 0
    # ax1.set_ylim(bottom, top)
    ax1.set_xticklabels(np.repeat(METHODS, len(conditions)), rotation=0, fontsize=20) #, fontweight='bold')

    for axis in ['top', 'bottom', 'left', 'right']:
        ax1.spines[axis].set_linewidth(2)

    # Due to the Y-axis scale being different across samples, it can be
    # hard to compare differences in medians across the samples. Add upper
    # X-axis tick labels with the sample medians to aid in comparison
    # (just use two decimal places of precision)
    # pos = np.arange(num_boxes) + 1
    # for id, (method, y) in enumerate(zip(METHODS, np.arange(0.01, 0.03 * len(METHODS), 0.03).tolist())):
    #     fig.text(0.90, y, METHODS[id],
    #              backgroundcolor=LABEL_COLOR_MAP_SMALLER[id],
    #              color='black', weight='roman', size='x-small')

    if save==True:
        plt.savefig(savefile)
    plt.show()


