import os
from datetime import datetime

import matplotlib.pyplot as plt

from marvl.plotters import WQFluxPlotter

base_path = 'shared-datalake/csiem-data-warehouse/wwmsp/model/V1.6.0/'
inputs = [
    '2013B/csiem_B009_20121101_20131231_WQ_FLUX.csv',
    '2015A/csiem_A001_20141101_20151231_WQ_FLUX.csv',
    '2020A/csiem_A001_20191101_20201231_WQ_FLUX.csv',
    '2021A/csiem_A001_20201101_20211231_WQ_FLUX.csv',
    '2022A/csiem_A001_20211101_20221231_WQ_FLUX.csv',
    '2023B_davy/csiem_B009_20221101_20240401_WQ_FLUX.csv'

]

for input_file in inputs:
    input_path = os.path.join(base_path, input_file)
    out_fname = f'{os.path.basename(input_file).split('.')[0]}.png'
    year = os.path.dirname(input_file)[0:4]

    plotter = WQFluxPlotter(input_path, resample_rule='W')

    fig, ax = plt.subplots(3, 1, figsize=(11, 13))
    plt.subplots_adjust(hspace=0.3)

    plotter.plot_load(ax[0], 'tn', 4, param_dict={'label': 'TN'})
    ax[0].set_xlim(datetime(int(year), 1, 1), datetime(int(year), 12, 31))
    if year in ['2015', '2020', '2023']:
        ax[0].set_ylim(0, 10)
    else:
        ax[0].set_ylim(0, 300)
    plotter.plot_split_fill(ax[0], 'Imports', 'Exports')
    total = plotter.processor.get_net_load('tn', 4, False).sum()
    ax[0].set_title(
        f'(a) {year} TN load through Fremantle (Total: {total:.2f} tonne)'
    )
    ax[0].legend(fontsize=9)

    plotter.plot_load(ax[1], 'no3', 4, param_dict={'label': 'NO$_3$'})
    plotter.plot_load(ax[1], 'nh4', 4, param_dict={'color': 'white', 'label': 'NH$_4$'})
    ax[1].set_xlim(datetime(int(year), 1, 1), datetime(int(year), 12, 31))
    plotter.plot_split_fill(ax[1], 'Imports', 'Exports')
    total = plotter.processor.get_net_load('dn', 4, False).sum()
    ax[1].set_title(
        f'(b) {year} DIN (NO$_3$ + NH$_4$) load through Fremantle (Total: {total:.2f} tonne)'
    )
    ax[1].legend(fontsize=9)

    plotter.plot_est_load(ax[2], 'tn', 5, 4, param_dict={'label': 'TN via CS-North'})
    plotter.plot_est_load(ax[2], 'tn', 11, 4, param_dict={'color': 'white', 'label': 'TN via CS-South'})

    ax2b = ax[2].twinx()
    plotter.plot_tracer_ratio(ax2b, 'tn', 5, 4, param_dict={'label': 'TN via CS-North ratio'})
    plotter.plot_tracer_ratio(ax2b, 'tn', 11, 4, param_dict={'linestyle': 'dotted', 'label': 'TN via CS-South ratio'})
    ax[2].set_xlim(datetime(int(year), 1, 1), datetime(int(year), 12, 31))
    plotter.plot_split_fill(ax[2], 'Imports', 'Exports')
    total = plotter.processor.get_net_est_load('tn', [5, 11], [4, 4]).sum()
    ax[2].set_title(
        f'(c) {year} Estimated TN load entering Cockburn Sound from the Swan-Canning (Total: {total:.2f} tonne)'
    )
    handles1, labels1 = ax[2].get_legend_handles_labels()
    handles2, labels2 = ax2b.get_legend_handles_labels()
    all_handles = handles1 + handles2
    all_labels = labels1 + labels2
    ax[2].legend(all_handles, all_labels, fontsize=9)

    plotter.plot_watermark(fig)
    fig.savefig(f'dev-plots/freo_{year}_{plotter.model_version}_{out_fname}', dpi=400)

    ax[0].clear()
    ax[1].clear()
    ax[2].clear()
    ax2b.clear()
    ax2b.remove()

    plotter.plot_discharge(ax[0], 5)
    plotter.plot_mean_discharge(ax[0], 5, 1, {'linestyle': 'dotted'})
    plotter.plot_mean_discharge(ax[0], 5)
    plotter.plot_mean_discharge(ax[0], 5, -1, {'linestyle': 'dashdot'})
    plotter.plot_split_fill(ax[0], 'Southward', 'Northward')
    ax[0].set_title('(a) Mean water discharge through Cockburn Sound (NS5)')
    ax[0].legend(fontsize=8)

    plotter.plot_discharge(ax[1], 10)
    plotter.plot_mean_discharge(ax[1], 10, 1, {'linestyle': 'dotted'})
    plotter.plot_mean_discharge(ax[1], 10)
    plotter.plot_mean_discharge(ax[1], 10, -1, {'linestyle': 'dashdot'})
    plotter.plot_split_fill(ax[1], 'Eastward', 'Westward')
    ax[1].set_title('(b) Mean water discharge through Cockburn Sound (NS10)')
    ax[1].legend(fontsize=8)

    plotter.plot_load(ax[2], 'tn', 10, True)
    plotter.plot_split_fill(ax[2], 'Imports', 'Exports')
    total = plotter.processor.get_net_load('tn', 10, True).sum()
    ax[2].set_title(f'(a) {year} TN load to Kwinana Shelf (Total: {total:.2f} tonne)')
    ax[2].legend()

    plotter.plot_watermark(fig)
    fig.savefig(f'dev-plots/shelf_{year}_{plotter.model_version}_{out_fname}', dpi=400)

    ax[0].clear()
    ax[1].clear()
    ax[2].clear()

    plotter.plot_load(ax[0], 'tn', 5)
    plotter.plot_load(ax[0], 'tn', 11, param_dict={'color': 'white'})
    plotter.plot_net_load(ax[0], 'tn', [5, 11], [False, False])
    plotter.plot_split_fill(ax[0], 'Imports', 'Exports')
    total = plotter.processor.get_net_load('tn', [5, 11], [False, False]).sum()
    ax[0].set_title(f'(a) {year} TN load through Cockburn Sound (Total: {total:.2f} tonne)')
    ax[0].legend(fontsize=9)

    plotter.plot_load(ax[1], 'no3', 5)
    plotter.plot_load(ax[1], 'no3', 11, invert=True, param_dict={'color': 'white'})
    plotter.plot_net_load(ax[1], 'no3', [5, 11], [False, True])
    plotter.plot_split_fill(ax[1], 'Imports', 'Exports')
    total = plotter.processor.get_net_load('no3', [5, 11], [False, True]).sum()
    ax[1].set_title(f'(b) {year} NO$_3$ load through Cockburn Sound (Total: {total:.2f} tonne)')
    ax[1].legend(fontsize=9)

    plotter.plot_load(ax[2], 'nh4', 5)
    plotter.plot_load(ax[2], 'nh4', 11, invert=True, param_dict={'color': 'white'})
    plotter.plot_net_load(ax[2], 'nh4', [5, 11], [False, True])
    plotter.plot_split_fill(ax[2], 'Imports', 'Exports')
    total = plotter.processor.get_net_load('nh4', [5, 11], [False, True]).sum()
    ax[2].set_title(f'(c) 2021 NH$_4$ load through Cockburn Sound (Total: {total:.2f} tonne)')
    ax[2].legend(fontsize=9)

    plotter.plot_watermark(fig)
    fig.savefig(f'dev-plots/tn_{year}_{plotter.model_version}_{out_fname}', dpi=400)

    ax[0].clear()
    ax[1].clear()
    ax[2].clear()

    plotter.plot_discharge(ax[0], 5)
    plotter.plot_mean_discharge(ax[0], 5, 1, {'linestyle': 'dotted'})
    plotter.plot_mean_discharge(ax[0], 5)
    plotter.plot_mean_discharge(ax[0], 5, -1, {'linestyle': 'dashdot'})
    plotter.plot_split_fill(ax[0], 'Southward', 'Northward')
    ax[0].set_title('(a) Mean water discharge through Cockburn Sound (NS5)')
    ax[0].legend(fontsize=8)

    plotter.processor._conv = (1800 *1. / (1000000))
    plotter.plot_load(ax[1], 'trc', 5)
    plotter.plot_load(ax[1], 'trc', 11, invert=True, param_dict={'color': 'white'})
    plotter.plot_net_load(ax[1], 'trc', [5, 11], [False, True])
    plotter.plot_split_fill(ax[1], 'Imports', 'Exports')
    total = plotter.processor.get_net_load('trc', [5, 11], [False, True]).sum()
    ax[1].set_title(f'(b) {year} River tracer load through Cockburn Sound (Total: {total:.2f} tonne)')
    ax[1].legend(fontsize=9)

    plotter.plot_load(ax[2], 'trc_dis', 5)
    plotter.plot_load(ax[2], 'trc_dis', 11, invert=False, param_dict={'color': 'white'})
    plotter.plot_net_load(ax[2], 'trc_dis', [5, 11], [False, False])
    plotter.plot_split_fill(ax[2], 'Imports', 'Exports')
    total = plotter.processor.get_net_load('trc_dis', [5, 11], [False, False]).sum()
    ax[2].set_title(f'(c) {year} Discharge tracer load through Cockburn Sound (Total: {total:.2f} tonne)')
    ax[2].legend(fontsize=9)

    plotter.plot_watermark(fig)
    fig.savefig(f'dev-plots/tracer_{year}_{plotter.model_version}_{out_fname}', dpi=400)
