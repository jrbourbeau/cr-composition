#!/usr/bin/env python

# Detector livetime calculation
#
# For a reference see the IT-73 spectrum livetime reference at
# https://wiki.icecube.wisc.edu/index.php/IceTop-73_Spectrum_Analysis#Live-time_calculation

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import datetime
import dask
from dask import delayed, multiprocessing
from dask.diagnostics import ProgressBar

import comptools as comp
import comptools.analysis.plotting as plotting


def livetime_fit_func(t, I0, T):
    return I0 * np.exp(-t/T)


@delayed
def get_livetime_counts_and_fit(config, month, time_bins):

    # Get time difference histogram counts from level3 pickle files
    counts = comp.datafunctions.get_level3_livetime_hist(config, month)
    # Fit decaying exponential to histogram
    time_midpoints = (time_bins[1:] + time_bins[:-1]) / 2
    time_mask = time_midpoints < 1.4
    popt, pcov = curve_fit(livetime_fit_func, time_midpoints[time_mask], counts[time_mask],
                           sigma=np.sqrt(counts[time_mask]), p0=[1e5, 1e-2])
    I0_fit, T_fit = popt
    I0_fit_err, T_fit_err = np.sqrt(np.diag(pcov))
    # Get livetime from fit parameters
    livetime = T_fit * np.sum(counts)
    livetime_err = T_fit_err * np.sum(counts)

    data_dict = {'month': month, 'counts': counts,
                 'livetime': livetime, 'livetime_err': livetime_err,
                 'I0_fit': I0_fit, 'I0_fit_err': I0_fit_err,
                 'T_fit': T_fit, 'T_fit_err': T_fit_err}

    month_str = datetime.date(2000, month, 1).strftime('%B')

    return data_dict


def save_livetime_plot(df, config, time_bins):
    fig, axarr = plt.subplots(3, 4, figsize=(10,8), sharex=True, sharey=True)
    for month, ax in zip(df.index, axarr.flatten()):
        row = df.loc[month]
        counts = row['counts']
        I0_fit = row['I0_fit']
        T_fit = row['T_fit']
        livetime = row['livetime']
        livetime_err = row['livetime_err']
        livetime_str = 'Livetime [s]:\n{:0.2e} +/- {:0.1f}'.format(livetime, livetime_err)

        # Plot time difference histogram and corresponding fit
        plotting.plot_steps(time_bins, counts, ax=ax)
        time_midpoints = (time_bins[1:] + time_bins[:-1]) / 2
        ax.plot(time_midpoints, livetime_fit_func(time_midpoints, I0_fit, T_fit),
                marker='None', ls='-', c='C1')
        month_str = datetime.date(2000, month, 1).strftime('%B')
        ax.set_title(month_str)
        ax.set_xlim((0, 2))
        ax.set_yscale('log', nonposy='clip')
        ax.text(0.6, 2.5e5, livetime_str, fontsize=10)
        ax.grid()

    fig.text(0.5, 0, 'Time between events [s]', ha='center', fontsize=16)
    fig.text(0, 0.5, 'Counts', va='center', rotation='vertical', fontsize=16)
    plt.tight_layout()
    outfile = os.path.join(comp.paths.figures_dir, 'livetime',
                           'livetime-array-{}.png'.format(config))
    comp.check_output_dir(outfile)
    plt.savefig(outfile)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description='Calculates and saves detector livetime plot')
    parser.add_argument('-c', '--config', dest='config', nargs='*',
                   choices=comp.datafunctions.get_data_configs(),
                   help='Detector configuration')
    args = parser.parse_args()

    time_bins = np.linspace(0, 2, 101)

    for config in args.config:

        results = [get_livetime_counts_and_fit(config, month, time_bins)
                   for month in range(1, 13)]
        df = delayed(pd.DataFrame)(results)
        with ProgressBar():
            print('Computing the monthly livetimes for {}'.format(config))
            df = df.compute(get=multiprocessing.get, num_workers=len(results))

        df.set_index('month', inplace=True)

        # Save corresponding monthly livetime plot
        save_livetime_plot(df, config, time_bins)

        # Save livetime dataframe for later use
        full_livetime = df['livetime'].sum()
        full_livetime_err = np.sqrt(np.sum([err**2 for err in df['livetime_err']]))

        data_dict = {'livetime(s)': full_livetime, 'livetime_err(s)': full_livetime_err}
        for month in df.index:
            month_str = datetime.date(2000, month, 1).strftime('%B')
            data_dict[month_str + '_livetime(s)'] = df.loc[month]['livetime']
            data_dict[month_str + '_livetime_err(s)'] = df.loc[month]['livetime_err']

        livetime_file = comp.get_livetime_file()
        try:
            livetime_df = pd.read_csv(livetime_file, index_col=0)
            livetime_df.loc[config] = data_dict
        except IOError:
            livetime_df = pd.DataFrame(data_dict, index=[config])
        livetime_df.to_csv(livetime_file)
