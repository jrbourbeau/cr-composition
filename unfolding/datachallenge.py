#!/usr/bin/env python

from __future__ import division
import os
import argparse
import itertools
from functools import partial
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from dask import delayed, get, multiprocessing
from dask.diagnostics import ProgressBar
import warnings

import comptools as comp
from run_unfolding import unfold

warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")


def get_test_counts(case, composition, num_groups, energy_midpoints,
                    flux_to_counts_scaling):
    log_energy_midpoints = np.log10(energy_midpoints)

    scale = 1.0 / num_groups
    if case == 'constant':
        counts = np.array([1000]*len(log_energy_midpoints))
    elif case == 'simple_power_law':
        comp_flux = comp.broken_power_law_flux(energy_midpoints,
                                               gamma_before=-3.0,
                                               energy_break=3e12)
        comp_flux = scale * comp_flux
        counts = flux_to_counts_scaling * comp_flux
    elif case == 'broken_power_law_0':
        comp_flux = comp.broken_power_law_flux(energy_midpoints,
                                               energy_break=10**7.0)
        comp_flux = scale * comp_flux
        counts = flux_to_counts_scaling * comp_flux
    elif case in ['broken_power_law_1', 'broken_power_law_2']:
        if composition in ['PPlus', 'He4Nucleus', 'light']:
            gamma_after = -3.0 if case == 'broken_power_law_1' else -4.0
        else:
            gamma_after = -4.0 if case == 'broken_power_law_1' else -3.0
        comp_flux = comp.broken_power_law_flux(energy_midpoints,
                                               energy_break=10**7.0,
                                               gamma_before=-2.7,
                                               gamma_after=gamma_after)
        comp_flux = scale * comp_flux
        counts = flux_to_counts_scaling * comp_flux
    elif case in ['H4a', 'H3a']:
        model_flux = comp.model_flux(model=case,
                                     energy=energy_midpoints,
                                     num_groups=num_groups)
        counts = model_flux['flux_{}'.format(composition)].values * flux_to_counts_scaling

    else:
        raise ValueError('Invalid case, "{}", entered'.format(case))

    return counts


def calculate_ratio(flux, flux_err_stat, flux_err_sys,
                    true_flux, true_flux_err_stat, true_flux_err_sys):

    diff = flux - true_flux
    # Error bar calculation
    diff_err_sys = np.sqrt(flux_err_sys**2 + true_flux_err_sys**2)
    diff_err_stat = np.sqrt(flux_err_stat**2 + true_flux_err_stat**2)

    frac_diff, frac_diff_sys = comp.ratio_error(num=diff,
                                                num_err=diff_err_sys,
                                                den=true_flux,
                                                den_err=true_flux_err_sys)
    frac_diff, frac_diff_stat = comp.ratio_error(num=diff,
                                                 num_err=diff_err_stat,
                                                 den=true_flux,
                                                 den_err=true_flux_err_stat)

    return frac_diff, frac_diff_stat, frac_diff_sys


def main(config, num_groups, prior, ts_stopping, case, p=None):

    figures_dir = os.path.join(comp.paths.figures_dir, 'unfolding', config,
                               'datachallenge', '{}_case'.format(case),
                               '{}_prior'.format(prior),
                               'ts_stopping_{}'.format(ts_stopping))

    # Calculate desired counts distribution for test case
    counts_true = pd.DataFrame(index=range(num_ebins),
                               columns=comp_list)
    for composition in comp_list:
        flux_to_counts_scaling = eff_area[composition] * livetime * solid_angle * energybins.energy_bin_widths
        counts_true[composition] = get_test_counts(case,
                                                   composition,
                                                   num_groups,
                                                   energybins.energy_midpoints,
                                                   flux_to_counts_scaling)
    counts_true['total'] = counts_true.sum(axis=1)

    # Plot true flux and H4a flux (as a visual reference)
    fig, axarr = plt.subplots(nrows=1, ncols=num_groups + 1,
                              sharex=True, sharey=True, figsize=(15, 5))
    for idx, composition in enumerate(comp_list + ['total']):
        ax = axarr[idx]
        model_flux = comp.model_flux(model='H4a',
                                     energy=energybins.energy_midpoints,
                                     num_groups=num_groups)
        model_comp_flux = model_flux['flux_{}'.format(composition)].values
        ax.plot(energybins.log_energy_midpoints,
                energybins.energy_midpoints**2.7 * model_comp_flux,
                color=color_dict[composition],
                ls='-.',
                lw=2,
                marker='None',
                label='H4a')
        comp_flux, _ = counts_to_flux(counts_true[composition],
                                      composition=composition)

        ax.plot(energybins.log_energy_midpoints, comp_flux,
                color=color_dict[composition],
                ls='-',
                lw=2,
                marker='None',
                label='Test case')
        ax.set_yscale("log", nonposy='clip')
        ax.set_xlabel('$\mathrm{\log_{10}(E/GeV)}$')
        if idx == 0:
            ax.set_ylabel('$\mathrm{ E^{2.7} \ J(E) \ [GeV^{1.7} m^{-2} sr^{-1} s^{-1}]}$')
        ax.set_title(composition)
        ax.grid(lw=1, which='both')
        ax.legend()
    true_flux_outfile = os.path.join(
        figures_dir,
        'true_flux_{}-groups_{}-case.png'.format(num_groups, case))
    comp.check_output_dir(true_flux_outfile)
    plt.savefig(true_flux_outfile)

    # Run analysis pipeline on simulation
    counts_observed = pd.DataFrame(0, index=range(num_ebins),
                                   columns=comp_list)
    counts_observed_err = pd.DataFrame(0, index=range(num_ebins),
                                       columns=comp_list)
    weights = pd.DataFrame(0, index=range(num_ebins),
                           columns=comp_list)
    # Construct mask for energy bin
    energy_bins = np.digitize(df_sim_data['MC_log_energy'],
                              bins=energybins.log_energy_bins) - 1
    for idx_log_energy, composition in itertools.product(
                                range(len(energybins.log_energy_midpoints)),
                                comp_list):

        # Filter out events that don't pass composition & energy mask
        comp_mask = df_sim_data['comp_group_{}'.format(num_groups)] == composition
        energy_mask = energy_bins == idx_log_energy
        df_sim_bin = df_sim_data.loc[comp_mask & energy_mask, :]

        # Reweight simulation events to get desired number of events
        weight = counts_true[composition][idx_log_energy] / df_sim_bin.shape[0]
        # weight = counts_true.loc[log_energy, composition] / df_sim_bin.shape[0]
        weights.loc[idx_log_energy, composition] = weight

        # Get predicted composition
        y_test = df_sim_bin['comp_target_{}'.format(num_groups)].values
        X_test = df_sim_bin[feature_list].values
        if p is not None:
            pred_target = custom_predict(y_test, p=p, num_groups=num_groups)
        else:
            pred_target = pipeline.predict(X_test)
        pred_comp = np.array(comp.decode_composition_groups(
                             pred_target, num_groups=num_groups))
        assert len(pred_comp) == df_sim_bin.shape[0]
        for p_comp in np.unique(pred_comp):
            pred_comp_mask = pred_comp == p_comp
            comp_counts, _ = np.histogram(df_sim_bin.loc[pred_comp_mask, 'reco_log_energy'],
                                          bins=energybins.log_energy_bins)
            counts_observed[p_comp] += weight * comp_counts
            counts_observed_err[p_comp] += [sum(weight**2 for _ in range(c)) for c in comp_counts]
    # Square root the sum of squares of the weight errors
    for composition in comp_list:
        counts_observed_err[composition] = np.sqrt(counts_observed_err[composition])
    counts_observed_err['total'] = np.sqrt(np.sum(counts_observed_err[composition]**2 for composition in comp_list))
    # Calculate total counts
    counts_observed['total'] = counts_observed.sum(axis=1)

    # Plot weights for each composition and energy bin
    fig, ax = plt.subplots()
    for composition in comp_list:
        weights[composition].plot(ls=':', label=composition,
                                  color=color_dict[composition], ax=ax)
    ax.set_xlabel('$\mathrm{\log_{10}(E/GeV)}$')
    ax.set_ylabel('Weights')
    ax.set_yscale("log", nonposy='clip')
    ax.grid(lw=1)
    ax.legend()
    weights_outfile = os.path.join(
        figures_dir, 'weights_{}-groups_{}.png'.format(num_groups, case))
    comp.check_output_dir(weights_outfile)
    plt.savefig(weights_outfile)

    # Format observed counts, detection efficiencies, and priors for PyUnfold use
    counts_pyunfold = np.empty(num_groups * len(energybins.energy_midpoints))
    counts_err_pyunfold = np.empty(num_groups * len(energybins.energy_midpoints))
    efficiencies = np.empty(num_groups * len(energybins.energy_midpoints))
    efficiencies_err = np.empty(num_groups * len(energybins.energy_midpoints))
    for idx, composition in enumerate(comp_list):
        counts_pyunfold[idx::num_groups] = counts_observed[composition]
        counts_err_pyunfold[idx::num_groups] = counts_observed_err[composition]
        efficiencies[idx::num_groups] = df_eff['eff_median_{}'.format(composition)]
        efficiencies_err[idx::num_groups] = df_eff['eff_err_low_{}'.format(composition)]

    formatted_df = pd.DataFrame({'counts': counts_pyunfold,
                                 'counts_err': counts_err_pyunfold,
                                 'efficiencies': efficiencies,
                                 'efficiencies_err': efficiencies_err})
    formatted_file = os.path.join(data_dir,
                                  'test_{}_{}_{}.hdf'.format(prior, case, ts_stopping))
    formatted_df.to_hdf(formatted_file, 'dataframe', mode='w')

    root_file = os.path.join(data_dir,
                             'test_{}_{}_{}.root'.format(prior, case, ts_stopping))
    comp.save_pyunfold_root_file(config=config,
                                 num_groups=num_groups,
                                 outfile=root_file,
                                 formatted_df_file=formatted_file,
                                 res_mat_file=res_mat_outfile,
                                 res_mat_err_file=res_mat_err_outfile)

    if prior == 'Jeffreys':
        prior_pyunfold = 'Jeffreys'
        print('Jeffreys prior')
    else:
        model_flux = comp.model_flux(model=prior,
                                     energy=energybins.energy_midpoints,
                                     num_groups=num_groups)
        prior_pyunfold = np.empty(num_groups * len(energybins.energy_midpoints))
        for idx, composition in enumerate(comp_list):
            prior_pyunfold[idx::num_groups] = model_flux['flux_{}'.format(composition)]
        # Want to ensure prior_pyunfold are probabilities (i.e. they add to 1)
        prior_pyunfold = prior_pyunfold / np.sum(prior_pyunfold)

    df_unfolding_iter = unfold(config_name=os.path.join(config, 'config.cfg'),
                               priors=prior_pyunfold,
                               input_file=root_file,
                               ts_stopping=ts_stopping)
    # Delete temporary ROOT file needed for PyUnfold
    os.remove(root_file)
    os.remove(formatted_file)

    # print('\n{} case (prior {}): {} iterations'.format(case, prior, df_unfolding_iter.shape[0]))

    output = {'prior': prior, 'ts_stopping': ts_stopping, 'case': case}
    counts, counts_sys_err, counts_stat_err = comp.unfolded_counts_dist(
                                                    df_unfolding_iter,
                                                    iteration=-1,
                                                    num_groups=num_groups)
    for idx, composition in enumerate(comp_list + ['total']):
        # Pre-unfolding flux plot
        initial_counts = counts_observed[composition].values
        initial_counts_err = counts_observed_err[composition].values
        # initial_counts_err = np.sqrt(initial_counts)
        initial_flux, initial_flux_err_stat = counts_to_flux(
                                            initial_counts,
                                            initial_counts_err,
                                            composition=composition)
        initial_flux_err_sys = np.zeros_like(initial_flux)

        # Unfolded flux plot
        flux, flux_err_sys = unfolded_counts_to_flux(
                                            counts[composition],
                                            counts_sys_err[composition])
        flux, flux_err_stat = unfolded_counts_to_flux(
                                            counts[composition],
                                            counts_stat_err[composition])

        # True flux
        true_counts = counts_true[composition].values
        true_counts_err = np.sqrt(true_counts)
        true_flux, true_flux_err_stat = counts_to_flux(
                                            true_counts,
                                            true_counts_err,
                                            composition=composition)
        true_flux_err_sys = np.zeros_like(true_flux)

        output['initial_counts_{}'.format(composition)] = initial_counts
        output['initial_counts_err_{}'.format(composition)] = initial_counts_err

        output['flux_{}'.format(composition)] = flux
        output['flux_err_stat_{}'.format(composition)] = flux_err_stat
        output['flux_err_sys_{}'.format(composition)] = flux_err_sys
        output['true_flux_{}'.format(composition)] = true_flux
        output['true_flux_err_stat_{}'.format(composition)] = true_flux_err_stat
        output['true_flux_err_sys_{}'.format(composition)] = true_flux_err_sys
        output['initial_flux_{}'.format(composition)] = initial_flux
        output['initial_flux_err_stat_{}'.format(composition)] = initial_flux_err_stat
        output['initial_flux_err_sys_{}'.format(composition)] = initial_flux_err_sys

    # Don't want to consume too much memory by keeping too many figures open
    plt.close('all')

    return output


def save_flux_plot(group, config, case, ts_stopping, num_groups):
    """Saves flux comparison plot
    """
    comp_list = comp.get_comp_list(num_groups=num_groups)
    energybins = comp.get_energybins(config)

    # Get plotting axis
    figures_dir = os.path.join(comp.paths.figures_dir, 'unfolding', config,
                               'datachallenge', '{}_case'.format(case),
                               'prior_comparisons',
                               'ts_stopping_{}'.format(ts_stopping))

    # Make initial counts (pre-unfolding) plot
    fig_counts, ax_counts = plt.subplots()

    fig = plt.figure(figsize=(12, 5))
    gs = gridspec.GridSpec(nrows=2, ncols=num_groups+1, hspace=0.075)
    axs_flux, axs_ratio = {}, {}
    for idx, composition in enumerate(comp_list + ['total']):
        if idx == 0:
            axs_flux[composition] = fig.add_subplot(gs[0, idx])
        else:
            axs_flux[composition] = fig.add_subplot(gs[0, idx], sharey=axs_flux[comp_list[0]])
        axs_ratio[composition] = fig.add_subplot(gs[1, idx], sharex=axs_flux[composition])
    prior_groupby = group.groupby('prior')
    marker_iter = iter('.^x*')
    ls_iter = iter(['-', ':', '-.', '--'])
    initial_flux_test, initial_flux_err_stat_test, initial_flux_err_sys_test = {}, {}, {}
    for prior_idx, (prior, df_group) in enumerate(prior_groupby):
        marker = next(marker_iter)
        ls = next(ls_iter)
        label = priors_labels[priors.index(prior)]
        for idx, composition in enumerate(comp_list + ['total']):
            ax_flux = axs_flux[composition]
            ax_ratio = axs_ratio[composition]

            color = sns.color_palette(comp.get_colormap(composition), len(priors)+3).as_hex()[-1*(prior_idx + 2)]
            true_color = sns.color_palette(comp.get_colormap(composition), len(priors)+3).as_hex()[-1]

            # True flux
            true_flux = df_group['true_flux_{}'.format(composition)].values[0]
            true_flux_err_stat = df_group['true_flux_err_stat_{}'.format(composition)].values[0]
            true_flux_err_sys = df_group['true_flux_err_sys_{}'.format(composition)].values[0]
            if prior_idx == 0:
                ax_flux.errorbar(energybins.log_energy_midpoints, true_flux, yerr=true_flux_err_stat,
                                 color=true_color, ls='None', marker='*',
                                 label='True flux', alpha=0.8)

            # Unfolded flux
            flux = df_group['flux_{}'.format(composition)].values[0]
            flux_err_stat = df_group['flux_err_stat_{}'.format(composition)].values[0]
            flux_err_sys = df_group['flux_err_sys_{}'.format(composition)].values[0]
            if not plot_initial_flux:
                comp.plot_steps(energybins.log_energy_bins, flux, yerr=flux_err_sys,
                                ax=ax_flux, alpha=0.4, fillalpha=0.4,
                                color=color, ls=ls)
                ax_flux.errorbar(energybins.log_energy_midpoints, flux, yerr=flux_err_stat,
                                 color=color, ls='None', marker=marker,
                                 label='Unfolded ({})'.format(label), alpha=0.8)

            # Initial (pre-unfolding) flux
            initial_flux = df_group['initial_flux_{}'.format(composition)].values[0]
            initial_flux_err_stat = df_group['initial_flux_err_stat_{}'.format(composition)].values[0]
            initial_flux_err_sys = df_group['initial_flux_err_sys_{}'.format(composition)].values[0]

            # Sanity check that all the initial_flux (what goes into the unfolding)
            # are the same for each prior.
            if prior_idx == 0:
                initial_flux_test[composition] = initial_flux
                initial_flux_err_stat_test[composition] = initial_flux_err_stat
                initial_flux_err_sys_test[composition] = initial_flux_err_sys

                initial_counts = df_group['initial_counts_{}'.format(composition)].values[0]
                initial_counts_err = df_group['initial_counts_err_{}'.format(composition)].values[0]
                comp.plot_steps(energybins.log_energy_bins, initial_counts, yerr=initial_counts_err,
                                ax=ax_counts, alpha=0.8, fillalpha=0.4,
                                color=color_dict[composition], label=composition)
                ax_counts.set_yscale("log", nonposy='clip')
                ax_counts.set_ylabel('Observed counts from BDT')
                # ax_counts.set_ylabel('$\mathrm{ E^{2.7} \ J(E) \ [GeV^{1.7} m^{-2} sr^{-1} s^{-1}]}$')
                ax_counts.set_xlabel('$\mathrm{\log_{10}(E/GeV)}$')
                ax_counts.set_xlim(6.4, 7.8)
                ax_counts.grid()
                ax_counts.legend()

            else:
                np.testing.assert_allclose(initial_flux_test[composition], initial_flux)
                np.testing.assert_allclose(initial_flux_err_stat_test[composition], initial_flux_err_stat)
                np.testing.assert_allclose(initial_flux_err_sys_test[composition], initial_flux_err_sys)

            if plot_initial_flux:
                comp.plot_steps(energybins.log_energy_bins, initial_flux, yerr=initial_flux_err_sys,
                                ax=ax_flux, alpha=0.4, fillalpha=0.4,
                                color=color, ls=ls)
                ax_flux.errorbar(energybins.log_energy_midpoints, initial_flux, yerr=initial_flux_err_stat,
                                 color=color, ls='None', marker=marker,
                                 label='Initial ({})'.format(label), alpha=0.8)

            ax_flux.set_yscale("log", nonposy='clip')
            ax_flux.set_xlim(6.4, 7.8)
            ax_flux.set_ylim(1e3, 1e5)
            ax_flux.grid(linestyle='dotted', which="both", lw=1)
            ax_flux.set_title(composition, fontsize=10)
            if composition == 'total':
                ax_flux.legend(fontsize=7, loc='lower left')
            if idx == 0:
                ax_flux.set_ylabel('$\mathrm{ E^{2.7} \ J(E) \ [GeV^{1.7} m^{-2} sr^{-1} s^{-1}]}$',
                                   fontsize=10)
            else:
                plt.setp(ax_flux.get_yticklabels(), visible=False)
            plt.setp(ax_flux.get_xticklabels(), visible=False)
            ax_flux.tick_params(axis='both', which='major', labelsize=10)

            if plot_initial_flux:
                frac_diff, frac_diff_stat, frac_diff_sys = calculate_ratio(
                                    initial_flux, initial_flux_err_stat, initial_flux_err_sys,
                                    true_flux, true_flux_err_stat, true_flux_err_sys)
            else:
                frac_diff, frac_diff_stat, frac_diff_sys = calculate_ratio(
                                    flux, flux_err_stat, flux_err_sys,
                                    true_flux, true_flux_err_stat, true_flux_err_sys)

            comp.plot_steps(energybins.log_energy_bins, frac_diff, yerr=frac_diff_sys,
                            ax=ax_ratio, alpha=0.4, fillalpha=0.4,
                            color=color, ls=ls)
            ax_ratio.errorbar(energybins.log_energy_midpoints, frac_diff, yerr=frac_diff_stat,
                              color=color, ls='None', marker=marker,
                              label='Unfolded ({})'.format(label), alpha=0.8)
            ax_ratio.axhline(0, ls='-.', lw=1, marker='None', color='k')

            ax_ratio.grid(linestyle='dotted', which="both", lw=1)
            ax_ratio.set_yticks(np.arange(-1, 1.5, 0.25))
            ax_ratio.set_ylim(-1, 1)
            if idx == 0:
                ax_ratio.set_ylabel('$\mathrm{(J - J_{true}) / J_{true}}$',
                                    fontsize=10)
            else:
                plt.setp(ax_ratio.get_yticklabels(), visible=False)
            ax_ratio.set_xlabel('$\mathrm{\log_{10}(E/GeV)}$', fontsize=10)
            ax_ratio.tick_params(axis='both', which='major', labelsize=10)

    plt.tight_layout()
    flux_outfile = os.path.join(figures_dir,
                                'flux_ratio_{}-groups_{}-case.png'.format(num_groups, case))
    comp.check_output_dir(flux_outfile)
    plt.savefig(flux_outfile)

    counts_outfile = os.path.join(figures_dir,
                                 'counts_{}-groups_{}-case.png'.format(num_groups, case))
    comp.check_output_dir(counts_outfile)
    fig_counts.savefig(counts_outfile)

    # Don't want to consume too much memory by keeping too many figures open
    plt.close('all')


def custom_predict(y, p=0.8, neighbor_weight=2.0, num_groups=4):
    """Function to perform random composition classification
    """
    np.random.seed(2)
    p_correct = p
    y_pred = np.empty_like(y)
    targets = list(range(num_groups))
    probs = np.empty_like(targets, dtype=float)
    for target in targets:
        comp_mask = y == target
        probs[target] = p_correct

        not_target = [i for i in targets if i != target]

        neighbors = [target - 1, target + 1]
        neighbors = [i for i in neighbors if i >= 0 and i < num_groups]

        not_neighbors = list(set(not_target).difference(neighbors))

        weight = (1 - p_correct) / (len(not_neighbors) + neighbor_weight * len(neighbors))

        probs[not_neighbors] = weight
        probs[neighbors] = neighbor_weight * weight

        # Get custom composition classification
        y_pred_target = np.random.choice(targets, size=comp_mask.sum(), p=probs)
        y_pred[comp_mask] = y_pred_target

    return y_pred


def calculate_sample_weights(compositions, energies, model=None,
                             compositon_weights=None,
                             energy_spectrum_weights=None):

    if not compositions.shape == energies.shape:
        raise ValueError('compositions and energies must have the same shape, '
                         'but got shapes {} and {}'.format(compositions.shape, energies.shape))

    num_groups = len(np.unique(compositions))
    assert num_groups in [2, 3, 4], 'Invalid number of groups, {}'.format(num_groups)

    if model is not None:
        sample_weight = calculate_model_sample_weights(compositions, energies, num_groups, model=model)
    elif any([compositon_weights, energy_spectrum_weights]):
        sample_weight = calculate_simple_sample_weights(compositions, energies, num_groups,
                                                        compositon_weights=compositon_weights,
                                                        energy_spectrum_weights=energy_spectrum_weights)
    else:
        raise ValueError('At least one of model, compositon_weights, '
                         'or energy_spectrum_weights must not be None')

    # Normalize such that sample_weight.sum() is 1
    sample_weight = sample_weight / sample_weight.sum()

    return sample_weight

def calculate_simple_sample_weights(compositions, energies, num_groups, compositon_weights=None,
                                    energy_spectrum_weights=None):

    if not any([compositon_weights, energy_spectrum_weights]):
        raise ValueError('Either compositon_weights or energy_spectrum_weights must be specified')

    num_groups = len(np.unique(compositions))
    assert len(compositon_weights) == num_groups
    comp_list = comp.get_comp_list(num_groups=num_groups)
    sample_weight = np.empty_like(energies, dtype=float)
    # Weighting based on composition
    if compositon_weights is not None:
        compositon_weights = map(float, compositon_weights)
        print('Using compositon_weights: {}'.format(compositon_weights))
        for composition, comp_weight in zip(comp_list, compositon_weights):
            comp_mask = compositions == composition
            sample_weight[comp_mask] = comp_weight
    # Weighting based on energy
    if energy_spectrum_weights is not None:
        # + 1 because the simulation was generated on an E^-1 spectrum
        sample_weight = sample_weight * energies**(energy_spectrum_weights + 1)

    return sample_weight


def calculate_model_sample_weights(compositions, energies, num_groups, model='H4a'):

    comp_list = comp.get_comp_list(num_groups=num_groups)
    sample_weight = np.empty_like(energies, dtype=float)
    models = ['H3a',
              'H4a',
              'simple_power_law',
              'broken_power_law']
    if model in models:
        flux_df = comp.model_flux(model=model, energy=energies, num_groups=num_groups)
        for composition in comp_list:
            comp_mask = compositions == composition
            comp_flux = flux_df.loc[comp_mask, 'flux_{}'.format(composition)].values
            sample_weight[comp_mask] = comp_flux
    else:
        raise ValueError('Invalid model name, {}, entered.'.format(model))

    return sample_weight


def get_composition_pipeline(pipeline, p, use_sample_weights):
    if p is None:
        pipeline_str = '{}_comp_{}_{}-groups'.format(pipeline, config, num_groups)
        pipeline = comp.load_trained_model(pipeline_str)
    if use_sample_weights is not None:
        model = use_sample_weights
        pipeline_str = '{}_comp_{}_{}-groups'.format(pipeline, config, num_groups)
        pipeline = comp.get_pipeline(pipeline_str)
        compositions = df_sim_train['comp_group_{}'.format(num_groups)].values
        energies = df_sim_train['reco_energy'].values
        sample_weight = calculate_sample_weights(compositions, energies, model=model)
        X = df_sim_train[feature_list].values
        y = df_sim_train['comp_target_{}'.format(num_groups)].values
        fit_params = {'classifier__sample_weight': sample_weight}
        pipeline.fit(X, y, **fit_params)

    return pipeline


if __name__ == '__main__':

    description = 'Script to run unfolding analysis on known input fluxes'
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('-c', '--config', dest='config',
                        default='IC86.2012',
                        choices=comp.simfunctions.get_sim_configs(),
                        help='Detector configuration')
    parser.add_argument('--num_groups', dest='num_groups', type=int,
                        default=4, choices=[2, 3, 4],
                        help='Detector configuration')
    parser.add_argument('--pipeline', dest='pipeline',
                        default='BDT',
                        help=('Composition classifier to use (e.g. "BDT", '
                              '"LinearSVC", etc.)'))
    parser.add_argument('--weights_model', dest='weights_model',
                        default=None,
                        help='Option to use sample weights when training composition classifier.')
    parser.add_argument('--energy_spectrum_weights', dest='energy_spectrum_weights',
                        default=None, type=float,
                        help='Option to use sample weights when training composition classifier.')
    parser.add_argument('--compositon_weights', dest='compositon_weights',
                        default=None, nargs='*',
                        help='Option to use sample weights when training composition classifier.')
    parser.add_argument('--prob_correct', dest='prob_correct',
                        type=float,
                        help=('Probability event is correctly classified for '
                              'custom composition classification'))
    parser.add_argument('--n_jobs', dest='n_jobs', type=int,
                        default=1,
                        help='Number of jobs to run in parallel')
    args = parser.parse_args()

    color_dict = comp.get_color_dict()

    config = args.config
    num_groups = args.num_groups
    p = args.prob_correct

    comp_list = comp.get_comp_list(num_groups=num_groups)
    energybins = comp.get_energybins(config)
    num_ebins = len(energybins.log_energy_midpoints)

    data_dir = os.path.join(comp.paths.comp_data_dir, config, 'unfolding',
                            'datachallenge')

    # Load simulation and train composition classifier
    df_sim_train, df_sim_test = comp.load_sim(config=config,
                                              energy_reco=False,
                                              log_energy_min=None,
                                              log_energy_max=None,
                                              test_size=0.5,
                                              verbose=True)

    feature_list, feature_labels = comp.get_training_features()

    print('Loading energy regressor...')
    energy_pipeline = comp.load_trained_model('RF_energy_{}'.format(config))
    for df in [df_sim_train, df_sim_test]:
        df['reco_log_energy'] = energy_pipeline.predict(df[feature_list].values)
        df['reco_energy'] = 10**df['reco_log_energy']

    print('Loading or fitting composition classifier...')
    if any([args.weights_model, args.energy_spectrum_weights, args.compositon_weights]):
        model = args.weights_model
        energy_spectrum_weights = args.energy_spectrum_weights
        compositon_weights = args.compositon_weights

        pipeline_str = '{}_comp_{}_{}-groups'.format(args.pipeline, config, num_groups)
        pipeline = comp.get_pipeline(pipeline_str)
        compositions = df_sim_train['comp_group_{}'.format(num_groups)].values
        energies = df_sim_train['reco_energy'].values
        sample_weight = calculate_sample_weights(compositions, energies, model=model,
                                                 compositon_weights=compositon_weights,
                                                 energy_spectrum_weights=energy_spectrum_weights)
        X = df_sim_train[feature_list].values
        y = df_sim_train['comp_target_{}'.format(num_groups)].values
        fit_params = {'classifier__sample_weight': sample_weight}
        pipeline.fit(X, y, **fit_params)
    elif p is None:
        pipeline_str = '{}_comp_{}_{}-groups'.format(args.pipeline, config, num_groups)
        pipeline = comp.load_trained_model(pipeline_str)

    df_sim_response = df_sim_test
    df_sim_data = df_sim_test

    # df_sim_response, df_sim_data = train_test_split(df_sim_test,
    #                                                 test_size=0.5,
    #                                                 shuffle=True,
    #                                                 random_state=2)

    # Solid angle
    theta_max = 40 if config == 'IC79.2010' else 65
    solid_angle = np.pi*np.sin(np.deg2rad(theta_max))**2
    livetime, livetime_err = comp.get_detector_livetime(config=config)

    # Get simulation thrown areas for each energy bin
    thrown_radii = comp.simfunctions.get_sim_thrown_radius(energybins.log_energy_midpoints)
    thrown_area = np.max(np.pi * thrown_radii**2)

    # Load fitted efficiencies and calculate effective areas
    eff_path = os.path.join(comp.paths.comp_data_dir, config, 'efficiencies',
                            'efficiency_fit_num_groups_{}.hdf'.format(num_groups))
    df_eff = pd.read_hdf(eff_path)

    eff_area, eff_area_err = {}, {}
    for composition in comp_list+['total']:
        eff_area[composition] = df_eff['eff_median_{}'.format(composition)].values * thrown_area
        eff_area_err[composition] = df_eff['eff_err_low_{}'.format(composition)].values * thrown_area

    # Format for PyUnfold response matrix use
    efficiencies = np.empty(num_groups * len(energybins.energy_midpoints))
    efficiencies_err = np.empty(num_groups * len(energybins.energy_midpoints))
    for idx, composition in enumerate(comp_list):
        efficiencies[idx::num_groups] = df_eff['eff_median_{}'.format(composition)]
        efficiencies_err[idx::num_groups] = df_eff['eff_err_low_{}'.format(composition)]

    print('Making response matrix...')
    X_test = df_sim_response[feature_list].values
    y_test = df_sim_response['comp_target_{}'.format(num_groups)].values
    if p is not None:
        pred_target = custom_predict(y_test, p=p, num_groups=num_groups)
    else:
        pred_target = pipeline.predict(X_test)

    log_reco_energy_sim_test = df_sim_response['reco_log_energy']
    log_true_energy_sim_test = df_sim_response['MC_log_energy']

    res_normalized, res_normalized_err = comp.normalized_response_matrix(
                                            true_energy=log_true_energy_sim_test,
                                            reco_energy=log_reco_energy_sim_test,
                                            true_target=y_test,
                                            pred_target=pred_target,
                                            efficiencies=efficiencies,
                                            efficiencies_err=efficiencies_err,
                                            energy_bins=energybins.log_energy_bins)
    res_mat_outfile = os.path.join(
                                data_dir,
                                'response_{}-groups.txt'.format(num_groups))
    res_mat_err_outfile = os.path.join(
                                data_dir,
                                'response_err_{}-groups.txt'.format(num_groups))
    comp.check_output_dir(res_mat_outfile)
    comp.check_output_dir(res_mat_err_outfile)
    np.savetxt(res_mat_outfile, res_normalized)
    np.savetxt(res_mat_err_outfile, res_normalized_err)

    # Define convenience functions that will convert counts to flux
    unfolded_counts_to_flux = partial(comp.get_flux,
                                      energybins=energybins.energy_bins,
                                      eff_area=thrown_area,
                                      livetime=livetime,
                                      livetime_err=livetime_err,
                                      solid_angle=solid_angle,
                                      scalingindex=2.7)

    def counts_to_flux(counts, counts_err=None, composition=None):
        return comp.get_flux(counts, counts_err,
                             energybins=energybins.energy_bins,
                             eff_area=eff_area[composition],
                             eff_area_err=eff_area_err[composition],
                             livetime=livetime,
                             livetime_err=livetime_err,
                             solid_angle=solid_angle,
                             scalingindex=2.7)

    priors = [
              'simple_power_law',
              'broken_power_law',
              # 'Jeffreys',
              'H3a',
              'H4a',
              ]

    priors_labels = [
                      'Simple PL',
                      'Broken PL',
                      # 'Jeffreys',
                      'H3a',
                      'H4a',
                      ]

    cases = [
             # 'constant',
             'simple_power_law',
             # # 'broken_power_law_0',
             # # 'broken_power_law_1',
             # 'broken_power_law_2',
             # 'H3a',
             'H4a',
             ]
    ts_values = [
                 # 0.01,
                 0.005,
                 0.001,
                 0.0005,
                 ]

    plot_initial_flux = False

    calculations = []
    for case, ts_stopping, prior in itertools.product(cases, ts_values, priors):
        calc = delayed(main)(config, num_groups, prior, ts_stopping=ts_stopping,
                             case=case, p=p)
        calculations.append(calc)

    df = delayed(pd.DataFrame.from_records)(calculations)
    with ProgressBar():
        print('Running analysis over known injected fluxes...')
        if args.n_jobs == 1:
            df = df.compute(num_workers=args.n_jobs, get=get)
        else:
            df = df.compute(num_workers=args.n_jobs, get=multiprocessing.get)

    # Plotting
    print('Making flux comparison plots...')
    for (case, ts_stopping), group in df.groupby(['case', 'ts_stopping']):
        save_flux_plot(group=group, config=config, case=case,
                       ts_stopping=ts_stopping, num_groups=num_groups)
