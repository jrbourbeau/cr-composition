#!/usr/bin/env python

from __future__ import division, print_function
import os
import argparse
from collections import defaultdict
from itertools import product
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn.apionly as sns
from sklearn.metrics import confusion_matrix
from ROOT import TH1F, TH2F, TFile
import dask.array as da
from dask.diagnostics import ProgressBar

from icecube.weighting.weighting import PDGCode
from icecube.weighting.fluxes import GaisserH3a, GaisserH4a, Hoerandel5

import comptools as comp


if 'cvmfs' in os.getenv('ROOTSYS'):
    raise comp.ComputingEnvironemtError('CVMFS ROOT cannot be used for unfolding')


def column_normalize(res, res_err, efficiencies, efficiencies_err):
    res_col_sum = res.sum(axis=0)
    res_col_sum_err = np.array([np.sqrt(np.sum(res_err[:, i]**2))
                                for i in range(res_err.shape[1])])

    normalizations, normalizations_err = comp.ratio_error(
                                            res_col_sum, res_col_sum_err,
                                            efficiencies, efficiencies_err,
                                            nan_to_num=True)

    res_normalized, res_normalized_err = comp.ratio_error(
                                            res, res_err,
                                            normalizations, normalizations_err,
                                            nan_to_num=True)

    res_normalized = np.nan_to_num(res_normalized)
    res_normalized_err = np.nan_to_num(res_normalized_err)

    # Test that the columns of res_normalized equal efficiencies
    np.testing.assert_allclose(res_normalized.sum(axis=0), efficiencies)

    return res_normalized, res_normalized_err


def response_matrix(true_energy, reco_energy, true_target, pred_target,
                    efficiencies, efficiencies_err, energy_bins=None):

    # Check that the input array shapes
    inputs = [true_energy, reco_energy, true_target, pred_target]
    assert len(set(map(np.ndim, inputs))) == 1
    assert len(set(map(np.shape, inputs))) == 1

    num_ebins = len(energy_bins) - 1
    num_groups = len(np.unique([true_target, pred_target]))

    true_ebin_indices = np.digitize(true_energy, energy_bins) - 1
    reco_ebin_indices = np.digitize(reco_energy, energy_bins) - 1

    res = np.zeros((num_ebins * num_groups, num_ebins * num_groups))
    bin_iter = product(range(num_ebins), range(num_ebins),
                       range(num_groups), range(num_groups))
    for true_ebin, reco_ebin, true_target_bin, pred_target_bin in bin_iter:
        # Get mask for events in true/reco energy and true/reco composition bin
        mask = np.logical_and.reduce((true_ebin_indices == true_ebin,
                                      reco_ebin_indices == reco_ebin,
                                      true_target == true_target_bin,
                                      pred_target == pred_target_bin))
        res[num_groups * reco_ebin + pred_target_bin,
            num_groups * true_ebin + true_target_bin] = mask.sum()
    # Calculate statistical error on response matrix
    res_err = np.sqrt(res)

    # Normalize response matrix column-wise (i.e. $P(E|C)$)
    res_normalized, res_normalized_err = column_normalize(res, res_err,
                                                          efficiencies,
                                                          efficiencies_err)

    return res_normalized, res_normalized_err


def save_pyunfold_root_file(config, num_groups, outfile=None,
                            formatted_df_file=None, res_mat_file=None,
                            res_mat_err_file=None):

    unfolding_dir  = os.path.join(comp.paths.comp_data_dir, config,
                                  'unfolding')
    # Bin Definitions
    binname = 'bin0'
    # ROOT Output
    if outfile is None:
        outfile  = os.path.join(unfolding_dir,
                                'pyunfold_input_{}-groups.root'.format(num_groups))
    comp.check_output_dir(outfile)
    if os.path.exists(outfile):
        os.remove(outfile)

    fout = TFile(outfile , 'UPDATE')
    # Check if bin directory exists, quit if so, otherwise create it!
    if not fout.GetDirectory(binname):
        pdir = fout.mkdir(binname, 'Bin number 0')
    else:
        fout.Close()
        print('\n=========================\n')
        raise ValueError('Directory {} already exists!\nEither try another '
                         'bin number or delete {} and start again. '
                         'Exiting...\n'.format(binname, outfile))

    # Go to home of ROOT file
    fout.cd(binname)

    if formatted_df_file is None:
        formatted_df_file  = os.path.join(
                unfolding_dir, 'unfolding-df_{}-groups.hdf'.format(num_groups))
    df_flux = pd.read_hdf(formatted_df_file)
    counts = df_flux['counts'].values
    if 'counts_err' in df_flux:
        counts_err = df_flux['counts_err'].values
    else:
        counts_err = None
    efficiencies = df_flux['efficiencies'].values
    efficiencies_err = df_flux['efficiencies_err'].values

    cbins = len(counts)+1
    carray = np.arange(cbins, dtype=float)

    ebins = len(counts)+1
    earray = np.arange(ebins, dtype=float)
    cbins -= 1
    ebins -= 1

    # Load response matrix array
    if res_mat_file is None:
        res_mat_file = os.path.join(unfolding_dir,
                                    'response_{}-groups.txt'.format(num_groups))
    response_array = np.loadtxt(res_mat_file)
    if res_mat_err_file is None:
        res_mat_err_file = os.path.join(
                                unfolding_dir,
                                'response_err_{}-groups.txt'.format(num_groups))
    response_err_array = np.loadtxt(res_mat_err_file)

    # Measured effects distribution
    ne_meas = TH1F('ne_meas', 'effects histogram', ebins, earray)
    ne_meas.GetXaxis().SetTitle('Effects')
    ne_meas.GetYaxis().SetTitle('Counts')
    ne_meas.SetStats(0)
    ne_meas.Sumw2()

    # Prepare Combined Weighted Histograms - To be Normalized by Model After Filling
    # Isotropic Weights of Causes - For Calculating Combined Species Efficiency
    eff = TH1F('Eff', 'Non-Normed Combined Efficiency', cbins, carray)
    eff.GetXaxis().SetTitle('Causes')
    eff.GetYaxis().SetTitle('Efficiency')
    eff.SetStats(0)
    eff.Sumw2()

    # Isotropic Weighted Mixing Matrix - For Calculating Combined Species MM
    response = TH2F('MM', 'Weighted Combined Mixing Matrix',
                    cbins, carray, ebins, earray)
    response.GetXaxis().SetTitle('Causes')
    response.GetYaxis().SetTitle('Effects')
    response.SetStats(0)
    response.Sumw2()

    for ci in range(0, cbins):

        # Fill measured effects histogram
        ne_meas.SetBinContent(ci+1, counts[ci])
        if counts_err is None:
            ne_meas.SetBinError(ci+1, np.sqrt(counts[ci]))
        else:
            ne_meas.SetBinError(ci+1, counts_err[ci])
        # print('ne_meas[{}] = {}'.format(ci+1, counts[ci]))

        for ek in range(0, ebins):
            # Fill response matrix entries
            response.SetBinContent(ci+1, ek+1, response_array[ek][ci])
            response.SetBinError(ci+1, ek+1, response_err_array[ek][ci])

        # # Fill efficiency histogram from response matrix
        # eff.SetBinContent(ci+1, np.sum(response_array[:, ci]))
        # eff.SetBinError(ci+1, np.sqrt(np.sum(response_err_array[:, ci]**2)))
        # Fill efficiency histogram from response matrix
        eff.SetBinContent(ci+1, efficiencies[ci])
        eff.SetBinError(ci+1, efficiencies_err[ci])

    # Write measured effects histogram to file
    ne_meas.Write()
    # eff.Write()
    # Write the cause and effect arrays to file
    CARRAY = TH1F('CARRAY','Cause Array', cbins, carray)
    CARRAY.GetXaxis().SetTitle('Causes')
    EARRAY = TH1F('EARRAY','Effect Array', ebins, earray)
    EARRAY.GetXaxis().SetTitle('Effects')
    CARRAY.Write()
    EARRAY.Write()
    # Write efficiencies histogram to file
    eff.Write()
    # Write response matrix to file
    response.Write()
    # # Write model name to file
    # modelName = 'whatever'
    # MODELNAME = TNamed("ModelName",modelName)
    # MODELNAME.Write()

    fout.Write()
    fout.Close()

    # print('Saving output file {}'.format(outfile))

if __name__ == '__main__':

    description = ('Save things needed for unfolding (e.g. response matrix, '
                   'priors, observed counts, etc.)')
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('-c', '--config', dest='config', default='IC86.2012',
                        choices=comp.simfunctions.get_sim_configs(),
                        help='Detector configuration')
    parser.add_argument('--num_groups', dest='num_groups', type=int,
                        default=4, choices=[2, 3, 4],
                        help='Number of composition groups')
    args = parser.parse_args()

    config = args.config
    num_groups = args.num_groups

    comp_list = comp.get_comp_list(num_groups=num_groups)
    energybins = comp.get_energybins(config=config)
    log_energy_min = energybins.log_energy_min
    log_energy_max = energybins.log_energy_max

    # Load simulation training/testing DataFrames
    print('Loading simulation training/testing DataFrames...')
    df_sim_train, df_sim_test = comp.load_sim(config=config,
                                              log_energy_min=log_energy_min,
                                              log_energy_max=log_energy_max,
                                              test_size=0.5)

    log_reco_energy_sim_test = df_sim_test['reco_log_energy']
    log_true_energy_sim_test = df_sim_test['MC_log_energy']

    feature_list, feature_labels = comp.get_training_features()
    # pipeline_str = 'BDT_comp_{}_{}-groups'.format(config, num_groups)
    # pipeline_str = 'xgboost_comp_{}_{}-groups'.format(config, num_groups)
    # pipeline_str = 'SVC_comp_{}_{}-groups'.format(config, num_groups)
    pipeline_str = 'linecut_comp_{}_{}-groups'.format(config, num_groups)
    # pipeline_str = 'LinearSVC_comp_{}_{}-groups'.format(config, num_groups)
    # pipeline_str = 'LogisticRegression_comp_{}_{}-groups'.format(config, num_groups)
    pipeline = comp.get_pipeline(pipeline_str)

    # Fit composition classifier
    print('Fitting composition classifier...')
    X_train = df_sim_train[feature_list].values
    y_train = df_sim_train['comp_target_{}'.format(num_groups)].values
    pipeline = pipeline.fit(X_train, y_train)

    # Load fitted effective area
    print('Loading detection efficiencies...')
    eff_path = os.path.join(
                    comp.paths.comp_data_dir, config, 'efficiencies',
                    'efficiency_fit_num_groups_{}.hdf'.format(num_groups))
    df_eff = pd.read_hdf(eff_path)
    # Format for PyUnfold response matrix use
    efficiencies, efficiencies_err = [], []
    for idx, row in df_eff.iterrows():
        for composition in comp_list:
            efficiencies.append(row['eff_median_{}'.format(composition)])
            efficiencies_err.append(row['eff_err_low_{}'.format(composition)])
    efficiencies = np.asarray(efficiencies)
    efficiencies_err = np.asarray(efficiencies_err)

    # Load data DataFrame
    print('Loading data DataFrame...')
    df_data = comp.load_data(config=config, columns=feature_list,
                             log_energy_min=log_energy_min,
                             log_energy_max=log_energy_max,
                             n_jobs=15,
                             verbose=True)

    X_data = comp.io.dataframe_to_array(df_data, feature_list + ['reco_log_energy'])
    log_energy_data = X_data[:, -1]
    X_data = X_data[:, :-1]

    print('Making composition predictions on data...')
    # Apply pipeline.predict method in chunks
    X_da = da.from_array(X_data, chunks=(len(X_data) // 100, X_data.shape[1]))
    data_predictions = da.map_blocks(pipeline.predict, X_da,
                                     dtype=int, drop_axis=1)
    # Convert from target to composition labels
    data_labels = da.map_blocks(comp.decode_composition_groups, data_predictions,
                                dtype=str, num_groups=num_groups)
    with ProgressBar():
        data_labels = data_labels.compute(num_workers=20)

    # Get number of identified comp in each energy bin
    print('Formatting observed counts...')
    unfolding_df = pd.DataFrame()
    for composition in comp_list:
        comp_mask = data_labels == composition
        counts = np.histogram(log_energy_data[comp_mask],
                              bins=energybins.log_energy_bins)[0]
        counts_err = np.sqrt(counts)
        unfolding_df['counts_' + composition] = counts
        unfolding_df['counts_' + composition + '_err'] = counts_err

    unfolding_df['counts_total'] = np.histogram(
                                        log_energy_data,
                                        bins=energybins.log_energy_bins)[0]
    unfolding_df['counts_total_err'] = np.sqrt(unfolding_df['counts_total'])
    unfolding_df.index.rename('log_energy_bin_idx', inplace=True)

    # fig, ax = plt.subplots()
    # for composition in comp_list:
    #     ax.plot(unfolding_df['counts_{}'.format(composition)], color=color_dict[composition])
    # ax.set_yscale("log", nonposy='clip')
    # ax.grid()
    # plt.show()

    # Response matrix
    print('Making response matrix...')
    pred_target = pipeline.predict(df_sim_test[feature_list].values)
    true_target = df_sim_test['comp_target_{}'.format(num_groups)].values
    # true_comp = df_sim_test['comp_group_{}'.format(num_groups)].values
    # pred_comp = np.array(comp.composition_encoding.decode_composition_groups(
    #                         test_predictions, num_groups=num_groups))
    res_normalized, res_normalized_err = response_matrix(
                                            log_true_energy_sim_test,
                                            log_reco_energy_sim_test,
                                            true_target, pred_target,
                                            efficiencies, efficiencies_err,
                                            energy_bins=energybins.log_energy_bins)
    res_mat_outfile = os.path.join(
                            comp.paths.comp_data_dir, config, 'unfolding',
                            'response_{}-groups.txt'.format(num_groups))
    res_mat_err_outfile = os.path.join(
                            comp.paths.comp_data_dir, config, 'unfolding',
                            'response_err_{}-groups.txt'.format(num_groups))
    comp.check_output_dir(res_mat_outfile)
    comp.check_output_dir(res_mat_err_outfile)
    np.savetxt(res_mat_outfile, res_normalized)
    np.savetxt(res_mat_err_outfile, res_normalized_err)

    # Priors array
    print('Calcuating priors...')
    df_sim = comp.load_sim(config=config, test_size=0,
                           log_energy_min=6.0,
                           log_energy_max=8.3)

    p = PDGCode().values
    pdg_codes = np.array([2212, 1000020040, 1000080160, 1000260560])
    particle_names = [p[pdg_code].name for pdg_code in pdg_codes]
    group_names = np.array(comp.composition_encoding.composition_group_labels(
                           particle_names, num_groups=num_groups))
    comp_to_pdg_list = {composition: pdg_codes[group_names == composition]
                        for composition in comp_list}
    # Replace O16Nucleus with N14Nucleus + Al27Nucleus
    for composition, pdg_list in comp_to_pdg_list.iteritems():
        if 1000080160 in pdg_list:
            pdg_list = pdg_list[pdg_list != 1000080160]
            comp_to_pdg_list[composition] = np.append(pdg_list,
                                                      [1000070140, 1000130270])
        else:
            continue
    priors_list = ['H3a', 'H4a', 'Polygonato']

    print('Making priors flux plot...')
    color_dict = comp.get_color_dict()
    fig, ax = plt.subplots()
    for flux, name, marker in zip([GaisserH3a(), GaisserH4a(), Hoerandel5()],
                                  priors_list,
                                  '.^*o'):
        for composition in comp_list:
            comp_flux = []
            for energy_mid in energybins.energy_midpoints:
                flux_energy_mid = flux(energy_mid,
                                       comp_to_pdg_list[composition]).sum()
                comp_flux.append(flux_energy_mid)
            # Normalize flux in each energy bin to a probability
            comp_flux = np.asarray(comp_flux)
            prior_key = '{}_flux_{}'.format(name, composition)
            unfolding_df[prior_key] = comp_flux

            # Plot result
            ax.plot(energybins.log_energy_midpoints,
                    energybins.energy_midpoints**2.7*comp_flux,
                    color=color_dict[composition], alpha=0.75,
                    marker=marker, ls=':',
                    label='{} ({})'.format(name, composition))
    ax.set_yscale("log", nonposy='clip')
    ax.set_xlabel('$\mathrm{\log_{10}(E/GeV)}$')
    ylabel = '$\mathrm{ E^{2.7} \ J(E) \ [GeV^{1.7} m^{-2} sr^{-1} s^{-1}]}$'
    ax.set_ylabel(ylabel)
    ax.grid()
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), frameon=False)
    priors_outfile = os.path.join(
                            comp.paths.figures_dir, 'unfolding',
                            'priors_flux_{}-groups.png'.format(num_groups))
    comp.check_output_dir(priors_outfile)
    plt.savefig(priors_outfile)
    plt.show()

    print('Making PyUnfold formatted DataFrame...')
    formatted_df = pd.DataFrame()
    counts_formatted = []
    priors_formatted = defaultdict(list)
    for index, row in unfolding_df.iterrows():
        for composition in comp_list:
            counts_formatted.append(row['counts_{}'.format(composition)])
            for priors_name in priors_list:
                p = row['{}_flux_{}'.format(priors_name, composition)]
                priors_formatted[priors_name].append(p)

    formatted_df['counts'] = counts_formatted
    formatted_df['counts_err'] = np.sqrt(counts_formatted)

    formatted_df['efficiencies'] = efficiencies
    formatted_df['efficiencies_err'] = efficiencies_err

    for key, value in priors_formatted.iteritems():
        formatted_df[key+'_flux'] = value
        formatted_df[key+'_prior'] = formatted_df[key+'_flux'] / formatted_df[key+'_flux'].sum()

    formatted_df.index.rename('log_energy_bin_idx', inplace=True)

    prior_cols = [col for col in formatted_df.columns if 'prior' in col]
    prior_sums = formatted_df[prior_cols].sum()
    np.testing.assert_allclose(prior_sums, np.ones_like(prior_sums))

    formatted_df_outfile = os.path.join(
                            comp.paths.comp_data_dir, config, 'unfolding',
                            'unfolding-df_{}-groups.hdf'.format(num_groups))
    comp.check_output_dir(formatted_df_outfile)
    formatted_df.to_hdf(formatted_df_outfile, 'dataframe',
                        format='table', mode='w')

    print('Saving PyUnfold input ROOT file...')
    save_pyunfold_root_file(config, num_groups)
