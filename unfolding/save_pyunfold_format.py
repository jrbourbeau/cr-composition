#!/usr/bin/env python

from __future__ import division, print_function
import os
import argparse
from collections import defaultdict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn.apionly as sns
from sklearn.metrics import confusion_matrix
import ROOT
from ROOT import TH1F, TH2F, TNamed

from icecube.weighting.weighting import PDGCode
from icecube.weighting.fluxes import GaisserH3a, GaisserH4a, Hoerandel5

import comptools as comp

color_dict = comp.analysis.get_color_dict()


def response_matrix(log_true_energy_sim_test, log_reco_energy_sim_test,
                    true_comp, pred_comp, comp_list):

    true_ebin_idxs = np.digitize(log_true_energy_sim_test,
                                 energybins.log_energy_bins) - 1
    reco_ebin_idxs = np.digitize(log_reco_energy_sim_test,
                                 energybins.log_energy_bins) - 1

    hstack_list = []
    for true_ebin_idx in range(-1, len(energybins.log_energy_midpoints)+1):
        is_underflow = true_ebin_idx == -1
        is_overflow = true_ebin_idx == energybins.energy_midpoints.shape[0]
        if is_underflow or is_overflow:
            continue
        true_ebin_mask = true_ebin_idxs == true_ebin_idx

        vstack_list = []
        for reco_ebin_idx in range(-1, len(energybins.log_energy_midpoints)+1):
            is_underflow = reco_ebin_idx == -1
            is_overflow = reco_ebin_idx == energybins.energy_midpoints.shape[0]
            if is_underflow or is_overflow:
                continue
            reco_ebin_mask = reco_ebin_idxs == reco_ebin_idx

            combined_mask = true_ebin_mask & reco_ebin_mask
            if combined_mask.sum() == 0:
                response_mat = np.zeros((num_groups, num_groups), dtype=int)
            else:
                response_mat = confusion_matrix(
                                    true_comp[true_ebin_mask & reco_ebin_mask],
                                    pred_comp[true_ebin_mask & reco_ebin_mask],
                                    labels=comp_list)
            # Transpose response matrix to get MC comp on x-axis
            # and reco comp on y-axis
            response_mat = response_mat.T
            vstack_list.append(response_mat)
        hstack_list.append(np.vstack(vstack_list))

    res = np.hstack(hstack_list)
    res_err = np.sqrt(res)

    # Normalize response matrix column-wise (i.e. $P(E|C)$)
    res_col_sum = res.sum(axis=0)
    res_col_sum_err = np.array([np.sqrt(np.nansum(res_err[:, i]**2))
                                for i in range(res_err.shape[1])])

    normalizations, normalizations_err = comp.analysis.ratio_error(
                                            res_col_sum, res_col_sum_err,
                                            efficiencies, efficiencies_err,
                                            nan_to_num=True)

    res_normalized, res_normalized_err = comp.analysis.ratio_error(
                                            res, res_err,
                                            normalizations, normalizations_err,
                                            nan_to_num=True)

    res_normalized = np.nan_to_num(res_normalized)
    res_normalized_err = np.nan_to_num(res_normalized_err)

    # Test that the columns of res_normalized equal efficiencies
    np.testing.assert_allclose(res_normalized.sum(axis=0), efficiencies)

    return res_normalized, res_normalized_err


def save_pyunfold_root_file(config, num_groups, outfile=None, formatted_df_file=None):

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

    fout = ROOT.TFile(outfile , 'UPDATE')
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
    efficiencies = df_flux['efficiencies'].values
    efficiencies_err = df_flux['efficiencies_err'].values

    print('counts = {}'.format(counts))
    print('efficiencies = {}'.format(efficiencies))

    cbins = len(counts)+1
    carray = np.arange(cbins, dtype=float)
    print('carray = {}'.format(carray))

    ebins = len(counts)+1
    earray = np.arange(ebins, dtype=float)
    print('earray = {}'.format(earray))
    cbins -= 1
    ebins -= 1

    # Load response matrix array
    res_mat_file = os.path.join(unfolding_dir,
                                'response_{}-groups.txt'.format(num_groups))
    response_array = np.loadtxt(res_mat_file)
    res_mat_err_file = os.path.join(
                unfolding_dir, 'response_err_{}-groups.txt'.format(num_groups))
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
        ne_meas.SetBinError(ci+1, np.sqrt(counts[ci]))

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
    eff.Write()
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

    print('Saving output file {}'.format(outfile))

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
    energybins = comp.analysis.get_energybins(config=config)
    log_energy_min = energybins.log_energy_min
    log_energy_max = energybins.log_energy_max

    # Load simulation training/testing DataFrames
    print('Loading simulation training/testing DataFrames...')
    df_sim_train, df_sim_test = comp.load_sim(config=config,
                                              log_energy_min=log_energy_min,
                                              log_energy_max=log_energy_max)

    log_reco_energy_sim_test = df_sim_test['reco_log_energy']
    log_true_energy_sim_test = df_sim_test['MC_log_energy']

    feature_list, feature_labels = comp.analysis.get_training_features()
    pipeline_str = 'BDT_comp_{}_{}-groups'.format(config, num_groups)
    pipeline = comp.get_pipeline(pipeline_str)

    # Fit composition classifier
    print('Fitting composition classifier...')
    pipeline = pipeline.fit(df_sim_train[feature_list],
                            df_sim_train['comp_target_{}'.format(num_groups)])

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
                             n_jobs=15, verbose=True)

    X_data = comp.dataframe_functions.dataframe_to_array(
                        df_data, feature_list + ['reco_log_energy'])
    log_energy_data = X_data[:, -1]
    X_data = X_data[:, :-1]

    print('Making data predictions...')
    data_predictions = pipeline.predict(X_data)
    data_labels = np.array(comp.composition_encoding.decode_composition_groups(
                           data_predictions, num_groups=num_groups))

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
    test_predictions = pipeline.predict(df_sim_test[feature_list])
    true_comp = df_sim_test['comp_group_{}'.format(num_groups)].values
    pred_comp = np.array(comp.composition_encoding.decode_composition_groups(
                            test_predictions, num_groups=num_groups))
    res_normalized, res_normalized_err = response_matrix(
                                                     log_true_energy_sim_test,
                                                     log_reco_energy_sim_test,
                                                     true_comp, pred_comp,
                                                     comp_list)

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
    formatted_df.to_hdf(formatted_df_outfile, 'dataframe', format='table')

    print('Saving PyUnfold input ROOT file...')
    save_pyunfold_root_file(config, num_groups)
