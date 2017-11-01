#!/usr/bin/env python

import os
import argparse
import numpy as np
import pandas as pd
import ROOT
from ROOT import TH1F, TH2F, TNamed

import comptools as comp


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Saves trained model for later use')
    parser.add_argument('-c', '--config', dest='config',
                   choices=comp.datafunctions.get_data_configs(),
                   help='Detector configuration')
    args = parser.parse_args()

    # Bin Definitions
    binname = 'bin0'
    # ROOT Output
    outfile  = '{}/pyunfold_input.root'.format(args.config)
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

    formatted_df_outfile  = os.path.join(comp.paths.comp_data_dir, 'unfolding',
                                         args.config,
                                         'unfolding-dataframe-PyUnfold-formatted.csv')
    df_flux = pd.read_csv(formatted_df_outfile, index_col='log_energy_bin_idx')
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
    res_mat_file = os.path.join(comp.paths.comp_data_dir, 'unfolding',
                                args.config, 'response.txt')
    response_array = np.loadtxt(res_mat_file)
    res_mat_err_file = os.path.join(comp.paths.comp_data_dir, 'unfolding',
                                    args.config, 'response_err.txt')
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
