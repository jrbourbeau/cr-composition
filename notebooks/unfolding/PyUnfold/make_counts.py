#!/usr/bin/env python

import numpy as np
import pandas as pd
import ROOT
from ROOT import TH1F, TH2F, TNamed
from ROOT import gROOT, gSystem

import itertools
import os
import re


if __name__ == "__main__":

    formatted_df_outfile = os.path.join('/data/user/jbourbeau/composition/unfolding',
                                        'unfolding-dataframe-PyUnfold-formatted.csv')
    df_flux = pd.read_csv(formatted_df_outfile, index_col='log_energy_bin_idx')
    counts = df_flux['counts'].values

    ebins = len(counts)+1
    earray = np.arange(ebins, dtype=float)
    print('earray = {}'.format(earray))
    ebins -= 1

    # ROOT Output
    binname = 'bin0'
    OutFile = 'counts.root'
    fout = ROOT.TFile(OutFile, "RECREATE")
    # Check if bin directory exists, quit if so, otherwise create it!
    if ( not fout.GetDirectory(binname) ):
        pdir = fout.mkdir(binname,"Bin number 0")
    else:
        fout.Close()
        print("\n=========================\n")
        errormessage = "Directory %s already exists!\nEither try another bin number or delete %s and start again. Exiting...\n"%(binname,OutFile)
        raise ValueError(errormessage)

    # Go to home of ROOT file
    fout.cd(binname)

    # Prepare Combined Weighted Histograms - To be Normalized by Model After Filling
    # Isotropic Weights of Causes - For Calculating Combined Species Efficiency
    Eff = TH1F('Compositions', 'Non-Normed Combined Efficiency', ebins, earray)
    Eff.GetXaxis().SetTitle('Effects')
    Eff.GetYaxis().SetTitle('Counts')
    Eff.SetStats(0)
    Eff.Sumw2()

    for ci in xrange(0,ebins):
        print('counts[{}] = {}'.format(ci, counts[ci]))
        Eff.SetBinContent(ci+1, counts[ci])
        Eff.SetBinError(ci+1, np.sqrt(counts[ci]))

    # Write the weighted histograms to file
    Eff.Write()

    fout.Write()
    fout.Close()
    print("Saving output file %s\n"%OutFile)

    print("\n=========================\n")
    print("Finished here! Exiting...")
