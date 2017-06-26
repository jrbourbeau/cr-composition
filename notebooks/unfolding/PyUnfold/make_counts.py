#!/usr/bin/env python

import numpy as np
import pandas as pd
import ROOT
from ROOT import TH1F, TH2F, TNamed
from ROOT import gROOT, gSystem

import itertools
import argparse
import os
import re


if __name__ == "__main__":

    # p = argparse.ArgumentParser(description='Runs findblobs.py on cluster en masse')
    # p.add_argument('--fluxmodel', dest='fluxmodel',
    #                default='h4a',
    #                choices=['h4a', 'h3a', 'Hoerandel5'],
    #                help='Flux model to use for prior distribution')
    #
    # args = p.parse_args()
    # for fluxmodel in ['h4a', 'h3a', 'Hoerandel5']:
        # with open('/home/jbourbeau/cr-composition/analysis/pyunfold_dict.json') as data_file:
        #     data = json.load(data_file)
        #

    formatted_df = pd.read_csv('../formatted-dataframe.csv')
    counts = formatted_df['counts'].values

    ebins = len(counts)+1
    earray = np.arange(ebins,dtype=float)
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
    Eff = TH1F('Compositions','Non-Normed Combined Efficiency',ebins,earray)
    Eff.GetXaxis().SetTitle('Effects')
    Eff.GetYaxis().SetTitle('Counts')
    Eff.SetStats(0)
    Eff.Sumw2()

    for ci in xrange(0,ebins):
        print('counts[ci] = {}'.format(counts[ci]))
        Eff.SetBinContent(ci+1,counts[ci])
        Eff.SetBinError(ci+1,np.sqrt(counts[ci]))

    # Write the weighted histograms to file
    Eff.Write()

    fout.Write()
    fout.Close()
    print("Saving output file %s\n"%OutFile)

    print("\n=========================\n")
    print("Finished here! Exiting...")
