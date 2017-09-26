#!/usr/bin/env python
"""
   Detector response script.
   Extracts config-file specified
   MC True and Reco variables, filling
   effective area and normalized probability
   matrices for each species, and isotropically
   weighted efficiency and mixing matrices
   for later-normalization based on MC model.
   Event cuts supplied on command line.
   Outputs ROOT file with above mentioned
   histograms.

.. codeauthor: Zig Hampel
"""

__version__ = "$Id"

try:
    import numpy as np
    import pandas as pd
    import ROOT
    from ROOT import TH1F, TH2F, TNamed
    from ROOT import gROOT, gSystem

    import itertools
    import os
    import re

except ImportError as e:
    print e
    raise ImportError


ROOT.gROOT.Reset()
ROOT.gROOT.SetBatch(True)
ROOT.gErrorIgnoreLevel=ROOT.kWarning

# Global program options
args = None

def main():


    EffName = "Eff"
    WMMName = "MM"

    # Bin Definitions
    binHeader = "bindefs"
    # binname = fluxmodel
    binname = "bin0"

    # ROOT Output
    OutFile = "response_matrix.root"
    # fout = ROOT.TFile(OutFile, "RECREATE")
    fout = ROOT.TFile(OutFile, "UPDATE")
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

    # formatted_df = pd.read_csv('../formatted-dataframe.csv')
    formatted_df_outfile = os.path.join('/data/user/jbourbeau/composition/unfolding',
                                        'unfolding-dataframe-PyUnfold-formatted.csv')
    df_flux = pd.read_csv(formatted_df_outfile, index_col='log_energy_bin_idx')
    counts = df_flux['counts'].values

    res_mat_file = os.path.join('/data/user/jbourbeau/composition/unfolding',
                                'response.txt')
    # res_mat_file = os.path.join('/data/user/jbourbeau/composition/unfolding',
    #                             'block_response.txt')
    block_response = np.loadtxt(res_mat_file)
    res_mat_err_file = os.path.join('/data/user/jbourbeau/composition/unfolding',
                                    'response_err.txt')
    # res_mat_err_file = os.path.join('/data/user/jbourbeau/composition/unfolding',
    #                                 'block_response_err.txt')
    block_response_err = np.loadtxt(res_mat_err_file)

    cbins = len(counts)+1
    carray = np.arange(cbins, dtype=float)
    print('carray = {}'.format(carray))
    print('cbins = {}'.format(cbins))

    ebins = len(counts)+1
    earray = np.arange(ebins, dtype=float)
    print('earray = {}'.format(earray))
    cbins -= 1
    ebins -= 1

    # Prepare Combined Weighted Histograms - To be Normalized by Model After Filling
    # Isotropic Weights of Causes - For Calculating Combined Species Efficiency
    Eff = TH1F("%s"%(EffName), 'Non-Normed Combined Efficiency', cbins, carray)
    Eff.GetXaxis().SetTitle('Causes')
    Eff.GetYaxis().SetTitle("Efficiency")
    Eff.SetStats(0)
    Eff.Sumw2()
    # Isotropic Weighted Mixing Matrix - For Calculating Combined Species MM
    WMM = TH2F("%s"%(WMMName), 'Weighted Combined Mixing Matrix',
               cbins, carray, ebins, earray)
    WMM.GetXaxis().SetTitle('Causes')
    WMM.GetYaxis().SetTitle('Effects')
    WMM.SetStats(0)
    WMM.Sumw2()
    # # Isotropic Weighted Mixing Matrix - For Calculating Combined Species MM
    # ModelMM = TH2F("Model_%s"%(WMMName),'Model Weighted Combined Matrix',cbins,carray,ebins,earray)
    # ModelMM.GetXaxis().SetTitle('Causes')
    # ModelMM.GetYaxis().SetTitle('Effects')
    # ModelMM.SetStats(0)
    # ModelMM.Sumw2()
    # Raw Number of Events in Each Bin
    NMM = TH2F("NMM", 'Number of Events Matrix', cbins, carray, ebins, earray)
    NMM.GetXaxis().SetTitle('Causes')
    NMM.GetYaxis().SetTitle('Effects')
    NMM.SetStats(0)
    NMM.Sumw2()


    throwArea = 1.
    for ci in xrange(0,cbins):
        # Calculate Flux-Weighted Efficiency
        # Eval = Eff.GetBinContent(ci+1)/binwidth[ci]/throwArea
        # dEval = Eff.GetBinError(ci+1)/binwidth[ci]/throwArea
        dEval, Eval = 1, 1
        Eff.SetBinContent(ci+1, Eval)
        # Eff.SetBinError(ci+1, dEval)

        # Normalize Species Probability Matrix
        sum2 = 0
        for ek in xrange(0,ebins):
            # Calculate Flux-Weighted Mixing Matrix
            # wmm_val = WMM.GetBinContent(ci+1,ek+1)/binwidth[ci]/throwArea
            # dwmm_val = WMM.GetBinError(ci+1,ek+1)/binwidth[ci]/throwArea

            # wmm_val, dwmm_val =  block_response[ebins-ek-1][ci], block_response_err[ebins-ek-1][ci]
            wmm_val, dwmm_val =  block_response[ek][ci], block_response_err[ek][ci]

            sum2 += dwmm_val**2
            WMM.SetBinContent(ci+1, ek+1, wmm_val)
            WMM.SetBinError(ci+1, ek+1, dwmm_val)

        Eff.SetBinError(ci+1, np.sqrt(sum2))

    # Write model name to file
    modelName = 'whatever'
    MODELNAME = TNamed("ModelName",modelName)
    MODELNAME.Write()
    # Write Cuts to file
    cuts = ''
    cuts_theta = cuts.replace("rec.zenithAngle","theta")
    CUTS = TNamed("cuts",cuts_theta)
    CUTS.Write()
    # Write the cause and effect arrays to file
    CARRAY = TH1F("CARRAY","Cause Array", cbins, carray)
    CARRAY.GetXaxis().SetTitle('Causes')
    EARRAY = TH1F("EARRAY","Effect Array", ebins, earray)
    EARRAY.GetXaxis().SetTitle('Effects')
    CARRAY.Write()
    EARRAY.Write()

    # Write the weighted histograms to file
    addtitle = r'Test'
    WMM.SetTitle(WMM.GetTitle()+addtitle)
    # ModelMM.SetTitle(ModelMM.GetTitle()+addtitle)
    Eff.SetTitle(Eff.GetTitle()+addtitle)
    WMM.Write()
    Eff.Write()
    # ModelMM.Write()
    NMM.Write()

    fout.Write()
    fout.Close()
    print("Saving output file %s\n"%OutFile)

    print("\n=========================\n")
    print("Finished here! Exiting...")


if __name__ == "__main__":

    main()
