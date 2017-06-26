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
    import argparse
    import os
    import re

    # from Utils.config import Configurator

except ImportError as e:
    print e
    raise ImportError


ROOT.gROOT.Reset()
ROOT.gROOT.SetBatch(True)
ROOT.gErrorIgnoreLevel=ROOT.kWarning

# photospline table reader and eval func
# from photospline.glam.glam import grideval
# from photospline import splinefitstable

# Extract Zenith Limits from Cut String
# def getZenithLimits(cuts):
#
#     # Grab zenith angle cut
#     if "rec.zenithAngle" not in cuts:
#         raise ValueError("\nNeed to define some zenithAngle cut.\nPlease, fix your mistake. Exiting...\n")
#
#     zcutstring = re.finditer("rec.zenithAngle([<>=]*)([0-9.e-]*)", cuts)
#     zcuts = []
#     for izcut in zcutstring:
#         comp, val = izcut.groups()
#         zcuts.append(val)
#     zcuts = np.asarray(zcuts,dtype=float)
#
#     if len(zcuts)>2:
#         print("\n=========================\n")
#         raise ValueError("Too many zenith cuts! There can only be one (or two)!\nPlease, fix your mistake. Exiting...\n")
#
#     zlo = np.min(zcuts)
#     zhi = np.max(zcuts)
#
#     if len(zcuts) == 1:
#         zlo = 0
#
#     return zlo, zhi


# class DataCutter:
#     def __init__(self,name="",useSpline=False,splinefile="",xfuncstring="x",yfuncstring="x",evalfuncstring=""):
#         self.name = name
#         self.file = splinefile
#         self.useSpline = useSpline
#         self.grideval = []
#
#         # Define the execution of x and eval functions
#         xfuncstring = "def xfunc(x): return %s"%xfuncstring
#         yfuncstring = "def yfunc(x): return %s"%yfuncstring
#         evalfuncstring =  "def evalfunc(ysp,ydata): return (%s)"%evalfuncstring
#         self.xfuncstring = xfuncstring
#         self.yfuncstring = yfuncstring
#         self.evalfuncstring = evalfuncstring
#
#         self.xfunc = []
#         self.yfunc = []
#         self.evalfunc = []#evalfunc
#         self.table = self.loadSplineTable(splinefile)
#         self.cutFunc = []
#         self.defineCutFunc()
#
#     # Load the spline table
#     def loadSplineTable(self,spFile):
#
#         table = None
#
#         print "\n======================================="
#         if self.useSpline:
#             print("\nUsing spline table for cut function.\n")
#             if os.path.exists(spFile):
#                 table = splinefitstable.read(spFile)
#
#                 print "\nSpline file: %s\n"%spFile
#                 print "\nDefining spline evaluation functions\n"
#                 print "\tIndependent variable\n\t\t", self.xfuncstring
#                 print "\tDependent variable\n\t\t", self.yfuncstring
#                 print "\tSpline conditional\n\t\t", self.evalfuncstring
#                 exec self.xfuncstring
#                 exec self.yfuncstring
#                 exec self.evalfuncstring
#                 self.xfunc = xfunc
#                 self.yfunc = yfunc
#                 self.evalfunc = evalfunc
#
#             else:
#                 print("Spline file %s not found. Exiting...")
#                 import sys
#                 sys.exit(0)
#         else:
#             print("\nNot using spline table for cut function.\n")
#
#         print "\n======================================="
#
#         return table
#
#     def defineCutFunc(self):
#         if (self.useSpline):
#             self.cutFunc = self.splineFunc
#         else:
#             self.cutFunc = self.contFunc
#
#     def contFunc(self,x,y):
#         return False
#
#     def splineFunc(self,x,y):
#         x = np.ones(1)*self.xfunc(x)
#         y = self.yfunc(y)
#         Continue = True
#         spEval = grideval(self.table,[x])
#         if (self.evalfunc(spEval[0],y)):
#             Continue = False
#         return Continue
#
# Global program options
args = None

def main():

    # global args

    # p = argparse.ArgumentParser(description="Script to Extract MC for a Detector Response")
    # p.add_argument("-b", "--binnumber", dest="binnumber", type=int, required=False, help="Bin Number")
    # p.add_argument("-m", "--modelname", dest="modelName", default="", help="Model Name")
    # args = p.parse_args()
    # usage = "%prog -c config_file"
    #
    # #### Get the Configuration Parameters from the Config File ####
    # config = Configurator.ConfigFM(args.config_name)

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

    formatted_df = pd.read_csv('../formatted-dataframe.csv')
    counts = formatted_df['counts'].values
    block_response = np.loadtxt('../block_response.txt')
    block_response_err = np.loadtxt('../block_response_err.txt')

    cbins = len(counts)+1
    # cbins = len(counts)+1
    carray = np.arange(cbins,dtype=float)
    print('carray = {}'.format(carray))
    print('cbins = {}'.format(cbins))

    ebins = len(counts)+1
    # ebins = len(counts)+1
    earray = np.arange(ebins,dtype=float)
    print('earray = {}'.format(earray))
    cbins -= 1
    ebins -= 1

    # Prepare Combined Weighted Histograms - To be Normalized by Model After Filling
    # Isotropic Weights of Causes - For Calculating Combined Species Efficiency
    Eff = TH1F("%s"%(EffName),'Non-Normed Combined Efficiency',cbins,carray)
    Eff.GetXaxis().SetTitle('Causes')
    Eff.GetYaxis().SetTitle("Efficiency")
    Eff.SetStats(0)
    Eff.Sumw2()
    # Isotropic Weighted Mixing Matrix - For Calculating Combined Species MM
    WMM = TH2F("%s"%(WMMName),'Weighted Combined Mixing Matrix',cbins,carray,ebins,earray)
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
    NMM = TH2F("NMM",'Number of Events Matrix',cbins,carray,ebins,earray)
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
        Eff.SetBinContent(ci+1,Eval)
        # Eff.SetBinError(ci+1,dEval)

        # Normalize Species Probability Matrix
        sum2 = 0
        for ek in xrange(0,ebins):
            # Calculate Flux-Weighted Mixing Matrix
            # wmm_val = WMM.GetBinContent(ci+1,ek+1)/binwidth[ci]/throwArea
            # dwmm_val = WMM.GetBinError(ci+1,ek+1)/binwidth[ci]/throwArea
            wmm_val, dwmm_val =  block_response[ebins-ek-1][ci], block_response_err[ebins-ek-1][ci]
            sum2 += dwmm_val**2
            WMM.SetBinContent(ci+1,ek+1,wmm_val)
            WMM.SetBinError(ci+1,ek+1,dwmm_val)

        Eff.SetBinError(ci+1,np.sqrt(sum2))

    # for ek in xrange(0,ebins):
    #     col_sum = 0
    #     for ci in xrange(0,cbins):
    #         col_sum += ModelMM.GetBinContent(ci+1,ek+1)
    #     for ci in xrange(0,cbins):
    #         if (col_sum>0.):
    #             model_val = ModelMM.GetBinContent(ci+1,ek+1)
    #             dmodel_val = ModelMM.GetBinError(ci+1,ek+1)
    #             ModelMM.SetBinContent(ci+1,ek+1,model_val/col_sum)
    #             ModelMM.SetBinError(ci+1,ek+1,dmodel_val/col_sum)

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
    CARRAY = TH1F("CARRAY","Cause Array",cbins,carray)
    CARRAY.GetXaxis().SetTitle('Causes')
    EARRAY = TH1F("EARRAY","Effect Array",ebins,earray)
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
    import sys
    # sys.exit()


if __name__ == "__main__":

    # p = argparse.ArgumentParser(description='Runs findblobs.py on cluster en masse')
    # p.add_argument('--fluxmodel', dest='fluxmodel',
    #                default='h4a',
    #                choices=['h4a', 'h3a', 'Hoerandel5'],
    #                help='Flux model to use for prior distribution')
    #
    # args = p.parse_args()

    main()
