#!/usr/bin/env python

import sys
import os
import argparse
import numpy as np
import pandas as pd
import pyprind
import ROOT

import PyUnfold
import comptools as comp


def unfold(config_name=None, return_dists=False, EffDist=None, plot_local=False,
           priors='Jeffreys', input_file=None, **kwargs):

    if config_name is None:
        raise ValueError('config_name must be provided')

    assert input_file is not None

    #### Get the Configuration Parameters from the Config File ####
    config = PyUnfold.Utils.ConfigFM(config_name)

    # Input Data ROOT File Name
    dataHeader = 'data'
    InputFile = input_file
    NE_meas_name = config.get(dataHeader, 'ne_meas', default='', cast=str)
    # isMC = config.get_boolean(dataHeader, 'ismc', default=False)

    # Analysis Bin
    binHeader = 'analysisbins'
    binnumberStr = config.get(binHeader,'bin',default=0,cast=str)
    stackFlag = config.get_boolean(binHeader,'stack',default=False)
    binList = ['bin'+val.replace(' ','') for val in binnumberStr.split(',')]
    nStack = len(binList)

    binnumber = 0
    if stackFlag is False:
        try:
            binnumber = np.int(binnumberStr)
        except:
            raise ValueError('\n\n*** You have requested more than 1 analysis bin without stacking. \n\tPlease fix your mistake. Exiting... ***\n')
        unfbinname = 'bin%i'%binnumber
    else:
        if nStack<=1:
            mess = '\n**** You have request to stack analysis bins, but have only requested %i bin, %s.'%(nStack,binList[0])
            mess += ' You need at least 2 bins to stack.\n'
            raise ValueError(mess+'\n\tPlease correct your mistake. Exiting... ***\n')
        unfbinname = 'bin0'

    # Unfolder Options
    unfoldHeader = 'unfolder'
    UnfolderName = config.get(unfoldHeader,'unfolder_name',default='Unfolder',cast=str)
    UnfMaxIter = config.get(unfoldHeader,'max_iter',default=100,cast=int)
    UnfSmoothIter = config.get_boolean(unfoldHeader,'smooth_with_reg',default=False)
    UnfVerbFlag = config.get_boolean(unfoldHeader,'verbose',default=False)

    # # Get Prior Definition
    # priorHeader = 'prior'
    # priorString = config.get(priorHeader,'func',default='Jeffreys',cast=str)
    # priorList = [val for val in priorString.split(',')]
    # nPriorFuncs = len(priorList)
    # if nPriorFuncs != nStack:
    #     if nPriorFuncs == 1:
    #         for i in range(nStack-1):
    #             priorList.append(priorList[0])
    #     else:
    #         mess = '\n**** You have requested an incorrect number of prior functions (%i), but there are %i analysis bins.'%(nPriorFuncs,nStack)
    #         raise ValueError(mess+'\n\tPlease correct your mistake. Exiting... ***\n')

    # Mixer Name and Error Propagation Type
    # Options: ACM, DCM
    mixHeader = 'mixer'
    MixName = config.get(mixHeader, 'mix_name', default='', cast=str)
    CovError = config.get(mixHeader, 'error_type', default='', cast=str)

    # Test Statistic - Stat Function & Options
    # Options: chi2, rmd, pf, ks
    tsHeader = 'teststatistic'
    tsname = config.get(tsHeader, 'ts_name', default='rmd', cast=str)
    tsTol = config.get(tsHeader, 'ts_tolerance', cast=float)
    tsRangeStr = config.get(tsHeader, 'ts_range', cast=str)
    tsRange = [float(val) for val in tsRangeStr.split(',')]
    tsVerbFlag = config.get_boolean(tsHeader, 'verbose', default=False)

    # Regularization Function, Initial Parameters, & Options
    regHeader = 'regularization'
    RegFunc = config.get(regHeader,'reg_func',default='',cast=str)
    #  Param Names
    ConfigParamNames = config.get(regHeader,'param_names',default='',cast=str)
    ParamNames = [x.strip() for x in ConfigParamNames.split(',')]
    #  Initial Parameters
    IPars = config.get(regHeader,'param_init',default='',cast=str)
    InitParams = [float(val) for val in IPars.split(',')]
    #  Limits
    PLow = config.get(regHeader,'param_lim_lo',default='',cast=str)
    PLimLo = [float(val) for val in PLow.split(',')]
    PHigh = config.get(regHeader,'param_lim_hi',default='',cast=str)
    PLimHi = [float(val) for val in PHigh.split(',')]
    RegRangeStr = config.get(regHeader,'reg_range',cast=str)
    RegRange = [float(val) for val in RegRangeStr.split(',')]
    #  Options
    RegPFlag = config.get_boolean(regHeader,'plot',default=False)
    RegVFlag = config.get_boolean(regHeader,'verbose',default=False)

    # Get MCInput
    mcHeader = 'mcinput'
    # StatsFile = config.get(mcHeader, 'stats_file', default='', cast=str)
    StatsFile = input_file
    Eff_hist_name = config.get(mcHeader, 'eff_hist', default='', cast=str)
    MM_hist_name = config.get(mcHeader, 'mm_hist', default='', cast=str)

    #### Setup the Observed and MC Data Arrays ####
    # Load MC Stats (NCmc), Cause Efficiency (Eff) and Migration Matrix ( P(E|C) )
    MCStats = PyUnfold.LoadStats.MCTables(StatsFile, BinName=binList,
        RespMatrixName=MM_hist_name, EffName=Eff_hist_name, Stack=stackFlag)
    Caxis = []
    Cedges = []
    cutList = []
    for index in range(nStack):
        axis, edge = MCStats.GetCauseAxis(index)
        Caxis.append(axis)
        Cedges.append(edge)
    Eaxis, Eedges = MCStats.GetEffectAxis()
    # Effect and Cause X and Y Labels from Respective Histograms
    Cxlab, Cylab, Ctitle = PyUnfold.rr.get_labels(StatsFile,Eff_hist_name,binList[0],verbose=False)

    # Load the Observed Data (n_eff), define total observed events (n_obs)
    # Get from ROOT input file if requested
    if EffDist is None:
        Exlab, Eylab, Etitle = PyUnfold.rr.get_labels(InputFile, NE_meas_name,
                                             unfbinname, verbose=False)
        Eaxis, Eedges, n_eff, n_eff_err = PyUnfold.rr.get1d(InputFile, NE_meas_name,
                                                   unfbinname)
        EffDist = PyUnfold.Utils.DataDist(Etitle, data=n_eff, error=n_eff_err,
                                 axis=Eaxis, edges=Eedges, xlabel=Exlab,
                                 ylabel=Eylab, units='')
    Exlab = EffDist.xlab
    Eylab = EffDist.ylab
    Etitle = EffDist.name
    n_eff = EffDist.getData()
    n_eff_err = EffDist.getError()
    n_obs = np.sum(n_eff)

    # Initial best guess (0th prior) expected prob dist (default: Jeffrey's Prior)
    if isinstance(priors, (list, tuple, np.ndarray, pd.Series)):
        n_c = np.asarray(priors)
    elif priors == 'Jeffreys':
        n_c = PyUnfold.Utils.UserPrior(['Jeffreys'], Caxis, n_obs)
    else:
        raise TypeError('priors must be a np.ndarray, '
                        'but got {}'.format(type(priors)))

    #### Setup the Tools Used in Unfolding ####
    # Prepare Regularizer
    Rglzr = [PyUnfold.Utils.Regularizer('REG', FitFunc=[RegFunc], Range=RegRange,
                InitialParams=InitParams, ParamLo=PLimLo, ParamHi=PLimHi,
                ParamNames=ParamNames, xarray=Caxis[i], xedges=Cedges[i],
                verbose=RegVFlag, plot=RegPFlag) for i in range(nStack)]
    # Prepare Test Statistic-er
    tsMeth = PyUnfold.Utils.GetTS(tsname)
    tsFunc = [tsMeth(tsname, tol=tsTol, Xaxis=Caxis[i], TestRange=tsRange ,verbose=tsVerbFlag)
                for i in range(nStack)]

    # Prepare Mixer
    Mixer = PyUnfold.Mix.Mixer(MixName, ErrorType=CovError, MCTables=MCStats,
                               EffectsDist=EffDist)

    # Unfolder!!!
    if stackFlag:
        UnfolderName += '_'+''.join(binList)

    Unfolder = PyUnfold.IterUnfold.IterativeUnfolder(
        UnfolderName, maxIter=UnfMaxIter, smoothIter=UnfSmoothIter, n_c=n_c,
        MixFunc=Mixer, RegFunc=Rglzr, TSFunc=tsFunc, Stack=stackFlag,
        verbose=UnfVerbFlag)
    # Iterate the Unfolder
    unfolding_result = Unfolder.IUnfold()

    return unfolding_result


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Script to run an iterative '
                                    'Bayesian unfolding with PyUnfold')
    parser.add_argument('-c', '--config', dest='config',
                   choices=comp.datafunctions.get_data_configs(),
                   help='Detector configuration')
    parser.add_argument('--num_groups', dest='num_groups', type=int,
                   default=2, choices=[2, 3, 4], help='Detector configuration')
    parser.add_argument('--config_file', dest='config_file',
                   help='Configuration file')
    parser.add_argument('--input_file', dest='input_file',
                   help='Input ROOT file')
    parser.add_argument('-o', '--outfile', dest='output_file',
                   help='Output DataFrame file')

    args = parser.parse_args()

    if not args.output_file:
        args.output_file = os.path.join(args.config, 'pyunfold_output_{}-groups.hdf'.format(args.num_groups))
        print('Writing to output file: {}'.format(args.output_file))
    if not args.config_file:
        args.config_file = os.path.join(args.config, 'config.cfg')
        print('Using config file: {}'.format(args.config_file))
    if not args.input_file:
        args.input_file = os.path.join(args.config,
                        'pyunfold_input_{}-groups.root'.format(args.num_groups))
        print('Using input ROOT file: {}'.format(args.input_file))

    # Load DataFrame with saved prior distributions
    df_file = os.path.join(comp.paths.comp_data_dir, args.config, 'unfolding',
                           'unfolding-df_{}-groups.hdf'.format(args.num_groups))
    df = pd.read_hdf(df_file)

    # Run unfolding for each of the priors
    names = ['Jeffreys', 'H3a', 'H4a', 'Polygonato']
    for prior_name in pyprind.prog_bar(names):
        priors = 'Jeffreys' if prior_name == 'Jeffreys' else df['{}_prior'.format(prior_name)]
        df_unfolding_iter = unfold(config_name=args.config_file, priors=priors, input_file=args.input_file)
        # Save to hdf file
        df_unfolding_iter.to_hdf(args.output_file, prior_name)
