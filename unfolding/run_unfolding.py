#!/usr/bin/env python

from __future__ import division, print_function
import os
import argparse
import itertools
import numpy as np
from numpy.testing import assert_allclose
import pandas as pd
import pyprind
import pyunfold

import comptools as comp


def unfold(config_name=None, EffDist=None, priors='Jeffreys', input_file=None,
           ts='ks', ts_stopping=0.01, **kwargs):

    np.random.seed(2)

    if config_name is None:
        raise ValueError('config_name must be provided')

    assert input_file is not None

    # Get the Configuration Parameters from the Config File
    config = PyUnfold.Utils.ConfigFM(config_name)

    # Input Data ROOT File Name
    dataHeader = 'data'
    InputFile = input_file
    NE_meas_name = config.get(dataHeader, 'ne_meas', default='', cast=str)
    # isMC = config.get_boolean(dataHeader, 'ismc', default=False)

    # Analysis Bin
    binHeader = 'analysisbins'
    binnumberStr = config.get(binHeader, 'bin', default=0, cast=str)
    stackFlag = config.get_boolean(binHeader, 'stack', default=False)
    binList = ['bin'+val.replace(' ', '') for val in binnumberStr.split(',')]
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
            mess = '\n**** You have request to stack analysis bins, but have only requested %i bin, %s.'%(nStack, binList[0])
            mess += ' You need at least 2 bins to stack.\n'
            raise ValueError(mess+'\n\tPlease correct your mistake. Exiting... ***\n')
        unfbinname = 'bin0'

    # Unfolder Options
    unfoldHeader = 'unfolder'
    UnfolderName = config.get(unfoldHeader, 'unfolder_name',  default='Unfolder', cast=str)
    UnfMaxIter = config.get(unfoldHeader, 'max_iter', default=100, cast=int)
    UnfSmoothIter = config.get_boolean(unfoldHeader, 'smooth_with_reg', default=False)
    UnfVerbFlag = config.get_boolean(unfoldHeader, 'verbose', default=False)

    # Get Prior Definition
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
    tsname = ts
    tsTol = ts_stopping
    tsRange = [0, 1e2]
    tsVerbFlag = False
    # tsHeader = 'teststatistic'
    # tsRangeStr = config.get(tsHeader, 'ts_range', cast=str)
    # tsname = config.get(tsHeader, 'ts_name', default='rmd', cast=str)
    # tsTol = config.get(tsHeader, 'ts_tolerance', cast=float)
    # tsRange = [float(val) for val in tsRangeStr.split(',')]
    # tsVerbFlag = config.get_boolean(tsHeader, 'verbose', default=False)

    # Regularization Function, Initial Parameters, & Options
    regHeader = 'regularization'
    RegFunc = config.get(regHeader, 'reg_func', default='', cast=str)
    #  Param Names
    ConfigParamNames = config.get(regHeader, 'param_names', default='', cast=str)
    ParamNames = [x.strip() for x in ConfigParamNames.split(',')]
    #  Initial Parameters
    IPars = config.get(regHeader, 'param_init', default='', cast=str)
    InitParams = [float(val) for val in IPars.split(',')]
    #  Limits
    PLow = config.get(regHeader, 'param_lim_lo', default='', cast=str)
    PLimLo = [float(val) for val in PLow.split(',')]
    PHigh = config.get(regHeader, 'param_lim_hi', default='', cast=str)
    PLimHi = [float(val) for val in PHigh.split(',')]
    RegRangeStr = config.get(regHeader, 'reg_range', cast=str)
    RegRange = [float(val) for val in RegRangeStr.split(',')]
    #  Options
    RegPFlag = config.get_boolean(regHeader, 'plot', default=False)
    RegVFlag = config.get_boolean(regHeader, 'verbose', default=False)

    # Get MCInput
    mcHeader = 'mcinput'
    # StatsFile = config.get(mcHeader, 'stats_file', default='', cast=str)
    StatsFile = input_file
    Eff_hist_name = config.get(mcHeader, 'eff_hist', default='', cast=str)
    MM_hist_name = config.get(mcHeader, 'mm_hist', default='', cast=str)

    # Setup the Observed and MC Data Arrays
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
    Cxlab, Cylab, Ctitle = PyUnfold.rr.get_labels(StatsFile, Eff_hist_name, binList[0], verbose=False)

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
        n_c = n_c / np.sum(n_c)
    else:
        raise TypeError('priors must be a array_like, '
                        'but got {}'.format(type(priors)))
    assert_allclose(np.sum(n_c), 1)

    # Setup the Tools Used in Unfolding
    # Prepare Regularizer
    Rglzr = [PyUnfold.Utils.Regularizer('REG', FitFunc=[RegFunc], Range=RegRange,
                InitialParams=InitParams, ParamLo=PLimLo, ParamHi=PLimHi,
                ParamNames=ParamNames, xarray=Caxis[i], xedges=Cedges[i],
                verbose=RegVFlag, plot=RegPFlag) for i in range(nStack)]
    # Prepare Test Statistic-er
    tsMeth = PyUnfold.Utils.get_ts(tsname)
    tsFunc = [tsMeth(tsname, tol=tsTol, Xaxis=Caxis[i], TestRange=tsRange, verbose=tsVerbFlag)
              for i in range(nStack)]

    # Prepare Mixer
    Mixer = PyUnfold.Mix.Mixer(MixName, ErrorType=CovError, MCTables=MCStats,
                               EffectsDist=EffDist)

    # Unfolder!!!
    if stackFlag:
        UnfolderName += '_'+''.join(binList)

    Unfolder = PyUnfold.IterUnfold.IterativeUnfolder(
        UnfolderName, max_iter=UnfMaxIter, smooth_iter=UnfSmoothIter, n_c=n_c,
        mix_func=Mixer, reg_func=Rglzr, ts_func=tsFunc, stack=stackFlag,
        verbose=UnfVerbFlag)
    # Iterate the Unfolder
    unfolding_result = Unfolder.IUnfold()

    return unfolding_result


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Script to run an iterative '
                                    'Bayesian unfolding with PyUnfold')
    parser.add_argument('-c', '--config', dest='config',
                        default='IC86.2012',
                        choices=comp.simfunctions.get_sim_configs(),
                        help='Detector configuration')
    parser.add_argument('--num_groups', dest='num_groups', type=int,
                        default=4, choices=[2, 3, 4], help='Detector configuration')
    parser.add_argument('--config_file', dest='config_file',
                        help='Configuration file')
    parser.add_argument('--input_file', dest='input_file',
                        help='Input ROOT file')
    parser.add_argument('-o', '--outfile', dest='output_file',
                        help='Output DataFrame file')
    parser.add_argument('--ts_stopping', dest='ts_stopping', type=float,
                        default=0.005,
                        help='Testing statistic stopping condition')

    args = parser.parse_args()

    config = args.config
    num_groups = args.num_groups

    comp_list = comp.get_comp_list(num_groups=num_groups)
    energybins = comp.get_energybins(config)
    num_ebins = len(energybins.log_energy_midpoints)

    unfolding_dir  = os.path.join(comp.paths.comp_data_dir,
                                  config,
                                  'unfolding',
                                  )
    if not args.output_file:
        output_file = os.path.join(unfolding_dir,
                                   'pyunfold_output_{}-groups.hdf'.format(num_groups))

    # Load DataFrame with saved prior distributions
    df_file = os.path.join(unfolding_dir,
                           'unfolding-df_{}-groups.hdf'.format(num_groups))
    df = pd.read_hdf(df_file)

    # Load simulation and train composition classifier
    df_sim_train, df_sim_test = comp.load_sim(config=config,
                                              energy_reco=False,
                                              log_energy_min=None,
                                              log_energy_max=None,
                                              test_size=0.5,
                                              n_jobs=10,
                                              verbose=True)

    feature_list, feature_labels = comp.get_training_features()

    print('Running energy reconstruction...')
    energy_pipeline = comp.load_trained_model('RF_energy_{}'.format(config))
    for df in [df_sim_train, df_sim_test]:
        X = df_sim_train[feature_list].values
        # Energy reconstruction
        df['reco_log_energy'] = energy_pipeline.predict(df[feature_list].values)
        df['reco_energy'] = 10**df['reco_log_energy']


    efficiencies, efficiencies_err = comp.get_detector_efficiencies(config=config,
                                                                    num_groups=num_groups,
                                                                    sigmoid='slant',
                                                                    pyunfold_format=True)


    print('Running composition classifications...')
    pipeline_str = 'xgboost_comp_{}_{}-groups'.format(config, num_groups)
    comp_pipeline = comp.load_trained_model(pipeline_str)
    pred_target = comp_pipeline.predict(df[feature_list].values)

    print('Making response matrix...')
    log_reco_energy_sim_test = df_sim_test['reco_log_energy']
    log_true_energy_sim_test = df_sim_test['MC_log_energy']
    true_target = df_sim_test['comp_target_{}'.format(num_groups)].values

    response, response_err = comp.response_matrix(true_energy=log_true_energy_sim_test,
                                                  reco_energy=log_reco_energy_sim_test,
                                                  true_target=true_target,
                                                  pred_target=pred_target,
                                                  efficiencies=efficiencies,
                                                  efficiencies_err=efficiencies_err,
                                                  energy_bins=energybins.log_energy_bins)

    # Run analysis pipeline on data
    print('Loading data into memory...')
    df_data = comp.load_data(config=config,
                             energy_reco=False,
                             log_energy_min=None,
                             log_energy_max=None,
                             columns=feature_list,
                             n_jobs=10,
                             verbose=True)

    print('Running energy and composition reconstructions...')
    df_data['pred_comp_target'] = comp_pipeline.predict(df_data[feature_list].values)
    df_data['reco_log_energy'] = energy_pipeline.predict(df_data[feature_list].values)

    counts_observed = {}
    counts_observed_err = {}
    for idx, composition in enumerate(comp_list):
        # Filter out events that don't pass composition & energy mask
        pred_comp_mask = df_data['pred_comp_target'] == idx
        energies = df_data.loc[pred_comp_mask, 'reco_log_energy'].values
        comp_counts, _ = np.histogram(energies,
                                      bins=energybins.log_energy_bins)
        counts_observed[composition] = comp_counts
        counts_observed_err[composition] = np.sqrt(comp_counts)

    counts_observed_err['total'] = np.sqrt(np.sum(counts_observed_err[composition]**2 for composition in comp_list))
    # Calculate total counts
    counts_observed['total'] = np.sum(counts_observed[composition] for composition in comp_list)

    # Format observed counts, detection efficiencies, and priors for PyUnfold use
    counts_pyunfold = np.empty(num_groups * len(energybins.energy_midpoints))
    counts_err_pyunfold = np.empty(num_groups * len(energybins.energy_midpoints))
    for idx, composition in enumerate(comp_list):
        counts_pyunfold[idx::num_groups] = counts_observed[composition]
        counts_err_pyunfold[idx::num_groups] = counts_observed_err[composition]

    # Run unfolding for each of the priors
    names = ['uniform', 'H3a', 'H4a', 'Polygonato']
    # names = ['Jeffreys', 'H3a', 'H4a', 'Polygonato']
    logger = pyunfold.callbacks.Logger()
    for prior_name in pyprind.prog_bar(names):
        prior = None if prior_name == 'uniform' else df['{}_prior'.format(prior_name)]
        # priors = 'Jeffreys' if prior_name == 'Jeffreys' else df['{}_prior'.format(prior_name)]
        df_unfolding_iter = pyunfold.iterative_unfold(data=counts_pyunfold,
                                                      data_err=counts_err_pyunfold,
                                                      response=response,
                                                      response_err=response_err,
                                                      efficiencies=efficiencies,
                                                      efficiencies_err=efficiencies_err,
                                                      ts='ks',
                                                      ts_stopping=args.ts_stopping,
                                                      prior=prior,
                                                      return_iterations=True,
                                                      callbacks=[logger,])
        print(df_unfolding_iter)
        # # Save to hdf file
        # df_unfolding_iter.to_hdf(output_file, prior_name)
