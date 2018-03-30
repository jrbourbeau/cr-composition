
import os
from itertools import product
import numpy as np
import pandas as pd
import socket

on_submitter = 'submitter' in socket.gethostname()
cvmfs_root = 'cvmfs' in os.getenv('ROOTSYS', '')
if on_submitter or cvmfs_root:
    pass
else:
    try:
        from ROOT import TH1F, TH2F, TFile
        import pyunfold as PyUnfold
    except ImportError:
        print('Could not import ROOT / PyUnfold.')

from .composition_encoding import get_comp_list
from .base import get_energybins, get_paths, check_output_dir
from .data_functions import ratio_error


def unfolded_counts_dist(unfolding_df, iteration=-1, num_groups=4):
    """
    Convert unfolded distributions DataFrame from PyUnfold counts arrays
    to a dictionary containing a counts array for each composition.

    Parameters
    ----------
    unfolding_df : pandas.DataFrame
        Unfolding DataFrame returned from PyUnfold.
    iteration : int, optional
        Specific unfolding iteration to retrieve unfolded counts
        (default is -1, the last iteration).
    num_groups : int, optional
        Number of composition groups (default is 4).

    Returns
    -------
    counts : dict
        Dictionary with composition-counts key-value pairs.
    counts_sys_err : dict
        Dictionary with composition-systematic error key-value pairs.
    counts_stat_err : dict
        Dictionary with composition-statistical error key-value pairs.
    """
    comp_list = get_comp_list(num_groups=num_groups)

    df_iter = unfolding_df.iloc[iteration]

    counts, counts_sys_err, counts_stat_err = {}, {}, {}
    for idx, composition in enumerate(comp_list):
        counts[composition] = df_iter['n_c'][idx::num_groups]
        counts_sys_err[composition] = df_iter['sys_err'][idx::num_groups]
        counts_stat_err[composition] = df_iter['stat_err'][idx::num_groups]

    counts['total'] = np.sum([counts[composition] for composition in comp_list], axis=0)
    counts_sys_err['total'] = np.sqrt(np.sum([counts_sys_err[composition]**2 for composition in comp_list], axis=0))
    counts_stat_err['total'] = np.sqrt(np.sum([counts_stat_err[composition]**2 for composition in comp_list], axis=0))

    return counts, counts_sys_err, counts_stat_err


def column_normalize(res, res_err, efficiencies, efficiencies_err):
    res_col_sum = res.sum(axis=0)
    res_col_sum_err = np.array([np.sqrt(np.sum(res_err[:, i]**2))
                                for i in range(res_err.shape[1])])

    normalizations, normalizations_err = ratio_error(
                                            res_col_sum, res_col_sum_err,
                                            efficiencies, efficiencies_err,
                                            nan_to_num=True)

    res_normalized, res_normalized_err = ratio_error(
                                            res, res_err,
                                            normalizations, normalizations_err,
                                            nan_to_num=True)

    res_normalized = np.nan_to_num(res_normalized)
    res_normalized_err = np.nan_to_num(res_normalized_err)

    # Test that the columns of res_normalized equal efficiencies
    np.testing.assert_allclose(res_normalized.sum(axis=0), efficiencies)

    return res_normalized, res_normalized_err


def response_matrix(true_energy, reco_energy, true_target, pred_target,
                    energy_bins=None):
    """Computes energy-composition response matrix

    Parameters
    ----------
    true_energy : array_like
        Array of true (MC) energies.
    reco_energy : array_like
        Array of reconstructed energies.
    true_target : array_like
        Array of true compositions that are encoded to numerical values.
    pred_target : array_like
        Array of predicted compositions that are encoded to numerical values.
    energy_bins : array_like, optional
        Energy bins to be used for constructing response matrix (default is
        to use energy bins from comptools.get_energybins() function).

    Returns
    -------
    res : numpy.ndarray
        Response matrix.
    res_err : numpy.ndarray
        Uncerainty of the response matrix.
    """

    # Check that the input array shapes
    inputs = [true_energy, reco_energy, true_target, pred_target]
    assert len(set(map(np.ndim, inputs))) == 1
    assert len(set(map(np.shape, inputs))) == 1

    if energy_bins is None:
        energy_bins = get_energybins().energy_bins
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

    return res, res_err


def normalized_response_matrix(true_energy, reco_energy, true_target,
                               pred_target, efficiencies, efficiencies_err,
                               energy_bins=None):
    """Computes normalized energy-composition response matrix

    Parameters
    ----------
    true_energy : array_like
        Array of true (MC) energies.
    reco_energy : array_like
        Array of reconstructed energies.
    true_target : array_like
        Array of true compositions that are encoded to numerical values.
    pred_target : array_like
        Array of predicted compositions that are encoded to numerical values.
    efficiencies : array_like
        Detection efficiencies (should be in a PyUnfold-compatable form).
    efficiencies_err : array_like
        Detection efficiencies uncertainties (should be in a
        PyUnfold-compatable form).
    energy_bins : array_like, optional
        Energy bins to be used for constructing response matrix (default is
        to use energy bins from comptools.get_energybins() function).

    Returns
    -------
    res_normalized : numpy.ndarray
        Normalized response matrix.
    res_normalized_err : numpy.ndarray
        Uncerainty of the normalized response matrix.
    """
    res, res_err = response_matrix(true_energy=true_energy,
                                   reco_energy=reco_energy,
                                   true_target=true_target,
                                   pred_target=pred_target,
                                   energy_bins=energy_bins)

    # Normalize response matrix column-wise (i.e. $P(E|C)$)
    res_normalized, res_normalized_err = column_normalize(res=res,
                                                          res_err=res_err,
                                                          efficiencies=efficiencies,
                                                          efficiencies_err=efficiencies_err)

    return res_normalized, res_normalized_err


def save_pyunfold_root_file(config, num_groups=4, outfile=None,
                            formatted_df_file=None, res_mat_file=None,
                            res_mat_err_file=None):

    paths = get_paths()

    unfolding_dir  = os.path.join(paths.comp_data_dir, config, 'unfolding')
    # Bin Definitions
    binname = 'bin0'
    # ROOT Output
    if outfile is None:
        outfile  = os.path.join(unfolding_dir,
                                'pyunfold_input_{}-groups.root'.format(num_groups))
    check_output_dir(outfile)
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

        # Fill response matrix entries
        for ek in range(0, ebins):
            response.SetBinContent(ci+1, ek+1, response_array[ek][ci])
            response.SetBinError(ci+1, ek+1, response_err_array[ek][ci])

        # Fill efficiency histogram from response matrix
        eff.SetBinContent(ci+1, efficiencies[ci])
        eff.SetBinError(ci+1, efficiencies_err[ci])

    # Write measured effects histogram to file
    ne_meas.Write()
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

    fout.Write()
    fout.Close()


def unfold(config_name=None, EffDist=None, priors='Jeffreys', input_file=None,
           ts='ks', ts_stopping=0.01, **kwargs):

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
    else:
        raise TypeError('priors must be a np.ndarray, '
                        'but got {}'.format(type(priors)))

    # Setup the Tools Used in Unfolding
    # Prepare Regularizer
    Rglzr = [PyUnfold.Utils.Regularizer('REG', FitFunc=[RegFunc], Range=RegRange,
                InitialParams=InitParams, ParamLo=PLimLo, ParamHi=PLimHi,
                ParamNames=ParamNames, xarray=Caxis[i], xedges=Cedges[i],
                verbose=RegVFlag, plot=RegPFlag) for i in range(nStack)]
    # Prepare Test Statistic-er
    tsMeth = PyUnfold.Utils.GetTS(tsname)
    tsFunc = [tsMeth(tsname, tol=tsTol, Xaxis=Caxis[i], TestRange=tsRange, verbose=tsVerbFlag)
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
