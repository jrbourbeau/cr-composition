
from __future__ import division
from collections import defaultdict
from dask import delayed, multiprocessing, compute
from dask.diagnostics import ProgressBar
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
import pyprind
from ..dataframe_functions import label_to_comp
from .base import get_energybins
from .data_functions import ratio_error
from .pipelines import get_pipeline


def get_frac_correct(df_train, df_test, feature_columns, pipeline, comp_list,
                    log_energy_bins=get_energybins().log_energy_bins):
    '''Calculates the fraction of correctly identified samples in each energy bin
    for each composition in comp_list. In addition, the statisitcal error for the
    fraction correctly identified is calculated.'''

    # Validate input
    assert isinstance(df_train, pd.DataFrame), 'df_train dataset must be a pandas DataFrame'
    assert isinstance(df_test, pd.DataFrame), 'df_test dataset must be a pandas DataFrame'

    # Fit pipeline and get mask for correctly identified events
    pipeline.fit(df_train[feature_columns], df_train.target)
    test_predictions = pipeline.predict(df_test[feature_columns])
    correctly_identified_mask = (test_predictions == df_test.target)

    # Construct MC composition masks
    MC_comp_mask = {}
    for composition in comp_list:
        MC_comp_mask[composition] = (df_test.target.apply(label_to_comp) == composition)
    MC_comp_mask['total'] = pd.Series([True]*len(df_test))

    frac_correct, frac_correct_err = {}, {}
    for composition in comp_list+['total']:
        comp_mask = MC_comp_mask[composition]
        # Get number of MC comp in each reco energy bin
        num_MC_energy = np.histogram(df_test.lap_log_energy[comp_mask], bins=log_energy_bins)[0]
        num_MC_energy_err = np.sqrt(num_MC_energy)

        # Get number of correctly identified comp in each reco energy bin
        num_reco_energy = np.histogram(df_test.lap_log_energy[comp_mask & correctly_identified_mask],
                                       bins=log_energy_bins)[0]
        num_reco_energy_err = np.sqrt(num_reco_energy)

        # Calculate correctly identified fractions as a function of MC energy
        frac_correct[composition], frac_correct_err[composition] = ratio_error(
            num_reco_energy, num_reco_energy_err,
            num_MC_energy, num_MC_energy_err)

    return frac_correct, frac_correct_err


def get_CV_frac_correct(df_train, train_columns, pipeline_str, comp_list,
                        log_energy_bins=get_energybins().log_energy_bins,
                        n_splits=10):

    pipeline = get_pipeline(pipeline_str)

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=2)
    frac_correct_folds = defaultdict(list)

    if not pipeline_str in ['BDT', 'stacked']:
        bar = pyprind.ProgBar(10, monitor=True, title='10-fold CV', stream=1)
        for train_index, test_index in skf.split(df_train, df_train.target):
            df_train_fold = df_train.iloc[train_index].reset_index(drop=True)
            df_test_fold = df_train.iloc[test_index].reset_index(drop=True)
            frac_correct, frac_correct_err = get_frac_correct(df_train_fold,
                            df_test_fold, train_columns, pipeline, comp_list)

            for composition in comp_list+['total']:
                frac_correct_folds[composition].append(frac_correct[composition])

            bar.update(force_flush=True)

        print(bar)
    else:
        # Set up get_frac_correct to run on each CV fold
        folds = []
        for train_index, test_index in skf.split(df_train, df_train.target):
            df_train_fold = df_train.iloc[train_index].reset_index(drop=True)
            df_test_fold = df_train.iloc[test_index].reset_index(drop=True)
            frac_correct = delayed(get_frac_correct)(df_train_fold, df_test_fold,
                        train_columns, pipeline, comp_list, log_energy_bins)
            folds.append(frac_correct)

        # Run get_frac_correct on each fold in parallel
        print('Running {}-fold CV model evaluation...'.format(n_splits))
        with ProgressBar():
            folds = compute(folds, get=multiprocessing.get,
                            num_works=min(n_splits, 20))[0]

        # Get process results from the output queue
        for fold in folds:
            frac_correct, frac_correct_err = fold
            for composition in comp_list+['total']:
                frac_correct_folds[composition].append(frac_correct[composition])


    return frac_correct_folds
