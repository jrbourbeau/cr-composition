
from __future__ import division
from collections import defaultdict
import multiprocessing as mp
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
import pyprind
# from .base import DataSet
from ..dataframe_functions import label_to_comp
from .base import get_energybins
from .data_functions import ratio_error
from .pipelines import get_pipeline


def get_frac_correct(df_train, df_test, feature_columns, pipeline, comp_list,
                    log_energy_bins=get_energybins().log_energy_bins,
                    frac_queue=None, frac_err_queue=None):
    '''Calculates the fraction of correctly identified samples in each energy bin
    for each composition in comp_list. In addition, the statisitcal error for the
    fraction correctly identified is calculated.'''

    # Validate input
    assert isinstance(df_train, pd.DataFrame), 'df_train dataset must be a pandas DataFrame'
    assert isinstance(df_test, pd.DataFrame), 'df_test dataset must be a pandas DataFrame'
    # assert train.y is not None, 'train must have true y values'
    # assert test.y is not None, 'test must have true y values'
    # assert test.log_energy is not None, 'teset must have log_energ values'
    # assert all([composition in train.labels for composition in comp_list]), 'comp_list and train.labels don\'t match'
    # assert all([composition in test.labels for composition in comp_list]), 'comp_list and test.labels don\'t match'

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

    if (frac_queue is None) and (frac_err_queue is None): # Not multiprocessing
        return frac_correct, frac_correct_err
    elif (frac_queue is not None) and (frac_err_queue is not None): # Multiprocessing
        frac_queue.put(frac_correct)
        frac_err_queue.put(frac_correct_err)
    else:
        raise('Only one of frac_queue or frac_err_queue were specified!')


def get_CV_frac_correct(df_train, train_columns, pipeline_str, comp_list,
                        log_energy_bins=get_energybins().log_energy_bins,
                        n_splits=10):

    pipeline = get_pipeline(pipeline_str)

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=2)
    frac_correct_folds = defaultdict(list)

    if not pipeline_str in ['GBDT', 'stacked']:
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
        print('Using scikit-learn {}...'.format(pipeline.named_steps['classifier'].__class__.__name__))
        # Define output queues
        frac_correct_queue = mp.Queue()
        frac_correct_err_queue = mp.Queue()
        # Setup a list of processes that we want to run
        processes = []
        for train_index, test_index in skf.split(df_train, df_train.target):
            df_train_fold = df_train.iloc[train_index].reset_index(drop=True)
            df_test_fold = df_train.iloc[test_index].reset_index(drop=True)
            process = mp.Process(target=get_frac_correct,
                                 args=(df_train_fold, df_test_fold, train_columns,
                                       pipeline, comp_list, log_energy_bins,
                                       frac_correct_queue, frac_correct_err_queue))
            processes.append(process)

        # Run processes
        print('Running {} folds in parallel...'.format(len(processes)))
        for fold_idx, p in enumerate(processes):
            p.start()

        # Exit the completed processes
        for fold_idx, p in enumerate(processes):
            p.join()
            print('Completed fold {}'.format(fold_idx))

        # Get process results from the output queue
        for fold_idx, p in enumerate(processes):
            frac_correct = frac_correct_queue.get()
            frac_correct_err = frac_correct_err_queue.get()
            for composition in comp_list+['total']:
                frac_correct_folds[composition].append(frac_correct[composition])
            # frac_correct = [frac_correct_queue.get() for p in processes]
            # frac_correct_err = [frac_correct_err_queue.get() for p in processes]

    return frac_correct_folds
