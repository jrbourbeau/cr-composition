
from __future__ import division
from collections import defaultdict
import dask
from dask import delayed, multiprocessing, compute
from dask.diagnostics import ProgressBar
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score
import pyprind
from ..dataframe_functions import label_to_comp, load_sim, dataframe_to_X_y
from ..composition_encoding import get_comp_list
from .base import get_energybins
from .data_functions import ratio_error
from .pipelines import get_pipeline
from .features import get_training_features



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


@delayed
def _cross_validate_comp(config, pipeline_str, param_name, param_value,
                                feature_list=None, target='comp_target_2',
                                scoring='r2', num_groups=2, n_splits=10):
    '''Calculates stratified k-fold CV scores for a given hyperparameter value

    Parameters
    ----------
    config : str
        Detector configuration.
    pipeline_str : str
        Name of pipeline to use (e.g. 'BDT', 'RF_energy', etc.).
    param_name : str
        Name of hyperparameter (e.g. 'max_depth', 'learning_rate', etc.).
    param_value : int, float, str
        Value to set hyperparameter to.
    feature_list : list, optional
        List of training feature columns to use (default is to use
        comptools.get_training_features()).
    target : str, optional
        Training target to use (default is 'comp_target_2').
    scoring : {'r2', 'mse', 'accuracy'}
        Scoring metric to calculate for each CV fold (default is 'r2').
    num_groups : int, optional
        Number of composition class groups to use (default is 2).
    n_splits : int, optional
        Number of folds to use in (KFold) cross-validation
        (default is 10).

    Returns
    -------
        data_dict : dict
            Return a dictionary with average scores as well as CV errors on those scores.

    '''
    assert scoring in ['accuracy', 'mse', 'r2'], 'Invalid scoring parameter'
    comp_list = get_comp_list(num_groups=num_groups)
    if feature_list is None:
        feature_list, _ = get_training_features()
    df_sim_train, df_sim_test = load_sim(config=config, verbose=False)

    pipeline = get_pipeline(pipeline_str)
    pipeline.named_steps['classifier'].set_params(**{param_name: param_value})
    # Only run on a single core
    pipeline.named_steps['classifier'].set_params(**{'n_jobs': 1})

    data_dict = {'classifier': pipeline_str, 'param_name': param_name,
                 'param_value': param_value, 'n_splits': n_splits}

    train_scores = defaultdict(list)
    test_scores = defaultdict(list)
    ks_pval = defaultdict(list)

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=2)
    if scoring == 'mse':
        scorer = mean_squared_error
    elif scoring == 'r2':
        scorer = r2_score
    else:
        scorer = accuracy_score

    for train_index, test_index in kf.split(df_sim_train):

        df_train_fold, df_test_fold = df_sim_train.iloc[train_index], df_sim_train.iloc[test_index]

        X_train, y_train = dataframe_to_X_y(df_train_fold, feature_list,
                                            target=target)
        X_test, y_test = dataframe_to_X_y(df_test_fold, feature_list,
                                          target=target)

        pipeline = pipeline.fit(X_train, y_train)

        train_pred = pipeline.predict(X_train)
        train_score = scorer(y_train, train_pred)
        train_scores['total'].append(train_score)

        test_pred = pipeline.predict(X_test)
        test_score = scorer(y_test, test_pred)
        test_scores['total'].append(test_score)

        # Get testing/training scores for each composition group
        for composition in comp_list:
            comp_mask_train = df_train_fold['comp_group_{}'.format(num_groups)] == composition
            comp_score_train = scorer(y_train[comp_mask_train], train_pred[comp_mask_train])
            train_scores[composition].append(comp_score_train)

            comp_mask_test = df_test_fold['comp_group_{}'.format(num_groups)] == composition
            comp_score_test = scorer(y_test[comp_mask_test], test_pred[comp_mask_test])
            test_scores[composition].append(comp_score_test)


    for label in comp_list + ['total']:
        data_dict['train_mean_{}'.format(label)] = np.mean(train_scores[label])
        data_dict['train_std_{}'.format(label)] = np.std(train_scores[label])
        data_dict['test_mean_{}'.format(label)] = np.mean(test_scores[label])
        data_dict['test_std_{}'.format(label)] = np.std(test_scores[label])

    return data_dict


def cross_validate_comp(config, pipeline_str, param_name, param_values,
                        feature_list=None, target='comp_target_2',
                        scoring='r2', num_groups=2, n_splits=10, n_jobs=1,
                        verbose=False):
    cv_dicts = []
    for param_value in param_values:
        cv_dict_delayed = _cross_validate_comp(
                    config, pipeline_str, param_name, param_value,
                    feature_list=feature_list, target=target,
                    scoring=scoring, num_groups=num_groups, n_splits=10)
        cv_dicts.append(cv_dict_delayed)

    df_cv = delayed(pd.DataFrame.from_records)(cv_dicts, index='param_value')

    get = dask.get if n_jobs == 1 else multiprocessing.get
    if verbose:
        with ProgressBar():
            print('Performing {}-fold CV on {} hyperparameter values ({} fits):'.format(
                n_splits, len(param_values),  n_splits*len(param_values)))
            df_cv = df_cv.compute(get=get, num_works=n_jobs)
    else:
        df_cv = df_cv.compute(get=get, num_works=n_jobs)

    return df_cv
