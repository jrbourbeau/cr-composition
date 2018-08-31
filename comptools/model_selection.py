
from __future__ import division
from collections import defaultdict
import dask
from dask import delayed, multiprocessing, threaded
from dask.diagnostics import ProgressBar
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, KFold, GridSearchCV
from sklearn.metrics import get_scorer

from .base import get_training_features
from .io import dataframe_to_X_y
from .composition_encoding import get_comp_list
from .data_functions import ratio_error
from .pipelines import get_pipeline


def _get_frac_correct(df_train, df_test, feature_columns, num_groups,
                      pipeline_str, comp_list, log_energy_bins):
    '''Calculates the fraction of correctly identified samples in each energy bin
    for each composition in comp_list. In addition, the statisitcal error for the
    fraction correctly identified is calculated.'''

    # Fit pipeline and get mask for correctly identified events
    target = 'comp_target_{}'.format(num_groups)
    pipeline = get_pipeline(pipeline_str)
    pipeline.fit(df_train[feature_columns], df_train[target])
    test_predictions = pipeline.predict(df_test[feature_columns])
    correctly_identified_mask = (test_predictions == df_test[target])

    # Construct MC composition masks
    MC_comp_mask = {}
    for composition in comp_list:
        MC_comp_mask[composition] = df_test['comp_group_{}'.format(num_groups)] == composition
    MC_comp_mask['total'] = np.ones(len(df_test), dtype=bool)

    data = {}
    for composition in comp_list + ['total']:
        comp_mask = MC_comp_mask[composition]
        # Get number of MC comp in each reco energy bin
        num_MC_energy = np.histogram(df_test.loc[comp_mask, 'MC_log_energy'], bins=log_energy_bins)[0]
        num_MC_energy_err = np.sqrt(num_MC_energy)

        # Get number of correctly identified comp in each reco energy bin
        num_reco_energy = np.histogram(df_test.loc[comp_mask & correctly_identified_mask, 'MC_log_energy'],
                                       bins=log_energy_bins)[0]
        num_reco_energy_err = np.sqrt(num_reco_energy)

        # Calculate correctly identified fractions as a function of MC energy
        frac_correct, frac_correct_err = ratio_error(
            num_reco_energy, num_reco_energy_err,
            num_MC_energy, num_MC_energy_err)
        data['frac_correct_{}'.format(composition)] = frac_correct
        data['frac_correct_err_{}'.format(composition)] = frac_correct_err

    return data


def get_CV_frac_correct(df_train, feature_list, target, pipeline_str, num_groups,
                        log_energy_bins, n_splits=10, n_jobs=1):

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=2)

    comp_list = get_comp_list(num_groups=num_groups)
    comp_target = target

    # Set up get_frac_correct to run on each CV fold
    folds = []
    for train_index, test_index in skf.split(df_train, df_train[comp_target]):
        df_train_fold = df_train.iloc[train_index]
        df_test_fold = df_train.iloc[test_index]
        frac_correct = delayed(_get_frac_correct)(
                    df_train_fold, df_test_fold, feature_list, num_groups,
                    pipeline_str, comp_list, log_energy_bins)
        folds.append(frac_correct)

    df_cv = delayed(pd.DataFrame.from_records)(folds)

    # Run get_frac_correct on each fold in parallel
    print('Running {}-fold CV model evaluation...'.format(n_splits))
    with ProgressBar():
        get = multiprocessing.get if n_jobs > 1 else dask.get
        df_cv = df_cv.compute(get=get, num_works=n_jobs)

    return df_cv


@delayed
def _cross_validate_comp(df_train, df_test, pipeline_str, param_name,
                         param_value, feature_list=None,
                         target='comp_target_2', scoring='r2', num_groups=2,
                         n_splits=10):
    '''Calculates stratified k-fold CV scores for a given hyperparameter value

    Parameters
    ----------
    df_train : pandas.DataFrame
        Training DataFrame (see comptools.load_sim()).
    df_test : pandas.DataFrame
        Testing DataFrame (see comptools.load_sim()).
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
    # assert scoring in ['accuracy', 'mse', 'r2'], 'Invalid scoring parameter'
    comp_list = get_comp_list(num_groups=num_groups)
    if feature_list is None:
        feature_list, _ = get_training_features()

    pipeline = get_pipeline(pipeline_str)
    pipeline.named_steps['classifier'].set_params(**{param_name: param_value})
    # Only run on a single core
    try:
        pipeline.named_steps['classifier'].set_params(**{'n_jobs': 1})
    except ValueError:
        pass

    data_dict = {'classifier': pipeline_str, 'param_name': param_name,
                 'param_value': param_value, 'n_splits': n_splits}

    train_scores = defaultdict(list)
    test_scores = defaultdict(list)

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=2)
    scorer = get_scorer(scoring)

    for train_index, test_index in kf.split(df_train.values):

        df_train_fold = df_train.iloc[train_index]
        df_test_fold = df_train.iloc[test_index]
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
            comp_key = 'comp_group_{}'.format(num_groups)

            comp_mask_train = df_train_fold[comp_key] == composition
            comp_score_train = scorer(y_train[comp_mask_train],
                                      train_pred[comp_mask_train])
            train_scores[composition].append(comp_score_train)

            comp_mask_test = df_test_fold[comp_key] == composition
            comp_score_test = scorer(y_test[comp_mask_test],
                                     test_pred[comp_mask_test])
            test_scores[composition].append(comp_score_test)

    for label in comp_list + ['total']:
        data_dict['train_mean_{}'.format(label)] = np.mean(train_scores[label])
        data_dict['train_std_{}'.format(label)] = np.std(train_scores[label])
        data_dict['test_mean_{}'.format(label)] = np.mean(test_scores[label])
        data_dict['test_std_{}'.format(label)] = np.std(test_scores[label])

    return data_dict


def cross_validate_comp(df_train, df_test, pipeline_str, param_name,
                        param_values, feature_list=None,
                        target='comp_target_2', scoring='accuracy',
                        num_groups=2, n_splits=10, n_jobs=1, verbose=False):
    '''Calculates stratified k-fold CV scores for a given hyperparameter value

    Similar to sklearn.model_selection.cross_validate, but returns results
    for individual composition groups as well as the combined CV result.

    Parameters
    ----------
    df_train : pandas.DataFrame
        Training DataFrame (see comptools.load_sim()).
    df_test : pandas.DataFrame
        Testing DataFrame (see comptools.load_sim()).
    pipeline_str : str
        Name of pipeline to use (e.g. 'BDT', 'RF_energy', etc.).
    param_name : str
        Name of hyperparameter (e.g. 'max_depth', 'learning_rate', etc.).
    param_values : array-like
        Values to set hyperparameter to.
    feature_list : list, optional
        List of training feature columns to use (default is to use
        comptools.get_training_features()).
    target : str, optional
        Training target to use (default is 'comp_target_2').
    scoring : str, optional
        Scoring metric to calculate for each CV fold (default is 'accuracy').
    num_groups : int, optional
        Number of composition class groups to use (default is 2).
    n_splits : int, optional
        Number of folds to use in (KFold) cross-validation
        (default is 10).
    n_jobs : int, optional
        Number of jobs to run in parallel (default is 1).
    verbose : bool, optional
        Option to print a progress bar (default is False).

    Returns
    -------
        df_cv : pandas.DataFrame
            Returns a DataFrame with average scores as well as CV errors
            on those scores for each composition.

    '''
    cv_dicts = []
    for param_value in param_values:
        cv_dict = _cross_validate_comp(
                    df_train, df_test, pipeline_str,
                    param_name, param_value,
                    feature_list=feature_list, target=target,
                    scoring=scoring, num_groups=num_groups, n_splits=n_splits)
        cv_dicts.append(cv_dict)

    df_cv = delayed(pd.DataFrame.from_records)(cv_dicts, index='param_value')

    get = dask.get if n_jobs == 1 else threaded.get
    # get = dask.get if n_jobs == 1 else multiprocessing.get
    if verbose:
        with ProgressBar():
            print('Performing {}-fold CV on {} hyperparameter values ({} fits):'.format(
                n_splits, len(param_values),  n_splits*len(param_values)))
            df_cv = df_cv.compute(get=get, num_works=n_jobs)
    else:
        df_cv = df_cv.compute(get=get, num_works=n_jobs)

    return df_cv


def get_param_grid(pipeline_name=None):
    """Returns dictionary with hyperparameter values to search

    Parameters
    ----------
    pipeline_name : str, optional
        Pipeline name. Should be formatted as
        <name>_comp_<config>_<num_groups>-groups. For example,
        pipeline_name=BDT_comp_IC86.2012_2-groups (default is None).

    Returns
    -------
    param_grid : dict
        Dictionary with hyperparameter names / values to be passed to
        GridSearchCV.
    """
    if pipeline_name is None:
        raise ValueError('Must enter a value for pipeline_name.')
    if 'BDT' in pipeline_name:
        param_grid = {'classifier__n_estimators': [10, 50, 100, 150, 200, 250, 300],
                      'classifier__max_depth': list(range(1, 11)),
                      'classifier__learning_rate': [0.1, 0.5, 0.8]}
    elif 'xgboost' in pipeline_name:
        # param_grid = {'classifier__n_estimators': [100, 150, 200],
        #               'classifier__max_depth': list(range(3, 7)),
        #               'classifier__learning_rate': [0.1, 0.15, 0.2]}
        param_grid = {'classifier__n_estimators': [100, 150, 200, 250, 300, 400, 500, 600],
                      'classifier__max_depth': list(range(3, 11)),
                      'classifier__learning_rate': [0.01, 0.025, 0.05, 0.1, 0.15, 0.2],
                      # 'classifier__subsample': [0.5, 0.75, 1.0],
                      }
    else:
        raise ValueError('Invalid pipeline entered: {}'.format(pipeline_name))

    return param_grid


def gridsearch_optimize(pipeline, param_grid, X_train, y_train,
                        scoring='accuracy', n_jobs=1, return_gridsearch=False):
    """Runs a grid search to optimize hyperparameters

    Parameters
    ----------
    pipeline : sklearn.model_selection.Pipeline
        Pipeline to fit.
    param_grid : dict
        Dictionary with hyperparameter names / values to be passed to
        GridSearchCV.
    X_train : array_like
        Training features.
    y_train : array_like
        Training labels.
    scoring : str
        Scoring metric to use (default is 'accuracy').
    n_jobs : int, optional
        Number of jobs to run in parallel (default is 1).
    return_gridsearch : bool, optional
        Whether to return the fitted GridSearchCV object, or the
        best_estimator_ object (default is False, so will return the
        best_estimator_).

    Returns
    -------
    best_pipeline : sklearn.model_selection.Pipeline
        Pipeline with optimal hyperparameter values that has been trained on
        the entire training dataset (X_train, y_train).
    gridsearch : sklearn.model_selection.GridSearchCV
        Fitted GridSearchCV object.
    """
    # Want to make sure pipeline isn't running in parallel
    # Will run GridSearchCV in parallel instead
    if hasattr(pipeline, 'classifier__n_jobs'):
        pipeline.set_params(classifier__n_jobs=1)

    param_str = '\n\t'.join(['{}: {}'.format(key, value) for key, value in param_grid.iteritems()])
    print('Running grid search over the following hyperparameters:\n\t{}'.format(param_str))
    gridsearch = GridSearchCV(pipeline,
                              param_grid=param_grid,
                              cv=10,
                              scoring=scoring,
                              n_jobs=n_jobs,
                              return_train_score=True,
                              verbose=2)
    gridsearch.fit(X_train, y_train)
    print('best GridSearchCV params = {}'.format(gridsearch.best_params_))

    if return_gridsearch:
        return gridsearch
    else:
        best_pipeline = gridsearch.best_estimator_
        return best_pipeline
