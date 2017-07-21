#!/usr/bin/env python

import os
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.externals import joblib

from xgboost import XGBClassifier

from . import export
from ..simfunctions import get_sim_configs
from ..base import get_paths

# Use get_pipeline to ensure that same hyperparameters are used each time a
# classifier is needed, and that the proper scaling is always done before
# fitting
@export
def get_pipeline(classifier_name='BDT'):
    ''' Function to get classifier pipeline.
    '''

    if classifier_name == 'RF':
        classifier = RandomForestClassifier(
            n_estimators=100, max_depth=6, n_jobs=20,
            # n_estimators=100, max_depth=7, min_samples_leaf=150, n_jobs=20,
            random_state=2)
    elif classifier_name == 'xgboost':
        classifier = XGBClassifier(n_estimators=125, nthread=10, silent=True, seed=2)
    elif classifier_name == 'Ada':
        classifier = AdaBoostClassifier(DecisionTreeClassifier(max_depth=5), n_estimators=100, learning_rate=0.1, random_state=2)
        # classifier = AdaBoostClassifier(n_estimators=50, learning_rate=0.1, random_state=2)
    elif classifier_name in ['GBDT', 'BDT']:
        classifier = GradientBoostingClassifier(loss='exponential', max_depth=5,
            n_estimators=300, random_state=2)
            # n_estimators=100, random_state=2)
    else:
        raise ValueError('{} is not a valid classifier name...'.format(classifier_name))

    pipeline = Pipeline([
        # ('scaler', StandardScaler()),
        # ('pca', PCA(n_components=4, random_state=2)),
        # ('lda', LinearDiscriminantAnalysis(n_discriminants=6)),
        ('classifier', classifier)])

    return pipeline


@export
def load_trained_model(config='IC86.2012', pipeline='BDT'):
    '''Function to load pre-trained model to avoid re-training
    '''

    if not config in get_sim_configs():
        raise ValueError('Do not have simulation for detector '
                         'configuration {}'.format(config))

    paths = get_paths()

    model_file = os.path.join(paths.project_root,
                     'models/{}_{}.pkl'.format(pipeline, config))
    try:
        model_dict = joblib.load(model_file)
    except IOError:
        raise IOError('There is no {} model saved for {} '
              '({} doesn\'t exist)'.format(pipeline, config, model_file))

    return model_dict


@export
def fit_pipeline(pipeline, train_df):

    assert isinstance(pipeline, (str, sklearn.pipeline.Pipeline))
    if isinstance(pipeline, str):
        pipeline = get_pipeline(pipeline)

    pipeline.fit(X, y)

    return pipeline
