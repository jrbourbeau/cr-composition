
import os
from sklearn.pipeline import Pipeline
from sklearn.ensemble import (RandomForestClassifier, RandomForestRegressor,
                              AdaBoostClassifier, GradientBoostingClassifier)
from sklearn.pipeline import make_pipeline
from sklearn.externals import joblib
from sklearn.decomposition import PCA
from sklearn.base import TransformerMixin
from mlxtend.preprocessing import standardize
from xgboost import XGBClassifier

from .base import get_paths


def get_pipeline(classifier_name='BDT'):
    """ Function to get classifier pipeline.
    """
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
    # elif classifier_name in ['GBDT', 'BDT']:
    #     classifier = GradientBoostingClassifier(
    #         loss='exponential', max_depth=3, n_estimators=100, random_state=2)
    #     # classifier = GradientBoostingClassifier(loss='deviance', max_depth=3,
    #     #     n_estimators=500, random_state=2)

    elif classifier_name == 'BDT_comp_IC79.2010':
        classifier = GradientBoostingClassifier(
            loss='deviance', max_depth=4, n_estimators=100, random_state=2)
    elif classifier_name == 'BDT_comp_IC79.2010_2-groups':
        classifier = GradientBoostingClassifier(
            loss='deviance', max_depth=4, n_estimators=100, random_state=2)

    elif classifier_name == 'BDT_comp_IC86.2012_2-groups':
        classifier = GradientBoostingClassifier(
            loss='deviance', max_depth=4, n_estimators=100, random_state=2)
    elif classifier_name == 'BDT_comp_IC86.2012_3-groups':
        classifier = GradientBoostingClassifier(
            loss='deviance', max_depth=3, n_estimators=100, random_state=2)
    elif classifier_name == 'BDT_comp_IC86.2012_4-groups':
        classifier = GradientBoostingClassifier(
            loss='deviance', max_depth=2, n_estimators=100, random_state=2)

    elif classifier_name == 'RF_energy_IC79.2010':
        classifier = RandomForestRegressor(
            n_estimators=100, max_depth=8, n_jobs=10, random_state=2)
    elif classifier_name == 'RF_energy_IC86.2012':
        classifier = RandomForestRegressor(
            n_estimators=100, max_depth=7, n_jobs=10, random_state=2)
    else:
        raise ValueError(
            '{} is not a valid classifier name'.format(classifier_name))

    pipeline = Pipeline([
        # ('scaler', StandardScaler()),
        # ('pca', PCA(n_components=4, random_state=2)),
        # ('lda', LinearDiscriminantAnalysis(n_discriminants=6)),
        ('classifier', classifier)])

    return pipeline


def load_trained_model(pipeline_str='BDT'):
    """Function to load pre-trained model to avoid re-training

    Parameters
    ----------
    pipeline_str : str, optional
        Name of model to load (default is 'BDT').

    Returns
    -------
    model_dict : dict
        Dictionary containing trained model as well as relevant metadata.

    """
    paths = get_paths()
    model_file = os.path.join(paths.project_root, 'models',
                              '{}.pkl'.format(pipeline_str))
    if not os.path.exists(model_file):
        raise IOError('There is no saved model file {}'.format(model_file))

    model_dict = joblib.load(model_file)

    return model_dict
