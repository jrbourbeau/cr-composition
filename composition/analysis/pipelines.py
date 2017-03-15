#!/usr/bin/env python

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, ExtraTreesClassifier, VotingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, NuSVC
from sklearn.decomposition import PCA
from mlxtend.feature_extraction import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.pipeline import make_pipeline, make_union
from sklearn.preprocessing import FunctionTransformer
from sklearn.naive_bayes import GaussianNB

from xgboost import XGBClassifier

from . import export

# Use get_pipeline to ensure that same hyperparameters are used each time a
# classifier is needed, and that the proper scaling is always done before
# fitting
@export
def get_pipeline(classifier_name):
    ''' Returns classifier pipeline '''

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
    elif classifier_name == 'KN':
        classifier = KNeighborsClassifier(n_neighbors=100, n_jobs=10)
    elif classifier_name == 'SGD':
        classifier = SGDClassifier(random_state=2, n_jobs=10)
        params = {'loss': 'log', 'penalty': 'none'}
        classifier.set_params(**params)
    elif classifier_name == 'NuSVC':
        classifier = NuSVC()
    elif classifier_name == 'GBDT':
        classifier = GradientBoostingClassifier(loss='exponential', max_depth=5,
            n_estimators=125, random_state=2)
    elif classifier_name == 'tpot':
        #2
        # exported_pipeline = make_pipeline(
        #     PCA(iterated_power=10, svd_solver="randomized"),
        #     make_union(VotingClassifier([("est", LogisticRegression(C=0.45, dual=False, penalty="l1"))]), FunctionTransformer(lambda X: X)),
        #     ExtraTreesClassifier(criterion="entropy", max_features=1.0, n_estimators=500)
        # )
        #3
        exported_pipeline = make_pipeline(
            make_union(VotingClassifier([("est", GaussianNB())]), FunctionTransformer(lambda X: X)),
            PCA(iterated_power=10, svd_solver="randomized"),
            RandomForestClassifier(n_estimators=500)
        )

        return exported_pipeline
    else:
        raise('{} is not a valid classifier name...'.format(classifier_name))

    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        # ('pca', PCA(n_components=4, random_state=2)),
        # ('lda', LinearDiscriminantAnalysis(n_discriminants=6)),
        ('classifier', classifier)])

    return pipeline
