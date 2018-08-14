
import os
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.ensemble import (RandomForestClassifier, RandomForestRegressor,
                              AdaBoostClassifier, GradientBoostingClassifier,
                              VotingClassifier)
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.externals import joblib
from xgboost import XGBClassifier, XGBRegressor

from sklearn.linear_model import LogisticRegression, LinearRegression, SGDClassifier
from mlxtend.classifier import StackingClassifier

from .base import get_paths

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics import euclidean_distances
from sklearn.pipeline import make_pipeline


def line(x, x1, y1, x2, y2):
    return (x - x1) * ((y2-y1) / (x2-x1)) + y1

class LineCutClassifier(BaseEstimator, ClassifierMixin):

    def __init__(self, demo_param='demo'):
        self.demo_param = demo_param

    def fit(self, X, y):

        # Check that X and y have correct shape
        X, y = check_X_y(X, y)
        # Store the classes seen during fit
        self.classes_ = unique_labels(y)
        assert len(self.classes_) == 4, 'Must have 4 classes'

        self.X_ = X
        self.y_ = y
        # Return the classifier
        return self

    def predict(self, X):

        # Check is fit had been called
        check_is_fitted(self, ['X_', 'y_'])

        # Input validation
        X = check_array(X)

        assert X.shape[1] == 3, 'Must have only 3 training features'

        y = np.empty(len(X), dtype=int)
        for idx, (log_s125, log_dEdX) in enumerate(X[:, (1, 2)]):
            log_dEdX_iron = line(log_s125, 0, 1.1, 2, 2.5)
            log_dEdX_oxygen = line(log_s125, 0, 0.9, 2, 2.4)
            log_dEdX_proton = line(log_s125, 0, 0.75, 2, 2.3)

            if log_dEdX <= log_dEdX_proton:
                y[idx] = 0
            elif (log_dEdX <= log_dEdX_oxygen) and (log_dEdX > log_dEdX_proton):
                y[idx] = 1
            elif (log_dEdX <= log_dEdX_iron) and (log_dEdX > log_dEdX_oxygen):
                y[idx] = 2
            else:
                y[idx] = 3

        return y


class CustomClassifier(BaseEstimator, ClassifierMixin):

    def __init__(self, p=0.8, neighbor_weight=2.0, num_groups=4,
                 random_state=2):
        self.p = p
        self.neighbor_weight = neighbor_weight
        self.num_groups = num_groups
        self.random_state = random_state

    def fit(self, X, y):
        # Check that X and y have correct shape
        X, y = check_X_y(X, y)
        # Store the classes seen during fit
        self.classes_ = unique_labels(y)
        if not len(self.classes_) == self.num_groups:
            raise ValueError('Must have {} classes'.format(self.num_groups))

        self.X_ = X
        self.y_ = y

        return self

    def predict(self, y):
        """Performs random composition classification
        """
        # # Check is fit had been called
        # check_is_fitted(self, ['X_', 'y_'])

        # Input validation
        # y = check_array(y)

        # Want to get reproducible random classifications
        np.random.seed(self.random_state)
        p_correct = self.p
        y_pred = np.empty_like(y)
        targets = list(range(self.num_groups))
        probs = np.empty_like(targets, dtype=float)
        for target in targets:
            comp_mask = y == target
            probs[target] = p_correct

            not_target = [i for i in targets if i != target]

            neighbors = [target - 1, target + 1]
            neighbors = [i for i in neighbors if i >= 0 and i < self.num_groups]

            not_neighbors = list(set(not_target).difference(neighbors))

            weight = (1 - p_correct) / (len(not_neighbors) + self.neighbor_weight * len(neighbors))

            probs[not_neighbors] = weight
            probs[neighbors] = self.neighbor_weight * weight

            # Get custom composition classification
            y_pred_target = np.random.choice(targets, size=comp_mask.sum(), p=probs)
            y_pred[comp_mask] = y_pred_target

        return y_pred


def get_pipeline(classifier_name='BDT'):
    """ Function to get classifier pipeline.
    """
    steps = []
    if classifier_name == 'RF':
        classifier = RandomForestClassifier(
            n_estimators=100, max_depth=6, n_jobs=20,
            # n_estimators=100, max_depth=7, min_samples_leaf=150, n_jobs=20,
            random_state=2)
    elif classifier_name == 'xgboost':
        classifier = XGBClassifier(n_estimators=125, nthread=10, silent=True,
                                   seed=2)
    elif classifier_name == 'Ada':
        classifier = AdaBoostClassifier(DecisionTreeClassifier(max_depth=5),
                                        n_estimators=100, learning_rate=0.1,
                                        random_state=2)
        # classifier = AdaBoostClassifier(n_estimators=50, learning_rate=0.1, random_state=2)
    # elif classifier_name in ['GBDT', 'BDT']:
    #     classifier = GradientBoostingClassifier(
    #         loss='exponential', max_depth=3, n_estimators=100, random_state=2)
    #     # classifier = GradientBoostingClassifier(loss='deviance', max_depth=3,
    #     #     n_estimators=500, random_state=2)

    elif classifier_name == 'BDT_comp_IC79.2010':
        classifier = GradientBoostingClassifier(
            loss='deviance', max_depth=4, n_estimators=100, random_state=2)
        steps.append(('classifier', classifier))
    elif classifier_name == 'BDT_comp_IC79.2010_2-groups':
        classifier = GradientBoostingClassifier(
            loss='deviance', max_depth=4, n_estimators=100, random_state=2)
        steps.append(('classifier', classifier))

    elif classifier_name == 'BDT_comp_IC86.2012_2-groups':
        classifier = GradientBoostingClassifier(loss='deviance',
                                                max_depth=4,
                                                n_estimators=100,
                                                random_state=2)
        steps.append(('classifier', classifier))
    elif classifier_name == 'BDT_comp_IC86.2012_3-groups':
        classifier = GradientBoostingClassifier(loss='deviance',
                                                max_depth=3,
                                                n_estimators=100,
                                                random_state=2)
        steps.append(('classifier', classifier))
    elif classifier_name == 'BDT_comp_IC86.2012_4-groups':
        classifier = GradientBoostingClassifier(loss='deviance',
                                                max_depth=2,
                                                n_estimators=100,
                                                random_state=2)
        steps.append(('classifier', classifier))
    elif 'CustomClassifier' in classifier_name:
        hyperparams_str = classifier_name.split('_')[1:]
        assert len(hyperparams_str) == 3, 'Too many CustomClassifier hyperparams. Got {} but should have 3.'.format(len(hyperparams_str))
        p = float(hyperparams_str[0])
        neighbor_weight = float(hyperparams_str[1])
        num_groups = int(hyperparams_str[2])
        classifier = CustomClassifier(p=p, neighbor_weight=neighbor_weight,
                                      num_groups=num_groups, random_state=2)
        steps.append(('classifier', classifier))

    elif classifier_name == 'RF_comp_IC86.2012_4-groups':
        classifier = RandomForestClassifier(max_depth=10, n_estimators=500,
                                            random_state=2, n_jobs=10)
        steps.append(('classifier', classifier))

    elif classifier_name == 'SVC_comp_IC86.2012_2-groups':
        classifier = SVC(C=0.5, random_state=2)
        steps.append(('scaler', StandardScaler()))
        steps.append(('classifier', classifier))
    elif classifier_name == 'SVC_comp_IC86.2012_4-groups':
        classifier = SVC(C=0.5, random_state=2)
        steps.append(('scaler', StandardScaler()))
        steps.append(('classifier', classifier))

    elif classifier_name == 'LinearSVC_comp_IC86.2012_2-groups':
        classifier = LinearSVC(random_state=2)
        steps.append(('scaler', StandardScaler()))
        steps.append(('classifier', classifier))
    elif classifier_name == 'LinearSVC_comp_IC86.2012_4-groups':
        classifier = LinearSVC(random_state=2)
        steps.append(('scaler', StandardScaler()))
        steps.append(('classifier', classifier))

    elif classifier_name == 'NuSVC_comp_IC86.2012_4-groups':
        classifier = NuSVC(random_state=2)
        steps.append(('scaler', StandardScaler()))
        steps.append(('classifier', classifier))

    elif classifier_name == 'xgboost_comp_IC86.2012_2-groups':
        classifier = XGBClassifier(learning_rate=0.05,
                                   max_depth=7,
                                   n_estimators=150,
                                   # subsample=0.75,
                                   random_state=2)
        steps.append(('classifier', classifier))

    elif classifier_name == 'xgboost_comp_IC86.2012_4-groups':
        classifier = XGBClassifier(max_depth=2,
                                   n_estimators=100,
                                   # subsample=0.75,
                                   random_state=2)
        steps.append(('classifier', classifier))

    elif classifier_name == 'LogisticRegression_comp_IC86.2012_4-groups':
        classifier = LogisticRegression(random_state=2)
        steps.append(('scaler', StandardScaler()))
        steps.append(('classifier', classifier))


    elif classifier_name == 'linecut_comp_IC86.2012_4-groups':
        classifier = LineCutClassifier()
        steps.append(('classifier', classifier))


    elif classifier_name == 'stacking_comp_IC86.2012_4-groups':
        classifiers = [SVC(random_state=2),
                       LinearSVC(random_state=2),
                       GradientBoostingClassifier(loss='deviance',
                                                  max_depth=2,
                                                  n_estimators=100,
                                                  random_state=2),
                                                  ]
        classifier = StackingClassifier(classifiers,
                                        meta_classifier=LogisticRegression())
        steps.append(('scaler', StandardScaler()))
        steps.append(('classifier', classifier))
    elif classifier_name == 'voting_comp_IC86.2012_4-groups':
        # classifiers = [SVC(random_state=2),
        #                LinearSVC(random_state=2),
        #                GradientBoostingClassifier(loss='deviance',
        #                                           max_depth=2,
        #                                           n_estimators=100,
        #                                           random_state=2),
        #                                           ]

        estimators = [('SVC', SVC(random_state=2)),
                      # ('LinearSVC', LinearSVC(random_state=2)),
                      ('LogisticRegression', LogisticRegression(random_state=2)),
                      # ('BDT', GradientBoostingClassifier(loss='deviance',
                      #                                    max_depth=2,
                      #                                    n_estimators=100,
                      #                                    random_state=2)),
                       ('xgboost', XGBClassifier(max_depth=3,
                                                 booster='gblinear',
                                                 n_estimators=100,
                                                 random_state=2))]
        classifier = VotingClassifier(estimators, voting='hard')
        steps.append(('scaler', StandardScaler()))
        steps.append(('classifier', classifier))


    elif classifier_name == 'RF_energy_IC79.2010':
        classifier = RandomForestRegressor(n_estimators=100,
                                           max_depth=8,
                                           n_jobs=10,
                                           random_state=2)
        steps.append(('classifier', classifier))
    elif classifier_name == 'RF_energy_IC86.2012':
        classifier = RandomForestRegressor(n_estimators=100,
                                           max_depth=7,
                                           n_jobs=10,
                                           random_state=2)
        steps.append(('classifier', classifier))
    elif classifier_name == 'xgboost_energy_IC86.2012':
        classifier = XGBRegressor(n_estimators=75,
                                  booster='gblinear',
                                  # subsample=0.75,
                                  random_state=2)
        steps.append(('classifier', classifier))
    elif classifier_name == 'linearregression_energy_IC86.2012':
        reg = make_pipeline(PolynomialFeatures(2),
                            # StandardScaler(),
                            LinearRegression(),
                            )
        return reg
    elif classifier_name == 'SGD_comp_IC86.2012_2-groups':
        # clf = make_pipeline(StandardScaler(),
        #                     SGDClassifier(random_state=2, n_jobs=1),
        #                     )
        clf = make_pipeline(StandardScaler(),
                            SGDClassifier(loss='hinge', alpha=1e-3, max_iter=50, tol=1e-3, shuffle=True, random_state=2),
                            )
        return clf
    else:
        raise ValueError(
            '{} is not a valid classifier name'.format(classifier_name))

    # pipeline = Pipeline([
    #     # ('scaler', StandardScaler()),
    #     # ('pca', PCA(n_components=4, random_state=2)),
    #     # ('lda', LinearDiscriminantAnalysis(n_discriminants=6)),
    #     ('classifier', classifier)])
    pipeline = Pipeline(steps)

    return pipeline
