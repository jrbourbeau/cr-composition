#!/usr/bin/env python

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, NuSVC

# Use get_pipeline to ensure that same hyperparameters are used each time a
# classifier is needed, and that the proper scaling is always done before
# fitting
def get_pipeline(classifier_name):
    ''' Returns classifier pipeline '''

    if classifier_name == 'RF':
        classifier = RandomForestClassifier(
            n_estimators=100, max_depth=10, n_jobs=10, random_state=2)
    elif classifier_name == 'KN':
        classifier = KNeighborsClassifier(n_neighbors=100, n_jobs=10)
    elif classifier_name == 'NuSVC':
        classifier = NuSVC()
    elif classifier_name == 'GBC':
        classifier = GradientBoostingClassifier(max_depth=5, random_state=2)
    else:
        raise('{} is not a valid classifier name...'.format(classifier_name))

    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', classifier)])

    return pipeline
