#!/usr/bin/env python

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import StratifiedShuffleSplit
from . import export

@export
def get_train_test_sets(df, feature_list, weight_type=None):

    if weight_type:
        feature_list += [weight_type]
    # Load and preprocess training data
    X, y = df[feature_list].values , df.MC_comp.values
    # Convert comp string labels to numerical labels
    le = LabelEncoder()
    y = le.fit_transform(y)

    # Split data into training and test samples
    sss = StratifiedShuffleSplit(n_splits=10, test_size=0.3, random_state=2)
    for train_index, test_index in sss.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

    # # Scale features and labels
    # # NOTE: the scaler is fit only to the training features
    # stdsc = StandardScaler()
    # X_train_std = stdsc.fit_transform(X_train)
    # X_test_std = stdsc.transform(X_test)

    # return X_train_std, X_test_std, y_train, y_test
    return X_train, X_test, y_train, y_test, le
