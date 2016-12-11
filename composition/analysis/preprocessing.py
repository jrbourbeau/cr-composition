#!/usr/bin/env python

import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import StratifiedShuffleSplit
from . import export

@export
def get_train_test_sets(df, feature_list, train_he=True, test_he=True):

    # mask = np.logical_or(
    #     (df.MC_comp == 'P'),
    #     (df.MC_comp == 'Fe')
    # )
    # df = df[mask]

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


    if not train_he:
        # Take out all He from training set
        He_mask = (le.inverse_transform(y_train) == 'He')
        X_train = X_train[np.logical_not(He_mask)]
        y_train = y_train[np.logical_not(He_mask)]
    if not test_he:
        # Take out all He from training set
        He_mask = (le.inverse_transform(y_test) == 'He')
        X_test = X_test[np.logical_not(He_mask)]
        y_test = y_test[np.logical_not(He_mask)]

    # # Scale features and labels
    # # NOTE: the scaler is fit only to the training features
    # stdsc = StandardScaler()
    # X_train_std = stdsc.fit_transform(X_train)
    # X_test_std = stdsc.transform(X_test)

    return X_train, X_test, y_train, y_test, le
