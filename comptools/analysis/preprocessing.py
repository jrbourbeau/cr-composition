#==============================================================================
# This module is outdated and should probably be removed! 
#==============================================================================


import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import StratifiedShuffleSplit, ShuffleSplit
from . import export
from ..dataframe_functions import load_dataframe
from .base import DataSet


@export
def get_sim_datasets(df, feature_list, target='MC_comp_class', labelencode=True,
    return_energy=False, return_comp=False):

    # Load and preprocess training data
    X, y = df[feature_list].values, df[target].values
    # Convert comp string labels to numerical labels
    if labelencode:
        le = LabelEncoder()
        y = le.fit_transform(y)
    else:
        le = None

    # Split data into training and test samples
    if labelencode: splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=2)
    else: splitter = ShuffleSplit(n_splits=1, test_size=0.3, random_state=2)
    for train_index, test_index in splitter.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

    # Create DataSet objects
    sim_train = DataSet(X=X_train, y=y_train, le=le)
    sim_test = DataSet(X=X_test, y=y_test, le=le)
    if return_energy:
        energy = df['lap_energy'].values
        energy_train, energy_test = energy[train_index], energy[test_index]
        sim_train.energy = energy_train
        sim_test.energy = energy_test

        log_energy = df['lap_log_energy'].values
        log_energy_train, log_energy_test = log_energy[train_index], log_energy[test_index]
        sim_train.log_energy = log_energy_train
        sim_test.log_energy = log_energy_test

    if return_comp:
        comp_class = df['MC_comp_class'].values
        comp_train, comp_test = comp_class[train_index], comp_class[test_index]
        sim_train.comp = comp_train
        sim_test.comp = comp_test


    return sim_train, sim_test


@export
def preprocess_sim(config='IC79', feature_list=None, target='MC_comp_class',
    labelencode=True, return_energy=False, return_comp=False):

    # Load simulation dataframe
    df = load_dataframe(datatype='sim', config=config)

    # Format testing and training sets
    feature_list, feature_labels = get_training_features(feature_list)
    features_str = ''
    for label in feature_labels:
        features_str += label + '\n\t'
    print('Selecting the following features:\n\t'+features_str)
    sim_train, sim_test = get_sim_datasets(df, feature_list, target=target,
        labelencode=labelencode, return_energy=return_energy, return_comp=return_comp)

    print('Number training events = {}'.format(len(sim_train)))
    print('Number testing events = {}'.format(len(sim_test)))

    return sim_train, sim_test


@export
def preprocess_data(config='IC79', feature_list=None, return_energy=False):

    # Load sim dataframe
    df = load_dataframe(datatype='data', config=config)
    # Format testing set
    feature_list, feature_labels = get_training_features()
    features_str = ''
    for label in feature_labels:
        features_str += label + '\n\t'
    print('Selecting the following features:\n\t'+features_str)
    X_test = df[feature_list].values

    print('Number testing events = ' + str(X_test.shape[0]))

    dataset = DataSet(X=X_test)
    if return_energy:
        # IMPORTANT: shuffle energys with same shuffle array used to randomize
        dataset.log_energy = df['lap_log_energy'].values

    return dataset
