#!/usr/bin/env python

import cPickle
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import StratifiedShuffleSplit
from . import export
from ..load_dataframe import load_dataframe


@export
def get_training_features():

    feature_list = ['lap_log_energy', 'lap_cos_zenith', 'log_NChannels_1_30',
                    'nchannels_nhits_ratio', 'lap_likelihood', 'log_NHits_1_30',
                    'StationDensity', 'stationdensity_charge_ratio', 'nchannels_nhits_ratio',
                    'log_s50', 'log_s125', 'log_s500', 'lap_beta']
    # feature_list = ['lap_log_energy', 'lap_cos_zenith', 'log_NChannels_1_30',
    #                 'nchannels_nhits_ratio', 'lap_likelihood', 'log_NHits_1_30',
    #                 'StationDensity', 'stationdensity_charge_ratio', 'nchannels_nhits_ratio',
    #                 'log_s50', 'log_s80', 'log_s125', 'log_s180', 'log_s250', 'log_s500',
    #                 'lap_beta']

    label_dict = {'reco_log_energy': '$\log_{10}(E_{\mathrm{reco}}/\mathrm{GeV})$',
                  'lap_log_energy': '$\log_{10}(E_{\mathrm{Lap}}/\mathrm{GeV})$',
                  'log_s50': '$\log_{10}(S_{\mathrm{50}})$',
                  'log_s80': '$\log_{10}(S_{\mathrm{80}})$',
                  'log_s125': '$\log_{10}(S_{\mathrm{125}})$',
                  'log_s180': '$\log_{10}(S_{\mathrm{180}})$',
                  'log_s250': '$\log_{10}(S_{\mathrm{250}})$',
                  'log_s500': '$\log_{10}(S_{\mathrm{500}})$',
                  'lap_likelihood': '$r\log_{10}(l)$',
                  'lap_beta': 'lap beta',
                  'InIce_log_charge_1_60': 'InIce charge',
                  'InIce_log_charge_1_45': 'InIce charge (top 75\%)',
                  'InIce_charge_1_30': 'InIce charge (top 50\%)',
                  'InIce_log_charge_1_30': '$\log_{10}(InIce charge (top 50))$',
                  #   'InIce_log_charge_1_30': '$\log_{10}$(InIce charge (top 50\%))',
                  #   'InIce_log_charge_1_30': 'InIce charge (top 50\%)',
                  'InIce_log_charge_1_15': 'InIce charge (top 25\%)',
                  'InIce_log_charge_1_6': 'InIce charge (top 10\%)',
                  'reco_cos_zenith': '$\cos(\\theta_{\mathrm{reco}})$',
                  'lap_cos_zenith': '$\cos(\\theta_{\mathrm{Lap}})$',
                  'LLHlap_cos_zenith': '$\cos(\\theta_{\mathrm{Lap}})$',
                  'LLHLF_cos_zenith': '$\cos(\\theta_{\mathrm{LLH+COG}})$',
                  'lap_chi2': '$\chi^2_{\mathrm{Lap}}/\mathrm{n.d.f}$',
                  'NChannels_1_60': 'NChannels',
                  'NChannels_1_45': 'NChannels (top 75\%)',
                  'NChannels_1_30': 'NChannels (top 50\%)',
                  'NChannels_1_15': 'NChannels (top 25\%)',
                  'NChannels_1_6': 'NChannels (top 10\%)',
                  'log_NChannels_1_30': '$\log_{10}$(NChannels (top 50\%))',
                  'StationDensity': 'StationDensity',
                  'charge_nchannels_ratio': 'Charge/NChannels',
                  'stationdensity_charge_ratio': 'StationDensity/Charge',
                  'NHits_1_30': 'NHits',
                  'log_NHits_1_30': '$\log_{10}$(NHits (top 50\%))',
                  'charge_nhits_ratio': 'Charge/NHits',
                  'nchannels_nhits_ratio': 'NChannels/NHits',
                  'stationdensity_nchannels_ratio': 'StationDensity/NChannels',
                  'stationdensity_nhits_ratio': 'StationDensity/NHits'
                  }
    feature_labels = np.array([label_dict[feature]
                               for feature in feature_list])

    return feature_list, feature_labels


@export
def get_train_test_sets(df, feature_list, comp_class=False,
                        train_he=True, test_he=True, return_energy=False):

    # Load and preprocess training data
    if comp_class:
        X, y = df[feature_list].values, df.MC_comp_class.values
    else:
        X, y = df[feature_list].values, df.MC_comp.values
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

    if return_energy:
        energy_test = X_test[:, 0]
        return X_train, X_test, y_train, y_test, le, energy_test

    else:
        return X_train, X_test, y_train, y_test, le


@export
def preprocess_sim(config='IC79', comp_class=True, return_energy=False, f_selection='sffs'):

    # Load sim dataframe
    df = load_dataframe(datatype='sim', config=config)

    # Format testing and training sets (with full set of features)
    feature_list, feature_labels = get_training_features()
    X_train, X_test, y_train, y_test, le = get_train_test_sets(
        df, feature_list, comp_class=comp_class)

    # Extract reconstructed energy feature for later use
    energy_train = X_train[:, 0]
    energy_test = X_test[:, 0]

    # Deserialize feature selection algorithm
    if comp_class:
        with open(
        '../../analysis/lightheavy/feature-selection/{}_nfeatures_8.pkl'.format(f_selection), 'rb') as f_obj:
            sfs = cPickle.load(f_obj)
    else:
        with open(
        '../../analysis/fourcompositions/feature-selection/{}_nfeatures_8.pkl'.format(f_selection), 'rb') as f_obj:
            sfs = cPickle.load(f_obj)
    print('\nFeatures selected = {}\n'.format([feature_list[idx] for idx in sfs.k_feature_idx_]))

    # Feature transformation
    X_train_sfs = sfs.transform(X_train)
    X_test_sfs = sfs.transform(X_test)

    print('Number training events = ' + str(y_train.shape[0]))
    print('Number testing events = ' + str(y_test.shape[0]))

    if return_energy:
        return X_train_sfs, X_test_sfs, y_train, y_test, le, energy_train, energy_test
    else:
        return X_train_sfs, X_test_sfs, y_train, y_test, le


@export
def preprocess_data(config='IC79', comp_class=True, return_energy=False, f_selection='sffs'):

    # Load sim dataframe
    df = load_dataframe(datatype='data', config=config)

    # Format testing and training sets (with full set of features)
    feature_list, feature_labels = get_training_features()
    # Shuffle rows in data frame
    df = df.sample(frac=1).reset_index(drop=True)
    X_test = df[feature_list].values

    # Extract reconstructed energy feature for later use
    energy_test = X_test[:, 0]

    # Deserialize feature selection algorithm
    if comp_class:
        with open(
        '../../analysis/lightheavy/feature-selection/{}_nfeatures_8.pkl'.format(f_selection), 'rb') as f_obj:
            sfs = cPickle.load(f_obj)
    else:
        with open(
        '../../analysis/fourcompositions/feature-selection/{}_nfeatures_8.pkl'.format(f_selection), 'rb') as f_obj:
            sfs = cPickle.load(f_obj)
    print('\nFeatures selected = {}\n'.format([feature_list[idx] for idx in sfs.k_feature_idx_]))

    # Feature transformation
    X_test_sfs = sfs.transform(X_test)

    print('Number testing events = ' + str(X_test_sfs.shape[0]))

    if return_energy:
        return X_test_sfs, energy_test
    else:
        return X_test_sfs
