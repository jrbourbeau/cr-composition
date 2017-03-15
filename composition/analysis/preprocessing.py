#!/usr/bin/env python

import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.cluster import KMeans
from . import export
from ..dataframe_functions import load_dataframe
from .base import DataSet


@export
def get_training_features():

    # Features used in the 3-year analysis
    feature_list = ['lap_cos_zenith', 'log_s125', 'log_dEdX']
    # feature_list = ['lap_cos_zenith', 'log_s125', 'log_dEdX', 'avg_inice_radius']
    # feature_list = ['lap_cos_zenith', 'log_s125', 'eloss_1500_standard']
    # feature_list = ['lap_cos_zenith', 'log_s125', 'eloss_1500_standard', 'num_millipede_particles']
    # feature_list = ['lap_cos_zenith', 'log_s125', 'eloss_1500_standard',
    #                 'n_he_stoch_standard', 'n_he_stoch_standard']


    # feature_list = ['lap_cos_zenith', 'log_s125', 'InIce_log_charge_1_30',
    #     'charge_nchannels_ratio', 'nhits_nchannels_ratio',
    #     'eloss_1500_standard']
    # feature_list = ['lap_log_energy', 'lap_cos_zenith', 'log_NChannels_1_30',
    #                 'nhits_nchannels_ratio', 'lap_rlogl', 'log_NHits_1_30',
    #                 'StationDensity', 'stationdensity_charge_ratio',
    #                 'log_s50', 'log_s125', 'log_s500',
    #                 'eloss_1500_standard', 'n_he_stoch_standard', 'n_he_stoch_standard',
    #                 '']

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
                  'lap_rlogl': '$r\log_{10}(l)$',
                  'lap_beta': 'lap beta',
                  'InIce_log_charge_1_60': 'InIce charge',
                  'InIce_log_charge_1_45': 'InIce charge (top 75\%)',
                  'InIce_charge_1_30': 'InIce charge (top 50\%)',
                  'InIce_log_charge_1_30': '$\log_{10}(InIce charge (top 50))$',
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
                  'nhits_nchannels_ratio': 'NHits/NChannels',
                  'stationdensity_nchannels_ratio': 'StationDensity/NChannels',
                  'stationdensity_nhits_ratio': 'StationDensity/NHits',
                  'llhratio': 'llhratio',
                  'n_he_stoch_standard': 'Num HE stochastics (standard)',
                  'n_he_stoch_strong': 'Num HE stochastics (strong)',
                  'eloss_1500_standard': 'dE/dX (standard)',
                  'log_dEdX': '$\log_{10}$(dE/dX)',
                  'eloss_1500_strong': 'dE/dX (strong)',
                  'num_millipede_particles': '$N_{\mathrm{mil}}$',
                  'avg_inice_radius': '$R_{\mathrm{core}}$',
                  'Laputop_InIce_FractionContainment': '$C_{\mathrm{IC}}$',
                  'max_inice_radius': '$R_{\mathrm{max}}$',
                  'invcharge_inice_radius': '$R_{\mathrm{q,core}}$'
                  }
    feature_labels = np.array([label_dict[feature]
                               for feature in feature_list])

    return feature_list, feature_labels


@export
def get_sim_datasets(df, feature_list, comp_class=False, return_energy=False):

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

    return sim_train, sim_test


# def deserialize_feature_selector(comp_class=True):
#     # Deserialize feature selection algorithm
#     if comp_class:
#         with open(
#                 '/home/jbourbeau/cr-composition/analysis/lightheavy/feature-selection/sfs_nfeatures_7_xgboost.pkl', 'rb') as f_obj:
#             sfs = cPickle.load(f_obj)
#     else:
#         with open(
#                 '../../analysis/fourcompositions/feature-selection/{}_nfeatures_8.pkl'.format(f_selection), 'rb') as f_obj:
#             sfs = cPickle.load(f_obj)
#
#     return sfs


@export
def preprocess_sim(config='IC79', comp_class=True, return_energy=False,
                   features=None, f_selection=None, cluster=False):

    # Load simulation dataframe
    df = load_dataframe(datatype='sim', config=config)

    # Format testing and training sets
    feature_list, feature_labels = get_training_features()
    features_str = ''
    for label in feature_labels:
        features_str += label + '\n\t'
    print('Selecting the following features:\n\t'+features_str)
    sim_train, sim_test = get_sim_datasets(df, feature_list,
        comp_class=comp_class, return_energy=return_energy)

    # if cluster:
    #     kmeans = KMeans(n_clusters=2)
    #     pred = kmeans.fit_predict(sim_train.X)

    # if f_selection:
    #     # Deserialize feature selection algorithm
    #     sfs = deserialize_feature_selector(comp_class)
    #     print('\nFeatures selected = {}\n'.format(
    #         [feature_list[idx] for idx in sfs.k_feature_idx_]))
    #
    #     # Feature transformation
    #     sim_train.X = sfs.transform(sim_train.X)
    #     sim_test.X = sfs.transform(sim_test.X)

    print('Number training events = {}'.format(len(sim_train)))
    print('Number testing events = {}'.format(len(sim_test)))

    return sim_train, sim_test


@export
def preprocess_data(config='IC79', comp_class=True, return_energy=False,
                   features=None, f_selection=None):

    # Load sim dataframe
    df = load_dataframe(datatype='data', config=config)
    # Format testing set
    feature_list, feature_labels = get_training_features()
    features_str = ''
    for label in feature_labels:
        features_str += label + '\n\t'
    print('Selecting the following features:\n\t'+features_str)
    X_test = df[feature_list].values
    # n_events = X_test.shape[0]
    # # Shuffle rows in test data
    # shuffle_array = np.random.choice(n_events, size=n_events, replace=False)
    # X_test = X_test[shuffle_array]


    # if f_selection:
    #     # Deserialize feature selection algorithm
    #     sfs = deserialize_feature_selector(comp_class)
    #     print('\nFeatures selected = {}\n'.format(
    #         [feature_list[idx] for idx in sfs.k_feature_idx_]))
    #
    #     # Feature transformation
    #     X_test = sfs.transform(X_test)

    print('Number testing events = ' + str(X_test.shape[0]))

    dataset = DataSet(X=X_test)
    if return_energy:
        # IMPORTANT: shuffle energys with same shuffle array used to randomize
        dataset.log_energy = df['lap_log_energy'].values

    return dataset
