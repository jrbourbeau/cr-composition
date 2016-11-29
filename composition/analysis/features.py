#!/usr/bin/env python

import numpy as np
from . import export

@export
def get_training_features():
    # feature_list = np.array(['MC_log_energy', 'InIce_log_charge'])
    # feature_list = np.array(['reco_log_energy', 'InIce_log_charge'])
    # feature_list = np.array(['reco_log_energy', 'InIce_log_charge', 'reco_cos_zenith'])
    # feature_list = ['reco_log_energy', 'InIce_log_charge_1_60', 'reco_cos_zenith', 'lap_chi2', 'NChannels_1_60']
    # feature_list = ['reco_log_energy', 'InIce_log_charge_1_45', 'reco_cos_zenith', 'lap_chi2', 'NChannels_1_45']
    # feature_list = ['reco_log_energy', 'InIce_log_charge_1_30', 'reco_cos_zenith', 'lap_chi2', 'NChannels_1_30']
    # feature_list = ['reco_log_energy', 'InIce_log_charge_1_15', 'reco_cos_zenith', 'lap_chi2', 'NChannels_1_15']
    feature_list = ['reco_log_energy', 'InIce_log_charge_1_6', 'reco_cos_zenith', 'lap_chi2', 'NChannels_1_6']
    # feature_list = ['reco_log_energy', 'InIce_log_charge_1_6', 'reco_cos_zenith', 'lap_chi2', 'NChannels_1_6', 'log_s125']

    label_dict = {'reco_log_energy': '$\log_{10}(E_{\mathrm{reco}}/\mathrm{GeV})$',
                  'log_s125': '$\log_{10}(S_{\mathrm{125}})$',
                  'InIce_log_charge_1_60': 'InIce charge',
                  'InIce_log_charge_1_45': 'InIce charge (top 75\%)',
                  'InIce_log_charge_1_30': 'InIce charge (top 50\%)',
                  'InIce_log_charge_1_15': 'InIce charge (top 25\%)',
                  'InIce_log_charge_1_6': 'InIce charge (top 10\%)',
                  'reco_cos_zenith': '$\cos(\\theta_{\mathrm{reco}})$',
                  'lap_chi2': '$\chi^2_{\mathrm{Lap}}/\mathrm{n.d.f}$',
                  'NChannels_1_60': 'NChannels',
                  'NChannels_1_45': 'NChannels (top 75\%)',
                  'NChannels_1_30': 'NChannels (top 50\%)',
                  'NChannels_1_15': 'NChannels (top 25\%)',
                  'NChannels_1_6': 'NChannels (top 10\%)'}
    feature_labels = np.array([label_dict[feature] for feature in feature_list])

    return feature_list, feature_labels
