#!/usr/bin/env python

import numpy as np

def get_training_features():
    # feature_list = np.array(['MC_log_energy', 'InIce_log_charge'])
    # feature_list = np.array(['reco_log_energy', 'InIce_log_charge'])
    # feature_list = np.array(['reco_log_energy', 'InIce_log_charge', 'reco_cos_zenith'])
    feature_list = ['reco_log_energy', 'InIce_log_charge_1_60', 'reco_cos_zenith', 'lap_chi2']

    return feature_list
