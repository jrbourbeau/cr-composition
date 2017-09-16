#!/usr/bin/env python

import numpy as np
from . import export

@export
def get_training_features(feature_list=None):

    # Features used in the 3-year analysis
    if feature_list is None:
        feature_list = ['lap_cos_zenith', 'log_s125', 'log_dEdX']
        # feature_list = ['lap_cos_zenith', 'log_s125', 'log_dEdX', 'log_d4r_peak_energy', 'log_d4r_peak_sigma']
        # feature_list = ['lap_cos_zenith', 'log_s125', 'log_dEdX', 'median_inice_radius', 'd4r_peak_energy']
        # feature_list = ['lap_cos_zenith', 'log_s125', 'log_dEdX', 'FractionContainment_Laputop_InIce']
        # feature_list = ['lap_cos_zenith', 'log_s125', 'log_dEdX', 'avg_inice_radius']
    # feature_list = ['lap_cos_zenith', 'log_s125', 'log_dEdX', 'max_inice_radius']
    # feature_list = ['lap_cos_zenith', 'log_s125', 'log_dEdX', 'avg_inice_radius']

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
                  'log_dEdX': '$\mathrm{\log_{10}(dE/dX)}$',
                  'eloss_1500_strong': 'dE/dX (strong)',
                  'num_millipede_particles': '$N_{\mathrm{mil}}$',
                  'avg_inice_radius': '$\mathrm{R_{\mu \ bundle}}$',
                  'invqweighted_inice_radius_1_60': '$\mathrm{R_{\mu \ bundle}}$',
                  'avg_inice_radius_1_60': '$\mathrm{R_{\mu \ bundle}}$',
                  'avg_inice_radius_Laputop': '$R_{\mathrm{core, Lap}}$',
                  'FractionContainment_Laputop_InIce': '$C_{\mathrm{IC}}$',
                  'Laputop_IceTop_FractionContainment': '$C_{\mathrm{IT}}$',
                  'max_inice_radius': '$R_{\mathrm{max}}$',
                  'invcharge_inice_radius': '$R_{\mathrm{q,core}}$',
                  'lap_zenith': 'zenith',
                  'NStations': 'NStations',
                  'IceTop_charge': 'IT charge',
                  'IceTop_charge_175m': 'Signal greater 175m',
                  'log_IceTop_charge_175m': '$\log_{10}(Q_{IT, 175})$',
                  'IT_charge_ratio': 'IT charge ratio',
                  'refit_beta': '$\mathrm{\\beta_{refit}}$',
                  'log_d4r_peak_energy': '$\mathrm{\log_{10}(E_{D4R})}$',
                  'log_d4r_peak_sigma': '$\mathrm{\log_{10}(\sigma E_{D4R})}$',
                  'd4r_N': 'D4R N',
                  'median_inice_radius': 'Median InIce'
                  }

    feature_labels = np.array([label_dict[feature]
                               for feature in feature_list])

    return feature_list, feature_labels
