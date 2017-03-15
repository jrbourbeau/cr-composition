#!/usr/bin/env python

import numpy as np
from . import export

@export
def get_training_features():

    feature_list = ['lap_log_energy', 'lap_cos_zenith', 'log_NChannels_1_30',
                    'nchannels_nhits_ratio', 'rlogl', 'log_NHits_1_30',
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
                  'rlogl': '$r\log_{10}(l)$',
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
                  'log_NChannels_1_30' : '$\log_{10}$(NChannels (top 50\%))',
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
    feature_labels = np.array([label_dict[feature] for feature in feature_list])

    return feature_list, feature_labels
