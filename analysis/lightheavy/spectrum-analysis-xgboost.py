#!/usr/bin/env python

from __future__ import division, print_function
from collections import defaultdict
import itertools
import numpy as np
from scipy import interp
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import seaborn.apionly as sns

from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc
from sklearn.model_selection import cross_val_score, StratifiedShuffleSplit, KFold, StratifiedKFold
from mlxtend.feature_selection import SequentialFeatureSelector as SFS

import composition as comp
import composition.analysis.plotting as plotting

color_dict = {'light': 'C0', 'heavy': 'C1', 'total': 'C2'}

X_train_sim, X_test_sim, y_train_sim, y_test_sim, le, energy_train_sim, energy_test_sim = comp.preprocess_sim(return_energy=True)
X_test_data, energy_test_data = comp.preprocess_data(return_energy=True)

# pipeline = comp.get_pipeline('xgboost')
# clf_name = pipeline.named_steps['classifier'].__class__.__name__
# print('=' * 30)
# print(clf_name)
# scores = cross_val_score(
#     estimator=pipeline, X=X_train_sim, y=y_train_sim, cv=3, n_jobs=15)
# print('CV score: {:.2%} (+/- {:.2%})'.format(scores.mean(), scores.std()))
# print('=' * 30)

# Define energy binning for this analysis
energybins = comp.analysis.get_energybins()

# # Calculate RF generalization error via 10-fold CV
# comp_list = ['light', 'heavy']
# # Split training data into CV training and testing folds
# kf = KFold(n_splits=10)
# frac_correct_folds = defaultdict(list)
# fold_num = 0
# print('Fold ', end='')
# for train_index, test_index in kf.split(X_train_sim):
#     fold_num += 1
#     print('{}...'.format(fold_num), end='')
#     X_train_fold, X_test_fold = X_train_sim[train_index], X_train_sim[test_index]
#     y_train_fold, y_test_fold = y_train_sim[train_index], y_train_sim[test_index]
#
#     energy_test_fold = energy_train_sim[test_index]
#
#     reco_frac, reco_frac_err = get_frac_correct(X_train_fold, X_test_fold,
#                                                 y_train_fold, y_test_fold,
#                                                 energy_test_fold, comp_list)
#     for composition in comp_list:
#         frac_correct_folds[composition].append(reco_frac[composition])
#     frac_correct_folds['total'].append(reco_frac['total'])
# frac_correct_gen_err = {key: np.std(frac_correct_folds[key], axis=0) for key in frac_correct_folds}

df_sim = comp.load_dataframe(datatype='sim', config='IC79')

# reco_frac, reco_frac_stat_err = get_frac_correct(X_train_sim, X_test_sim,
#                                                  y_train_sim, y_test_sim,
#                                                  energy_test_sim, comp_list)
# step_x = log_energy_midpoints
# step_x = np.append(step_x[0]-log_energy_bin_width/2, step_x)
# step_x = np.append(step_x, step_x[-1]+log_energy_bin_width/2)
# # Plot fraction of events correctlt classified vs energy
# fig, ax = plt.subplots()
# for composition in comp_list + ['total']:
#     err = np.sqrt(frac_correct_gen_err[composition]**2 + reco_frac_stat_err[composition]**2)
#     plotting.plot_steps(log_energy_midpoints, reco_frac[composition], err, ax, color_dict[composition], composition)
# plt.xlabel('$\log_{10}(E_{\mathrm{reco}}/\mathrm{GeV})$')
# ax.set_ylabel('Fraction correctly identified')
# ax.set_ylim([0.0, 1.0])
# ax.set_xlim([6.3, 8.0])
# ax.grid(linestyle=':')
# leg = plt.legend(loc='upper center', frameon=False,
#           bbox_to_anchor=(0.5,  # horizontal
#                           1.1),# vertical
#           ncol=len(comp_list)+1, fancybox=False)
# # set the linewidth of each legend object
# for legobj in leg.legendHandles:
#     legobj.set_linewidth(3.0)
#
# # place a text box in upper left in axes coords
# textstr = '$\mathrm{\underline{Training \ features}}$: \n'
# # for i, label in enumerate(feature_labels):
# # for i, idx in enumerate(sfs.k_feature_idx_):
# # #     if i>1:
# # #         break
# #     print(feature_labels[idx])
# # #     textstr += '{}) '.format(i+1) + feature_labels[idx] + '\n'
# #     if (i == len(feature_labels)-1):
# #         textstr += '{}) '.format(i+1) + feature_labels[idx]
# #     else:
# #         textstr += '{}) '.format(i+1) + feature_labels[idx] + '\n'
# props = dict(facecolor='white', linewidth=0)
# # ax.text(1.025, 0.855, textstr, transform=ax.transAxes, fontsize=8,
# #         verticalalignment='top', bbox=props)
# cv_str = 'Accuracy: {:0.2f}\% (+/- {:.1}\%)'.format(scores.mean()*100, scores.std()*100)
# # print(cvstr)
# # props = dict(facecolor='white', linewidth=0)
# # ax.text(1.025, 0.9825, cvstr, transform=ax.transAxes, fontsize=8,
# #         verticalalignment='top', bbox=props)
# ax.text(7.4, 0.2, cv_str,
#         ha="center", va="center", size=8,
#         bbox=dict(boxstyle='round', fc="white", ec="gray", lw=0.8))
# plt.show()
#
#
# # ## Spectrum
# # [ [back to top](#top) ]
#
# # In[11]:
#
# def get_num_comp_reco(X_train, y_train, X_test, log_energy_test, comp_list):
#
#     pipeline.fit(X_train, y_train)
#     test_predictions = pipeline.predict(X_test)
#
#     # Get number of correctly identified comp in each reco energy bin
#     num_reco_energy, num_reco_energy_err = {}, {}
#     for composition in comp_list:
#         num_reco_energy[composition] = np.histogram(
#             log_energy_test[le.inverse_transform(test_predictions) == composition],
#             bins=log_energy_bins)[0]
#         num_reco_energy_err[composition] = np.sqrt(num_reco_energy[composition])
#
#     num_reco_energy['total'] = np.histogram(log_energy_test, bins=log_energy_bins)[0]
#     num_reco_energy_err['total'] = np.sqrt(num_reco_energy['total'])
#
#     return num_reco_energy, num_reco_energy_err
#
#
# # In[ ]:
#
# df_sim = comp.load_dataframe(datatype='sim', config='IC79')
#
#
# # In[14]:
#
# comp_list = ['light', 'heavy']
# # Get number of events per energy bin
# num_reco_energy, num_reco_energy_err = get_num_comp_reco(X_train_sim, y_train_sim,
#                                                          X_test_data, energy_test_data,
#                                                          comp_list)
# import pprint
# pprint.pprint(num_reco_energy)
# print(np.sum(num_reco_energy['light']+num_reco_energy['heavy']))
# print(np.sum(num_reco_energy['total']))
# # Solid angle
# solid_angle = 2*np.pi*(1-np.cos(np.arccos(0.8)))
#
#
# # In[15]:
#
# # Live-time information
# goodrunlist = pd.read_table('/data/ana/CosmicRay/IceTop_GRL/IC79_2010_GoodRunInfo_4IceTop.txt', skiprows=[0, 3])
# goodrunlist.head()
#
#
# # In[16]:
#
# livetimes = goodrunlist['LiveTime(s)']
# livetime = np.sum(livetimes[goodrunlist['Good_it_L2'] == 1])
# print('livetime (seconds) = {}'.format(livetime))
# print('livetime (days) = {}'.format(livetime/(24*60*60)))
#
#
# # In[17]:
#
# fig, ax = plt.subplots()
# for composition in comp_list + ['total']:
#     # Calculate dN/dE
#     y = num_reco_energy[composition]
#     y_err = num_reco_energy_err[composition]
#     # Add time duration
#     y = y / livetime
#     y_err = y / livetime
# #     ax.errorbar(log_energy_midpoints, y, yerr=y_err,
# #                 color=color_dict[composition], label=composition,
# #                 marker='.', linestyle='None')
#     plotting.plot_steps(log_energy_midpoints, y, y_err, ax, color_dict[composition], composition)
# ax.set_yscale("log", nonposy='clip')
# plt.xlabel('$\log_{10}(E_{\mathrm{reco}}/\mathrm{GeV})$')
# ax.set_ylabel('Rate [s$^{-1}$]')
# ax.set_xlim([6.2, 8.0])
# # ax.set_ylim([10**2, 10**5])
# ax.grid(linestyle=':')
# leg = plt.legend(loc='upper center', frameon=False,
#           bbox_to_anchor=(0.5,  # horizontal
#                           1.1),# vertical
#           ncol=len(comp_list)+1, fancybox=False)
# # set the linewidth of each legend object
# for legobj in leg.legendHandles:
#     legobj.set_linewidth(3.0)
#
# plt.show()
#
#
# # In[18]:
#
# eff_area, eff_area_error, energy_midpoints = comp.analysis.get_effective_area(df_sim, energy_bins)
#
#
# # In[19]:
#
# # Plot fraction of events vs energy
# fig, ax = plt.subplots()
# for composition in comp_list + ['total']:
#     # Calculate dN/dE
#     y = num_reco_energy[composition]/energy_bin_widths
#     y_err = num_reco_energy_err[composition]/energy_bin_widths
#     # Add effective area
#     y, y_err = comp.analysis.ratio_error(y, y_err, eff_area, eff_area_error)
#     # Add solid angle
#     y = y / solid_angle
#     y_err = y_err / solid_angle
#     # Add time duration
#     y = y / livetime
#     y_err = y / livetime
#     # Add energy scaling
# #     energy_err = get_energy_res(df_sim, energy_bins)
# #     energy_err = np.array(energy_err)
# #     print(10**energy_err)
#     y = energy_midpoints**2.7 * y
#     y_err = energy_midpoints**2.7 * y_err
#     print(y)
#     print(y_err)
# #     ax.errorbar(log_energy_midpoints, y, yerr=y_err, label=composition, color=color_dict[composition],
# #            marker='.', markersize=8)
#     plotting.plot_steps(log_energy_midpoints, y, y_err, ax, color_dict[composition], composition)
# ax.set_yscale("log", nonposy='clip')
# # ax.set_xscale("log", nonposy='clip')
# plt.xlabel('$\log_{10}(E_{\mathrm{reco}}/\mathrm{GeV})$')
# ax.set_ylabel('$\mathrm{E}^{2.7} \\frac{\mathrm{dN}}{\mathrm{dE dA d\Omega dt}} \ [\mathrm{GeV}^{1.7} \mathrm{m}^{-2} \mathrm{sr}^{-1} \mathrm{s}^{-1}]$')
# ax.set_xlim([6.3, 8])
# ax.set_ylim([10**3, 10**5])
# ax.grid(linestyle='dotted', which="both")
# leg = plt.legend(loc='upper center', frameon=False,
#           bbox_to_anchor=(0.5,  # horizontal
#                           1.1),# vertical
#           ncol=len(comp_list)+1, fancybox=False)
# # set the linewidth of each legend object
# for legobj in leg.legendHandles:
#     legobj.set_linewidth(3.0)
#
# # plt.savefig('/home/jbourbeau/public_html/figures/spectrum.png')
# plt.show()
#
#
# # ## Unfolding
# # [ [back to top](#top) ]
#
# # In[20]:
#
# reco_frac['light']
#
#
# # In[21]:
#
# reco_frac['heavy']
#
#
# # In[22]:
#
# num_reco_energy['light']
#
#
# # In[23]:
#
# num_reco_energy['heavy']
#
#
# # In[24]:
#
# pipeline.fit(X_train_sim, y_train_sim)
# test_predictions = pipeline.predict(X_test_sim)
# true_comp = le.inverse_transform(y_test_sim)
# pred_comp = le.inverse_transform(test_predictions)
# print(true_comp)
# print(pred_comp)
#
#
# # In[25]:
#
# bin_idxs = np.digitize(energy_test_sim, log_energy_bins) - 1
# energy_bin_idx = np.unique(bin_idxs)
# energy_bin_idx = energy_bin_idx[1:]
# print(energy_bin_idx)
# num_reco_energy_unfolded = defaultdict(list)
# for bin_idx in energy_bin_idx:
#     energy_bin_mask = bin_idxs == bin_idx
#     confmat = confusion_matrix(true_comp[energy_bin_mask], pred_comp[energy_bin_mask], labels=comp_list)
#     confmat = np.divide(confmat.T, confmat.sum(axis=1, dtype=float)).T
#     inv_confmat = np.linalg.inv(confmat)
#     counts = np.array([num_reco_energy[composition][bin_idx] for composition in comp_list])
#     unfolded_counts = np.dot(inv_confmat, counts)
# #     unfolded_counts[unfolded_counts < 0] = 0
#     num_reco_energy_unfolded['light'].append(unfolded_counts[0])
#     num_reco_energy_unfolded['heavy'].append(unfolded_counts[1])
#     num_reco_energy_unfolded['total'].append(unfolded_counts.sum())
# print(num_reco_energy_unfolded)
#
#
# # In[26]:
#
# unfolded_counts.sum()
#
#
# # In[27]:
#
# fig, ax = plt.subplots()
# for composition in comp_list + ['total']:
#     # Calculate dN/dE
#     y = num_reco_energy_unfolded[composition]/energy_bin_widths
#     y_err = np.sqrt(y)/energy_bin_widths
#     # Add effective area
#     y, y_err = comp.analysis.ratio_error(y, y_err, eff_area, eff_area_error)
#     # Add solid angle
#     y = y / solid_angle
#     y_err = y_err / solid_angle
#     # Add time duration
#     y = y / livetime
#     y_err = y / livetime
#     # Add energy scaling
# #     energy_err = get_energy_res(df_sim, energy_bins)
# #     energy_err = np.array(energy_err)
# #     print(10**energy_err)
#     y = energy_midpoints**2.7 * y
#     y_err = energy_midpoints**2.7 * y_err
#     print(y)
#     print(y_err)
# #     ax.errorbar(log_energy_midpoints, y, yerr=y_err, label=composition, color=color_dict[composition],
# #            marker='.', markersize=8)
#     plotting.plot_steps(log_energy_midpoints, y, y_err, ax, color_dict[composition], composition)
# ax.set_yscale("log", nonposy='clip')
# # ax.set_xscale("log", nonposy='clip')
# plt.xlabel('$\log_{10}(E_{\mathrm{reco}}/\mathrm{GeV})$')
# ax.set_ylabel('$\mathrm{E}^{2.7} \\frac{\mathrm{dN}}{\mathrm{dE dA d\Omega dt}} \ [\mathrm{GeV}^{1.7} \mathrm{m}^{-2} \mathrm{sr}^{-1} \mathrm{s}^{-1}]$')
# ax.set_xlim([6.3, 8])
# ax.set_ylim([10**3, 10**5])
# ax.grid(linestyle='dotted', which="both")
# leg = plt.legend(loc='upper center', frameon=False,
#           bbox_to_anchor=(0.5,  # horizontal
#                           1.1),# vertical
#           ncol=len(comp_list)+1, fancybox=False)
# # set the linewidth of each legend object
# for legobj in leg.legendHandles:
#     legobj.set_linewidth(3.0)
#
# # plt.savefig('/home/jbourbeau/public_html/figures/spectrum.png')
# plt.show()
#
#
# # ### Iterative method
#
# # Get confusion matrix for each energy bin
#
# # In[99]:
#
# bin_idxs = np.digitize(energy_test_sim, log_energy_bins) - 1
# energy_bin_idx = np.unique(bin_idxs)
# energy_bin_idx = energy_bin_idx[1:]
# print(energy_bin_idx)
# num_reco_energy_unfolded = defaultdict(list)
# response_mat = []
# for bin_idx in energy_bin_idx:
#     energy_bin_mask = bin_idxs == bin_idx
#     confmat = confusion_matrix(true_comp[energy_bin_mask], pred_comp[energy_bin_mask], labels=comp_list)
#     confmat = np.divide(confmat.T, confmat.sum(axis=1, dtype=float)).T
#     response_mat.append(confmat)
#
#
# # In[100]:
#
# response_mat
#
#
# # In[134]:
#
# r = np.dstack((np.copy(num_reco_energy['light']), np.copy(num_reco_energy['heavy'])))[0]
# for unfold_iter in range(50):
#     print('Unfolding iteration {}...'.format(unfold_iter))
#     if unfold_iter == 0:
#         u = r
#     fs = []
#     for bin_idx in energy_bin_idx:
# #         print(u)
#         f = np.dot(response_mat[bin_idx], u[bin_idx])
#         f[f < 0] = 0
#         fs.append(f)
# #         print(f)
#     u = u + (r - fs)
# #     u[u < 0] = 0
# #     print(u)
# unfolded_counts_iter = {}
# unfolded_counts_iter['light'] = u[:,0]
# unfolded_counts_iter['heavy'] = u[:,1]
# unfolded_counts_iter['total'] = u.sum(axis=1)
# print(unfolded_counts_iter)
#
#
# # In[135]:
#
# fig, ax = plt.subplots()
# for composition in comp_list + ['total']:
#     # Calculate dN/dE
#     y = unfolded_counts_iter[composition]/energy_bin_widths
#     y_err = np.sqrt(y)/energy_bin_widths
#     # Add effective area
#     y, y_err = comp.analysis.ratio_error(y, y_err, eff_area, eff_area_error)
#     # Add solid angle
#     y = y / solid_angle
#     y_err = y_err / solid_angle
#     # Add time duration
#     y = y / livetime
#     y_err = y / livetime
#     # Add energy scaling
# #     energy_err = get_energy_res(df_sim, energy_bins)
# #     energy_err = np.array(energy_err)
# #     print(10**energy_err)
#     y = energy_midpoints**2.7 * y
#     y_err = energy_midpoints**2.7 * y_err
#     print(y)
#     print(y_err)
# #     ax.errorbar(log_energy_midpoints, y, yerr=y_err, label=composition, color=color_dict[composition],
# #            marker='.', markersize=8)
#     plotting.plot_steps(log_energy_midpoints, y, y_err, ax, color_dict[composition], composition)
# ax.set_yscale("log", nonposy='clip')
# # ax.set_xscale("log", nonposy='clip')
# plt.xlabel('$\log_{10}(E_{\mathrm{reco}}/\mathrm{GeV})$')
# ax.set_ylabel('$\mathrm{E}^{2.7} \\frac{\mathrm{dN}}{\mathrm{dE dA d\Omega dt}} \ [\mathrm{GeV}^{1.7} \mathrm{m}^{-2} \mathrm{sr}^{-1} \mathrm{s}^{-1}]$')
# ax.set_xlim([6.3, 8])
# ax.set_ylim([10**3, 10**5])
# ax.grid(linestyle='dotted', which="both")
# leg = plt.legend(loc='upper center', frameon=False,
#           bbox_to_anchor=(0.5,  # horizontal
#                           1.1),# vertical
#           ncol=len(comp_list)+1, fancybox=False)
# # set the linewidth of each legend object
# for legobj in leg.legendHandles:
#     legobj.set_linewidth(3.0)
#
# # plt.savefig('/home/jbourbeau/public_html/figures/spectrum.png')
# plt.show()
#
#
# # In[106]:
#
# print(num_reco_energy)
#
#
# # In[107]:
#
# comp_list = ['light', 'heavy']
# pipeline = comp.get_pipeline('RF')
# pipeline.fit(X_train_sim, y_train_sim)
# test_predictions = pipeline.predict(X_test_sim)
# # correctly_identified_mask = (test_predictions == y_test)
# # confmat = confusion_matrix(y_true=y_test, y_pred=y_pred)/len(y_pred)
# true_comp = le.inverse_transform(y_test_sim)
# pred_comp = le.inverse_transform(test_predictions)
# confmat = confusion_matrix(true_comp, pred_comp, labels=comp_list)
#
# def plot_confusion_matrix(cm, classes,
#                           normalize=False,
#                           title='Confusion matrix',
#                           cmap=plt.cm.Greens):
#     """
#     This function prints and plots the confusion matrix.
#     Normalization can be applied by setting `normalize=True`.
#     """
#     if normalize:
#         cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
#         print("Normalized confusion matrix")
#     else:
#         print('Confusion matrix, without normalization')
#
#     print(cm)
#
#     plt.imshow(cm, interpolation='None', cmap=cmap,
#                vmin=0, vmax=1.0)
#     plt.title(title)
#     plt.colorbar()
#     tick_marks = np.arange(len(classes))
#     plt.xticks(tick_marks, classes, rotation=45)
#     plt.yticks(tick_marks, classes)
#
#     thresh = cm.max() / 2.
#     for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
#         plt.text(j, i, '{:0.3f}'.format(cm[i, j]),
#                  horizontalalignment="center",
#                  color="white" if cm[i, j] > thresh else "black")
#
#     plt.tight_layout()
#     plt.ylabel('True composition')
#     plt.xlabel('Predicted composition')
#
# fig, ax = plt.subplots()
# plot_confusion_matrix(confmat, classes=['light', 'heavy'], normalize=True,
#                       title='Confusion matrix, without normalization')
#
# # # Plot normalized confusion matrix
# # plt.figure()
# # plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
# #                       title='Normalized confusion matrix')
#
# plt.show()
#
#
# # In[63]:
#
# comp_list = ['light', 'heavy']
# pipeline = comp.get_pipeline('RF')
# pipeline.fit(X_train_sim, y_train_sim)
# test_predictions = pipeline.predict(X_test_sim)
# # correctly_identified_mask = (test_predictions == y_test)
# # confmat = confusion_matrix(y_true=y_test, y_pred=y_pred)/len(y_pred)
# true_comp = le.inverse_transform(y_test_sim)
# pred_comp = le.inverse_transform(test_predictions)
# confmat = confusion_matrix(true_comp, pred_comp, labels=comp_list)
#
# inverse = np.linalg.inv(confmat)
# inverse
#
#
# # In[64]:
#
# confmat
#
#
# # In[66]:
#
# comp_list = ['light', 'heavy']
# # Get number of events per energy bin
# num_reco_energy, num_reco_energy_err = get_num_comp_reco(X_train_sim, y_train_sim, X_test_data, comp_list)
# # Energy-related variables
# energy_bin_width = 0.1
# energy_bins = np.arange(6.2, 8.1, energy_bin_width)
# energy_midpoints = (energy_bins[1:] + energy_bins[:-1]) / 2
# energy_bin_widths = 10**energy_bins[1:] - 10**energy_bins[:-1]
# def get_energy_res(df_sim, energy_bins):
#     reco_log_energy = df_sim['lap_log_energy'].values
#     MC_log_energy = df_sim['MC_log_energy'].values
#     energy_res = reco_log_energy - MC_log_energy
#     bin_centers, bin_medians, energy_err = comp.analysis.data_functions.get_medians(reco_log_energy,
#                                                                                energy_res,
#                                                                                energy_bins)
#     return np.abs(bin_medians)
# # Solid angle
# solid_angle = 2*np.pi*(1-np.cos(np.arccos(0.85)))
# # solid_angle = 2*np.pi*(1-np.cos(40*(np.pi/180)))
# print(solid_angle)
# print(2*np.pi*(1-np.cos(40*(np.pi/180))))
# # Live-time information
# start_time = np.amin(df_data['start_time_mjd'].values)
# end_time = np.amax(df_data['end_time_mjd'].values)
# day_to_sec = 24 * 60 * 60.
# dt = day_to_sec * (end_time - start_time)
# print(dt)
# # Plot fraction of events vs energy
# fig, ax = plt.subplots()
# for i, composition in enumerate(comp_list):
#     num_reco_bin = np.array([[i, j] for i, j in zip(num_reco_energy['light'], num_reco_energy['heavy'])])
# #     print(num_reco_bin)
#     num_reco = np.array([np.dot(inverse, i) for i in num_reco_bin])
#     print(num_reco)
#     num_reco_2 = {'light': num_reco[:, 0], 'heavy': num_reco[:, 1]}
#     # Calculate dN/dE
#     y = num_reco_2[composition]/energy_bin_widths
#     y_err = num_reco_energy_err[composition]/energy_bin_widths
#     # Add effective area
#     y, y_err = comp.analysis.ratio_error(y, y_err, eff_area, eff_area_error)
#     # Add solid angle
#     y = y / solid_angle
#     y_err = y_err / solid_angle
#     # Add time duration
#     y = y / dt
#     y_err = y / dt
#     # Add energy scaling
#     energy_err = get_energy_res(df_sim, energy_bins)
#     energy_err = np.array(energy_err)
# #     print(10**energy_err)
#     y = (10**energy_midpoints)**2.7 * y
#     y_err = (10**energy_midpoints)**2.7 * y_err
#     plotting.plot_steps(energy_midpoints, y, y_err, ax, color_dict[composition], composition)
# ax.set_yscale("log", nonposy='clip')
# plt.xlabel('$\log_{10}(E_{\mathrm{reco}}/\mathrm{GeV})$')
# ax.set_ylabel('$\mathrm{E}^{2.7} \\frac{\mathrm{dN}}{\mathrm{dE dA d\Omega dt}} \ [\mathrm{GeV}^{1.7} \mathrm{m}^{-2} \mathrm{sr}^{-1} \mathrm{s}^{-1}]$')
# ax.set_xlim([6.2, 8.0])
# # ax.set_ylim([10**2, 10**5])
# ax.grid()
# leg = plt.legend(loc='upper center',
#           bbox_to_anchor=(0.5,  # horizontal
#                           1.1),# vertical
#           ncol=len(comp_list)+1, fancybox=False)
# # set the linewidth of each legend object
# for legobj in leg.legendHandles:
#     legobj.set_linewidth(3.0)
#
# plt.show()
#
#
# # In[44]:
#
# pipeline.get_params()['classifier__max_depth']
#
#
# # In[47]:
#
# energy_bin_width = 0.1
# energy_bins = np.arange(6.2, 8.1, energy_bin_width)
# fig, axarr = plt.subplots(1, 2)
# for composition, ax in zip(comp_list, axarr.flatten()):
#     MC_comp_mask = (df_sim['MC_comp_class'] == composition)
#     MC_log_energy = df_sim['MC_log_energy'][MC_comp_mask].values
#     reco_log_energy = df_sim['lap_log_energy'][MC_comp_mask].values
#     plotting.histogram_2D(MC_log_energy, reco_log_energy, energy_bins, log_counts=True, ax=ax)
#     ax.plot([0,10], [0,10], marker='None', linestyle='-.')
#     ax.set_xlim([6.2, 8])
#     ax.set_ylim([6.2, 8])
#     ax.set_xlabel('$\log_{10}(E_{\mathrm{MC}}/\mathrm{GeV})$')
#     ax.set_ylabel('$\log_{10}(E_{\mathrm{reco}}/\mathrm{GeV})$')
#     ax.set_title('{} response matrix'.format(composition))
# plt.tight_layout()
# plt.show()
#
#
# # In[10]:
#
# energy_bins = np.arange(6.2, 8.1, energy_bin_width)
# 10**energy_bins[1:] - 10**energy_bins[:-1]
#
#
# # In[ ]:
#
# probs = pipeline.named_steps['classifier'].predict_proba(X_test)
# prob_1 = probs[:, 0][MC_iron_mask]
# prob_2 = probs[:, 1][MC_iron_mask]
# # print(min(prob_1-prob_2))
# # print(max(prob_1-prob_2))
# # plt.hist(prob_1-prob_2, bins=30, log=True)
# plt.hist(prob_1, bins=np.linspace(0, 1, 50), log=True)
# plt.hist(prob_2, bins=np.linspace(0, 1, 50), log=True)
#
#
# # In[ ]:
#
# probs = pipeline.named_steps['classifier'].predict_proba(X_test)
# dp1 = (probs[:, 0]-probs[:, 1])[MC_proton_mask]
# print(min(dp1))
# print(max(dp1))
# dp2 = (probs[:, 0]-probs[:, 1])[MC_iron_mask]
# print(min(dp2))
# print(max(dp2))
# fig, ax = plt.subplots()
# # plt.hist(prob_1-prob_2, bins=30, log=True)
# counts, edges, pathes = plt.hist(dp1, bins=np.linspace(-1, 1, 100), log=True, label='Proton', alpha=0.75)
# counts, edges, pathes = plt.hist(dp2, bins=np.linspace(-1, 1, 100), log=True, label='Iron', alpha=0.75)
# plt.legend(loc=2)
# plt.show()
# pipeline.named_steps['classifier'].classes_
#
#
# # In[ ]:
#
# print(pipeline.named_steps['classifier'].classes_)
# le.inverse_transform(pipeline.named_steps['classifier'].classes_)
#
#
# # In[ ]:
#
# pipeline.named_steps['classifier'].decision_path(X_test)
#
#
# # In[48]:
#
# comp_list = ['light', 'heavy']
# pipeline = comp.get_pipeline('RF')
# pipeline.fit(X_train_sim, y_train_sim)
# # test_probs = defaultdict(list)
# fig, ax = plt.subplots()
# test_predictions = pipeline.predict(X_test_data)
# test_probs = pipeline.predict_proba(X_test_data)
# for class_ in pipeline.classes_:
#     test_predictions == le.inverse_transform(class_)
#     plt.hist(test_probs[:, class_], bins=np.linspace(0, 1, 50),
#              histtype='step', label=composition,
#              color=color_dict[composition], alpha=0.8, log=True)
# plt.ylabel('Counts')
# plt.xlabel('Testing set class probabilities')
# plt.legend()
# plt.grid()
# plt.show()
#
#
# # In[5]:
#
# pipeline = comp.get_pipeline('RF')
# pipeline.fit(X_train, y_train)
# test_predictions = pipeline.predict(X_test)
#
# comp_list = ['P', 'He', 'O', 'Fe']
# fig, ax = plt.subplots()
# test_probs = pipeline.predict_proba(X_test)
# fig, axarr = plt.subplots(2, 2, sharex=True, sharey=True)
# for composition, ax in zip(comp_list, axarr.flatten()):
#     comp_mask = (le.inverse_transform(y_test) == composition)
#     probs = np.copy(test_probs[comp_mask])
#     print('probs = {}'.format(probs.shape))
#     weighted_mass = np.zeros(len(probs))
#     for class_ in pipeline.classes_:
#         c = le.inverse_transform(class_)
#         weighted_mass += comp.simfunctions.comp2mass(c) * probs[:, class_]
#     print('min = {}'.format(min(weighted_mass)))
#     print('max = {}'.format(max(weighted_mass)))
#     ax.hist(weighted_mass, bins=np.linspace(0, 5, 100),
#              histtype='step', label=None, color='darkgray',
#              alpha=1.0, log=False)
#     for c in comp_list:
#         ax.axvline(comp.simfunctions.comp2mass(c), color=color_dict[c],
#                    marker='None', linestyle='-')
#     ax.set_ylabel('Counts')
#     ax.set_xlabel('Weighted atomic number')
#     ax.set_title('MC {}'.format(composition))
#     ax.grid()
# plt.tight_layout()
# plt.show()
#
#
# # In[15]:
#
# pipeline = comp.get_pipeline('RF')
# pipeline.fit(X_train, y_train)
# test_predictions = pipeline.predict(X_test)
#
# comp_list = ['P', 'He', 'O', 'Fe']
# fig, ax = plt.subplots()
# test_probs = pipeline.predict_proba(X_test)
# fig, axarr = plt.subplots(2, 2, sharex=True, sharey=True)
# for composition, ax in zip(comp_list, axarr.flatten()):
#     comp_mask = (le.inverse_transform(y_test) == composition)
#     probs = np.copy(test_probs[comp_mask])
#     weighted_mass = np.zeros(len(probs))
#     for class_ in pipeline.classes_:
#         c = le.inverse_transform(class_)
#         ax.hist(probs[:, class_], bins=np.linspace(0, 1, 50),
#                  histtype='step', label=c, color=color_dict[c],
#                  alpha=1.0, log=True)
#     ax.legend(title='Reco comp', framealpha=0.5)
#     ax.set_ylabel('Counts')
#     ax.set_xlabel('Testing set class probabilities')
#     ax.set_title('MC {}'.format(composition))
#     ax.grid()
# plt.tight_layout()
# plt.show()
#
#
# # In[25]:
#
# comp_list = ['light', 'heavy']
# test_probs = defaultdict(list)
# fig, ax = plt.subplots()
# # test_probs = pipeline.predict_proba(X_test)
# for event in pipeline.predict_proba(X_test_data):
#     composition = le.inverse_transform(np.argmax(event))
#     test_probs[composition].append(np.amax(event))
# for composition in comp_list:
#     plt.hist(test_probs[composition], bins=np.linspace(0, 1, 100),
#              histtype='step', label=composition,
#              color=color_dict[composition], alpha=0.8, log=False)
# plt.ylabel('Counts')
# plt.xlabel('Testing set class probabilities')
# plt.legend(title='Reco comp')
# plt.grid()
# plt.show()
#
#
# # In[ ]:
#
#
#
#
# # In[ ]:
#
#
#
