
#############################################################################
# Functions associated with finding and analyzing the effective area
#############################################################################

from __future__ import division
import collections
import numpy as np
from scipy import optimize

from icecube.weighting.weighting import from_simprod

from ..simfunctions import get_level3_sim_files, sim_to_comp
from . import export

@export
def effective_area(df, log_energy_bins):

    # MC_energy = df['MC_energy'].values
    # log_MC_energy = np.log10(MC_energy)
    # # Throw out particles outside energy range
    # mask = (log_MC_energy > log_energy_bins[0]) & (log_MC_energy < log_MC_energy[-1])
    # MC_energy = MC_energy[mask]
    # log_MC_energy = log_MC_energy[mask]
    log_MC_energy = df['MC_log_energy'].values
    print(log_MC_energy)

    sim_id = df['sim'].values
    # sim_id = df[mask]['sim'].values
    num_ptype = len(np.unique(df['MC_type']))

    # Set radii for finding effective area
    energy_range_to_radii = np.array([800, 1100, 1700, 2600, 2900])
    energy_breaks = np.array([5, 6, 7, 8, 9])
    event_energy_bins = np.digitize(log_MC_energy, energy_breaks) - 1

    weights = []
    resamples = 100
    for energy_bin_idx, sim in zip(event_energy_bins, sim_id):
        if energy_bin_idx < 0 or energy_bin_idx > len(energy_range_to_radii): continue
        weight = np.pi * energy_range_to_radii[energy_bin_idx]**2
        if sim in ['7579', '7791', '7851', '7784']:
            num_thrown = 480*resamples
        else:
            if sim == '7262':
                num_thrown = (19999/20000)*1000*resamples
            elif sim == '7263':
                num_thrown = (19998/20000)*1000*resamples
            else:
                num_thrown = 1000*resamples
        if energy_bin_idx == 2:
            weight = weight/(2*num_thrown)
        else:
            weight = weight/num_thrown
        weights.append(weight)
    print(np.array(weights))

    energy_hist = np.histogram(log_MC_energy, bins=log_energy_bins, weights=weights)[0]
    error_hist = np.sqrt(np.histogram(log_MC_energy, bins=log_energy_bins,
                                weights=np.array(weights)**2)[0])

    eff_area = energy_hist/num_ptype
    eff_area_error = error_hist/num_ptype

    # eff_area  = (10**-6)*E_hist.bincontent/n_gen
    # eff_area_error = (10**-6)*np.sqrt(error_hist.bincontent)/n_gen

    return eff_area, eff_area_error


@export
def get_effective_area(df_sim, energy_bins, energy='MC', verbose=True):

    assert energy in ['MC', 'reco'], 'energy must be MC or reco'

    if verbose: print('Calculating effective area...')
    simlist = np.unique(df_sim['sim'])
    print('simlist = {}'.format(simlist))
    # Get the number of times each composition is present
    comp_counter = collections.Counter([sim_to_comp(sim) for sim in simlist])
    print('comp_counter = {}'.format(comp_counter))
    for i, sim in enumerate(simlist):
        gcd_file, sim_files = get_level3_sim_files(sim)
        num_files = len(sim_files)
        if verbose: print('Simulation set {}: {} files'.format(sim, num_files))
        composition = sim_to_comp(sim)
        if i == 0:
            generator = num_files*from_simprod(int(sim))
        else:
            generator += num_files*from_simprod(int(sim))
    energy_key = 'MC_energy' if energy == 'MC' else 'lap_energy'
    energy = df_sim[energy_key].values
    ptype = df_sim['MC_type'].values
    num_ptypes = np.unique(ptype).size
    print('num_ptypes = {}'.format(num_ptypes))
    cos_theta = np.cos(df_sim['MC_zenith']).values
    areas = 1.0/generator(energy, ptype, cos_theta)
    # binwidth = 2*np.pi*(1-np.cos(40*(np.pi/180)))*np.diff(energy_bins)
    binwidth = 2*np.pi*(1-np.cos(40*(np.pi/180)))*np.diff(energy_bins)*num_ptypes
    eff_area = np.histogram(energy, weights=areas, bins=energy_bins)[0]/binwidth
    eff_area_error = np.sqrt(np.histogram(energy,
            bins=energy_bins, weights=areas**2)[0])/binwidth

    energy_midpoints = (energy_bins[1:] + energy_bins[:-1]) / 2

    return eff_area, eff_area_error, energy_midpoints


def sigmoid(energy, p0, p1, p2):
    return p0 / (1 + np.exp(-p1*np.log10(energy) + p2))


def get_sigmoid_params(df_sim, energy_bins):

    eff_area, eff_area_error, energy_midpoints = get_effective_area(df_sim, energy_bins, verbose=True)

    p_init = [1.4e5, 8.0, 50.0]
    print(energy_midpoints)
    eff_area_init = sigmoid(energy_midpoints, *p_init)
    print(eff_area)
    print(eff_area_init)

    # Fit with error bars
    # popt, pcov = optimize.curve_fit(sigmoid, np.log10(energy_midpoints), eff_area)
    popt, pcov = optimize.curve_fit(sigmoid, energy_midpoints, eff_area, sigma=eff_area_error)
    # popt, pcov = optimize.curve_fit(sigmoid, energy_midpoints, eff_area, p0=p_init, sigma=eff_area_error)
    eff_area_fit = sigmoid(energy_midpoints, *popt)
    print(eff_area_fit)
    chi = np.sum(((eff_area_fit - eff_area)/eff_area_error) ** 2) / len(energy_midpoints)
    print('ppot = {}'.format(popt))
    print('chi2 = {}'.format(chi))

    return popt, pcov
    #
    # plt.title('Fitting Effective Area ('+comp+')')
    # plt.xlabel('Log10(Energy/GeV)')
    # plt.ylabel('Effective Area (m**2)')
    #
    # plt.errorbar(x, y, yerr=sigma, fmt='.', label='data')
    # plt.plot(x, yfit, 'k', label='fit')
    # plt.plot(x, yfit2, 'g', label='fit2')
    # plt.plot(x, yguess, 'r', label='guess')
    # plt.legend(loc='upper left')
    # plt.grid(True)
    # #plt.yscale('log')
    #
    # plt.show()

# def line_fit(eff_area, eff_area_error, energy_midpoints):
#     """ Flat line fit for effective area at high energies """
#     high_energy_mask = (energy_midpoints >= 10**6.2)
#     # def linefit(x, m, b):
#     #     return m * x + b
#     linefit = lambda x, b: b
#     x = energy_midpoints[high_energy_mask]
#     y = eff_area[high_energy_mask]
#     sigma = eff_area_error[high_energy_mask]
#     popt, pcov = optimize.curve_fit(linefit,
#             x, y, sigma=sigma)
#     intercept = popt[0]
#     yfit = linefit(x, intercept)
#     yfit = np.array([yfit for i in range(len(x))])
#     chi2 = np.sum(((yfit - y)/sigma) **2) / len(x)
#     print('Chi2 = {}'.format(chi2))
#
#     return intercept
#
# # def get_effective_area(sim_reco, cut_mask):
#
#
# #     log_energy = np.log10(sim_reco['MC_energy'])
# #     ebins = np.arange(4, 9.51, 0.05)
# #     emids = getMids(ebins)
# #     # Set radii for finding effective area
# #     energy_ranges = ['low', 'mid', 'high']
# #     energy_range_to_sim = get_energy_range_to_sim()
# #     # Set radii for finding effective area
# #     energy_range_to_radii = {}
# #     for energy_range in energy_ranges:
# #         energy_range_to_radii[energy_range] = np.array([600, 800, 1100, 1700, 2600, 2900])
# #     energy_range_to_radii['low'][1] = 600
# #     energy_breaks = np.array([4, 5, 6, 7, 8, 9])
# #     rgrp = np.digitize(emids, energy_breaks) - 1
#
# #     eff, sig, relerr = {}, {}, {}
# #     for energy_range in energy_ranges:
#
# #         # Get efficiency and sigma
# #         erange_mask = np.array([sim in energy_range_to_sim[energy_range]
# #             for sim in sim_reco['sim']])
# #         observed_events = np.histogram(log_energy[cut_mask & erange_mask],
# #                 bins=ebins)[0]
# #         thrown_events = np.histogram(log_energy[erange_mask], bins=ebins)[0]
# #         with np.errstate(divide='ignore', invalid='ignore'):
# #             eff[energy_range] = observed_events / thrown_events
# #             var = (observed_events+1)*(observed_events+2)/((thrown_events+2)*(thrown_events+3)) - (observed_events+1)**2/((thrown_events+2)**2)
# #         sig[energy_range] = np.sqrt(var)
#
# #         # Multiply by throw area
# #         thrown_radius = np.array([energy_range_to_radii[energy_range][i]
# #             for i in rgrp])
# #         eff[energy_range] *= np.pi*(thrown_radius**2)
# #         sig[energy_range] *= np.pi*(thrown_radius**2)
#
# #         # Deal with parts of the arrays with no information
# #         for i in range(len(eff[energy_range])):
# #             if thrown_events[i] == 0:
# #                 eff[energy_range][i] = 0
# #                 sig[energy_range][i] = np.inf
#
# #     # Combine low, mid, and high energy datasets
# #     eff_tot = (np.sum([eff[energy_range]/sig[energy_range] for energy_range in energy_ranges], axis=0) /
# #             np.sum([1/sig[energy_range] for energy_range in energy_ranges], axis=0))
# #     sig_tot = np.sqrt(1 / np.sum([1/sig[energy_range]**2 for energy_range in energy_ranges], axis=0))
# #     with np.errstate(divide='ignore'):
# #         relerr  = sig_tot / eff_tot
#
# #     # # UGH : find better way to do this
# #     # if reco:
# #     #     eff_tot = eff_tot[20:]
# #     #     sig_tot = sig_tot[20:]
# #     #     relerr  = relerr[20:]
#
# #     # thrown_radius = np.array([energy_range_to_radii['low'][i]
# #     #     for i in rgrp])
# #     # return thrown_radius, sig_tot, relerr
# #     return eff_tot, sig_tot, relerr
#
# """ Get the effective area for a given composition and cut """
# def getEff(s, cut, comp='joint', reco=True):
#
#     eff, sig, relerr = {},{},{}
#     log_energy = np.log10(s['MC_energy'])
#     Ebins = getEbins()
#     Emids = getMids(Ebins)
#     # print('Emids = {}'.format(Emids))
#     erangeDict = getErange()
#
#     c0 = cut
#     if comp != 'joint':
#         compcut = s['comp'] == comp
#         c0 = cut * compcut
#
#     # Set radii for finding effective area
#     rDict = {}
#     keys = ['low', 'mid', 'high']
#     for key in keys:
#         rDict[key] = np.array([600, 800, 1100, 1700, 2600, 2900])
#     rDict['low'][1] = 600
#     print('rDict = {}'.format(rDict))
#     Ebreaks = np.array([4, 5, 6, 7, 8, 9])
#     rgrp = np.digitize(Emids, Ebreaks) - 1
#     print('rgrp = {}'.format(rgrp))
#
#
#     for key in keys:
#
#         # Get efficiency and sigma
#         simcut = np.array([sim in erangeDict[key] for sim in s['sim']])
#         observed_events = np.histogram(log_energy[c0*simcut], bins=Ebins)[0]
#         thrown_events = s['MC'][comp][key].astype('float')
#         eff[key], sig[key], relerr[key] = np.zeros((3, len(observed_events)))
#         with np.errstate(divide='ignore', invalid='ignore'):
#             eff[key] = observed_events / thrown_events
#             var = (observed_events+1)*(observed_events+2)/((thrown_events+2)*(thrown_events+3)) - (observed_events+1)**2/((thrown_events+2)**2)
#         sig[key] = np.sqrt(var)
#
#         # Multiply by throw area
#         thrown_radius = np.array([rDict[key][i] for i in rgrp])
#         print('thrown_radius = {}'.format(thrown_radius))
#         eff[key] *= np.pi*(thrown_radius**2)
#         sig[key] *= np.pi*(thrown_radius**2)
#
#         # Deal with parts of the arrays with no information
#         for i in range(len(eff[key])):
#             if thrown_events[i] == 0:
#                 eff[key][i] = 0
#                 sig[key][i] = np.inf
#
#     # Combine low, mid, and high energy datasets
#     eff_tot = (np.sum([eff[key]/sig[key] for key in keys], axis=0) /
#             np.sum([1/sig[key] for key in keys], axis=0))
#     sig_tot = np.sqrt(1 / np.sum([1/sig[key]**2 for key in keys], axis=0))
#     with np.errstate(divide='ignore'):
#         relerr  = sig_tot / eff_tot
#
#     # UGH : find better way to do this
#     if reco:
#         eff_tot = eff_tot[20:]
#         sig_tot = sig_tot[20:]
#         relerr  = relerr[20:]
#
#     return eff_tot, sig_tot, relerr
#
#
# """ Get effective area information fast for preset cuts """
# def getEff_fast(linear=False):
#     mypaths = paths.Paths()
#     eFile = '{}/EffectiveArea.npy'.format(mypaths.llh_dir + '/resources')
#     temp = np.load(eFile)
#     temp = temp.item()
#     effarea, sigma, relerr = temp['effarea'], temp['sigma'], temp['relerr']
#     if linear:
#         effarea = np.ones(len(effarea), dtype=float)
#     return effarea, sigma, relerr
#
#
# # """ Flat line fit for effective area at high energies  """
# # def line_fit(s, cut, comp='joint', st=45):
#
# #     lineFit = lambda x, b: b
# #     Emids = getMids(getEbins())
# #     eff, sigma, relerr = getEff(s, cut, comp=comp)
# #     Emids, eff, sigma, relerr = Emids[st:], eff[st:], sigma[st:], relerr[st:]
# #     x = Emids.astype('float64')
# #     y = eff
# #     popt, pcov = optimize.curve_fit(lineFit, x, y, sigma=sigma)
# #     yfit = lineFit(x, *popt)
# #     yfit = np.array([yfit for i in range(len(x))])
# #     chi2 = np.sum(((yfit - y)/sigma) **2) / len(x)
# #     print 'Chi2:', chi2
# #     return yfit, relerr
#
#
#
#
#
# """ Look at chi2 for flat line fit for a variety of cuts """
# def chi2(s, cut):
#
#     lineFit = lambda x, b: b
#     st = 44
#     Emids = getEmids()
#
#     chi2List = []
#     QTable = arange(1, 10, 0.25)
#     for i in range(len(QTable)):
#         qcut = cut * (s['Q1']>QTable[i])
#         eff, sigma, relerr = getEff(s, qcut)
#
#         x = Emids.astype('float64')[st:]
#         y = eff[st:]
#
#         popt, pcov = optimize.curve_fit(lineFit, x, y, sigma=sigma)
#         yfit = lineFit(x, *popt)
#         yfit = [yfit for i in range(len(x))]
#         chi2 = np.sum(((yfit - y)/sigma) ** 2) / len(x)
#         chi2List.append(chi2)
#
#     plt.title('Chi2 vs Charge')
#     plt.xlabel('Maxcharge > (VEM)')
#     plt.ylabel('Chi2')
#     plt.plot(QTable, chi2List, '.')
#     plt.show()
#
#
# """ Show the effective area """
# def plotter(s, cut, comp='joint', emin=6.2, ndiv=False, out=False, fit=False):
#
#     fig, ax = plt.subplots()
#     #ax.set_title('Effective Area vs Energy')
#     ax.set_xlabel(r'$\log_{10}(E/\mathrm{GeV})$')
#     ax.set_ylabel(r'Effective Area ($m^2$)')
#     Emids = getMids(getEbins(reco=True))
#
#     eff, sigma, relerr = getEff(s, cut, comp=comp)
#     #eff2, relerr2 = line_fit(s, cut, comp=comp, st=st)
#
#     lineFit = lambda x, b: b
#     x = Emids.astype('float64')
#     y = eff
#     c0 = x >= emin
#     popt, pcov = optimize.curve_fit(lineFit, x[c0], y[c0], sigma=sigma[c0])
#     yfit = lineFit(x[c0], *popt)
#     yfit = np.array([yfit for i in range(len(x[c0]))])
#     chi2 = np.sum(((yfit - y[c0])/sigma[c0]) **2) / len(x[c0])
#     print 'Chi2:', chi2
#     eff2 = yfit
#
#     # Give the option for combined bins
#     if ndiv:
#         eff_joint, en_joint = [], []
#         for j in range(len(eff)/ndiv):
#             start = ndiv*j
#             end = ndiv*(j+1)
#             eff_joint.append(mean(eff[start:end]))
#             en_joint.append(mean(Emids[start:end]))
#         ax.plot(en_joint, eff_joint, 'o', label=comp)
#     else:
#         ax.errorbar(Emids, eff, yerr=sigma, fmt='.', label=comp)
#         if fit:
#             ax.plot(Emids[c0], eff2)
#
#     #ax.legend(loc='upper left')
#     #ax.set_yscale('log')
#     if out:
#         plt.savefig(out)
#     plt.show()
#
#
#
# if __name__ == "__main__":
#
#     # Setup global paths
#     mypaths = paths.Paths()
#
#     p = argparse.ArgumentParser()
#     p.add_argument('-c', '--config', dest='config',
#             default='IT73',
#             help='Detector configuration [IT73|IT81]')
#     p.add_argument('-o', '--outdir', dest='outdir',
#                 default='/home/jbourbeau/public_html/figures/ShowerLLH/qualitycuts', help='Output directory')
#     p.add_argument('-b', '--bintype', dest='bintype',
#                 default='logdist',
#                 choices=['standard', 'nozenith', 'logdist'],
#                 help='Option for a variety of preset bin values')
#     args = p.parse_args()
#
#     sim_reco = load_sim(config=args.config, bintype=args.bintype)
#     standard_mask = sim_reco['cuts']['llh']
#
#     eff_area, eff_area_error, energy_midpoints = get_effective_area(sim_reco, standard_mask, args.bintype)
#     # Plot effective area
#     fig, ax = plt.subplots()
#     ax.errorbar(energy_midpoints, eff_area, yerr=eff_area_error, marker='.')
#     intercept = line_fit(eff_area, eff_area_error, energy_midpoints)
#     print('Maximum effective area = {}'.format(intercept))
#     high_energy_mask = (energy_midpoints >= 10**6.2)
#     high_energies = energy_midpoints[high_energy_mask]
#     ax.plot([high_energies[0], high_energies[-1]],
#             [intercept, intercept], marker='None',
#             linestyle='-', color='k')
#     ax.ticklabel_format(axis='y', style='sci', scilimits=(-2, 2))
#     ax.grid()
#     ax.set_xscale('log')
#     ax.set_ylabel('$\mathrm{Effective \ Area} \ [\mathrm{m^2}]$')
#     ax.set_xlabel('$\mathrm{E_{MC}}/\mathrm{GeV}$')
#     ax.set_title(r'ShowerLLH - IT73 - {} LLH bins'.format(args.bintype))
#     outfile = args.outdir + '/' + \
#         'effarea_vs_energy_{}.png'.format(args.bintype)
#     plt.savefig(outfile)
#     plt.close()
#
#     # d['effarea'], d['sigma'], d['relerr'] = getEff(s, s['cuts']['llh'])
#     # outFile = '%s/%s_EffectiveArea.npy' % (mypaths.llh_dir + '/resources', args.config)
#     # np.save(outFile, d)
#
#     # Effective area as we add different cuts
#     sim_reco = load_sim(config=args.config, bintype=args.bintype)
#     standard_mask = sim_reco['cuts']['llh']
#     seperate_cuts = [[True]*len(sim_reco['sim']), sim_reco['cuts']['llh1'],
#             sim_reco['cuts']['llh2'], sim_reco['cuts']['llh3'],
#             sim_reco['cuts']['llh4'], sim_reco['cuts']['llh5']]
#     cut_labels = ['Before cuts', 'Reconstruction passed', 'Zenith', 'Containment',
#             'Loudest not on edge', 'Max charge']
#     cut_mask = np.array([True]*len(sim_reco['MC_energy']))
#     fig, axarr = plt.subplots(2, 3, sharex=True, sharey=False)
#     for llh_mask, label, ax in zip(seperate_cuts, cut_labels, axarr.flatten()):
#         cut_mask *= llh_mask
#         eff_area, eff_area_error, energy_midpoints = \
#                 get_effective_area(sim_reco,
#                                 cut_mask,
#                                 args.bintype)
#         ax.errorbar(energy_midpoints, eff_area,
#                 yerr=eff_area_error, marker='.',
#                 alpha=0.75)
#         ax.ticklabel_format(axis='y', style='sci', scilimits=(-2, 2))
#         ax.set_title(label)
#         ax.grid()
#         ax.set_xscale('log')
#         if label == 'Max charge':
#             intercept = line_fit(eff_area, eff_area_error, energy_midpoints)
#             print('Maximum effective area = {}'.format(intercept))
#             high_energy_mask = (energy_midpoints >= 10**6.2)
#             high_energies = energy_midpoints[high_energy_mask]
#             ax.plot([high_energies[0], high_energies[-1]],
#                     [intercept, intercept], marker='None',
#                     linestyle='-', color='k')
#         # ax.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
#         #             ncol=3, mode="expand", borderaxespad=0.)
#     # plt.legend(bbox_to_anchor=(1.05, 1), loc=2,
#     #         borderaxespad=0.)
#     # fig.text(0.5, 1.00, 'Cumulative cuts', ha='center')
#     fig.text(0.5, 0.00, '$\mathrm{E_{MC}}/\mathrm{GeV}$', ha='center')
#     fig.text(-0.01, 0.5, '$\mathrm{Effective \ Area} \ [\mathrm{m^2}]$', va='center', rotation='vertical')
#     outfile = 'effective-area_vs_MC-energy_cumulative-cuts'
#     # outfile = 'effective-area_vs_MC-energy_cumulative-cuts_CleanedHLCTankPulses'
#     outfile = args.outdir + '/' + outfile + '.png'
#     plt.tight_layout()
#     plt.savefig(outfile)
