from __future__ import division
import os
import glob
import numpy as np
import pandas as pd
import healpy as hp
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import FormatStrFormatter
from matplotlib.colors import ListedColormap
from scipy import optimize

from .base import get_paths, requires_icecube

try:
    from icecube import astro
except ImportError as e:
    pass


def equatorial_to_healpy(ra, dec):

    hp_theta = np.pi/2 - dec
    hp_phi = ra

    return hp_theta, hp_phi


def add_to_skymap(theta, phi, skymap, n_side=64, weights=1.0):

    pix_array = hp.ang2pix(n_side, theta, phi)
    if not isinstance(weights, (int, float, np.array, list, tuple)):
        raise ValueError('weights must be a number of array-like')
    if isinstance(weights, (int, float)):
        weights = [weights] * pix_array.shape[0]
    for pix, event_weight in zip(pix_array, weights):
        skymap[pix] += event_weight

    return skymap


@requires_icecube
def append_to_skymaps(row, times, data_map, ref_map, local_map,
                      n_side, n_resamples):

    local_zenith = row['lap_zenith']
    local_azimuth = row['lap_azimuth']
    local_time = row['start_time_mjd']

    # Local skymap
    pix = hp.ang2pix(n_side, local_zenith, local_azimuth)
    local_skymap[pix] += 1

    # Data skymap
    ra, dec = astro.dir_to_equa(local_zenith, local_azimuth, local_time)
    hp_theta, hp_phi = equatorial_to_healpy(ra, dec)
    pix = hp.ang2pix(n_side, hp_theta, hp_phi)
    data_skymap[pix] += 1

    # Reference skymap
    rand_times = np.random.choice(times, size=n_resamples)
    ra, dec = astro.dir_to_equa(local_zenith, local_azimuth, rand_times)
    hp_theta, hp_phi = equatorial_to_healpy(ra, dec)
    ref_skymap = append_time_scrambled_events(
                                hp_theta, hp_phi, ref_skymap,
                                n_side=n_side, n_resamples=20)

    return


@requires_icecube
def make_skymaps(df, times, n_resamples=20, n_side=64, verbose=False):

    n_pix = hp.nside2npix(n_side)
    data_skymap = np.zeros(n_pix, dtype=float)
    ref_skymap = np.zeros(n_pix, dtype=float)
    local_skymap = np.zeros(n_pix, dtype=float)

    for idx, row in df.iterrows():
        local_zenith = row['lap_zenith']
        local_azimuth = row['lap_azimuth']
        local_time = row['start_time_mjd']

        # Local skymap
        pix = hp.ang2pix(n_side, local_zenith, local_azimuth)
        local_skymap[pix] += 1

        # Data skymap
        ra, dec = astro.dir_to_equa(local_zenith, local_azimuth, local_time)
        hp_theta, hp_phi = equatorial_to_healpy(ra, dec)
        pix = hp.ang2pix(n_side, hp_theta, hp_phi)
        data_skymap[pix] += 1

        # Reference skymap
        rand_times = np.random.choice(times, size=n_resamples)
        ra, dec = astro.dir_to_equa(local_zenith, local_azimuth, rand_times)
        hp_theta, hp_phi = equatorial_to_healpy(ra, dec)
        ref_skymap = add_to_skymap(hp_theta, hp_phi, ref_skymap,
                                   n_side=n_side, weights=1/n_resamples)

    return data_skymap, ref_skymap, local_skymap


def cos_fit_func(x, *p):
    harmonic_terms = [ p[n] * np.cos(n * (p[n+1]-x)) for n in np.arange(1, len(p), 2, dtype=int)]
    return np.sum(harmonic_terms, axis=0) + p[0]
    # return sum([p[2*i+1] * np.cos((i+1) * (x-p[2*i+2])) for i in range(int(len(p)/2))]) + p[0]


def get_proj_fit_params(x, y, l=10, sigmay=None):

    # Guess at best fit parameters
    # amplitude = 1e-3
    amplitude = (3./np.sqrt(2)) * np.std(y)
    phase     = 0
    parm_init = [amplitude, phase]*l

    # Reduce amplitude as we go to higher l values
    for i in range(0, len(parm_init), 2):
        parm_init[i] *= 2.**(-i)
    parm_init = [0] + parm_init

    # Do best fit
    # popt, pcov = optimize.curve_fit(cos_fit_func, x, y, p0=[1e-3, 0]*l, sigma=sigmay)
    popt, pcov = optimize.curve_fit(cos_fit_func, x, y, p0=parm_init, sigma=sigmay)
    fitVals = cos_fit_func(x, *popt)
    ndof  = len(popt)
    if sigmay is not None:
        chi2 = (1. / (len(y)-ndof)) * sum((y - fitVals)**2 / sigmay**2)
    else:
        chi2 = (1. / (len(y)-ndof)) * sum((y - fitVals)**2)
    perr = np.sqrt(np.diag(pcov))

    return popt, perr, chi2


def get_proj_relint(relint, relint_err=None, decmin=-90, decmax=-40, ramin=0,
                    ramax=360, n_bins=24, units='deg'):

    if units not in ['deg', 'rad']:
        raise ValueError('units must be either "deg" or "rad"')

    n_pix = relint.shape[0]
    n_side = hp.npix2nside(n_pix)
    # Cut to desired dec range (equiv to healpy theta range)
    theta, phi = hp.pix2ang(n_side, range(n_pix))
    thetamax =  np.deg2rad(90 - decmin)
    thetamin = np.deg2rad(90 - decmax)
    dec_mask = (theta <= thetamax) & (theta >= thetamin)
    # Bin in right ascension
    ramin = np.deg2rad(ramin)
    ramax = np.deg2rad(ramax)
    rabins= np.linspace(ramin, ramax, n_bins+1, dtype=float)
    phi_bin_num = np.digitize(phi, rabins) - 1

    ri = np.zeros(n_bins)
    ri_err = np.zeros(n_bins)
    unseen_mask = relint == hp.UNSEEN
    for idx in range(n_bins):
        phi_bin_mask = (phi_bin_num == idx)
        combined_mask = phi_bin_mask & dec_mask & ~unseen_mask
        ri[idx] = np.mean(relint[combined_mask])
        if relint_err is not None:
            ri_err[idx] = np.sqrt(np.sum(relint_err[combined_mask]**2))/combined_mask.sum()

    ra = (rabins[1:] + rabins[:-1]) / 2
    ra_err = (rabins[1:] - rabins[:-1]) / 2
    if units == 'deg':
        ra = np.rad2deg(ra)
        ra_err = np.rad2deg(ra_err)

    if relint_err is None:
        return ri, ra, ra_err
    else:
        return ri, ri_err, ra, ra_err


def smooth_map(skymap, smooth_rad_deg=5.0):

    npix  = skymap.shape[0]
    nside = hp.npix2nside(npix)
    smooth_rad = np.deg2rad(smooth_rad_deg)
    smooth_map = np.zeros(npix, dtype=float)

    vec = np.transpose(hp.pix2vec(nside, np.arange(npix)))
    for i in range(npix):
        neighbors = hp.query_disc(nside, vec[i], smooth_rad)
        smooth_map[i] += skymap[neighbors].sum()

    return smooth_map


def mask_map(skymap, decmin=None, decmax=None):

    if decmin is None and decmax is None:
        raise ValueError('decmin and/or decmax must be specified')

    npix  = skymap.shape[0]
    nside = hp.npix2nside(npix)
    theta, phi = hp.pix2ang(nside, range(npix))

    theta_mask = np.ones(npix, dtype=bool)
    if decmin is not None:
        theta_min = np.deg2rad(90 - decmin)
        theta_mask *= (theta >= theta_min)
    if decmax is not None:
        theta_max = np.deg2rad(90 - decmax)
        theta_mask *= (theta <= theta_max)

    masked_map = np.copy(skymap)
    masked_map[theta_mask] = hp.UNSEEN

    return masked_map


# Li Ma Significance
def get_significance_map(data_map, ref_map, alpha=1/20., data_map_wsq=None,
                         ref_map_wsq=None):

    with np.errstate(invalid='ignore', divide='ignore'):
        # Allow for scaling term if weighted square maps necessary
        scale = 1.
        n_on  = data_map * scale
        n_off = ref_map/alpha * scale

        sign = np.sign(data_map - ref_map)
        sig_map = sign * np.sqrt(2*(n_on*np.log(((1+alpha)*n_on) / (alpha*(n_on+n_off)))
            + n_off * np.log(((1+alpha)*n_off) / (n_on+n_off))))

    return sig_map


def get_relint_map(data_map, ref_map):

    with np.errstate(invalid='ignore', divide='ignore'):
        relint = (data_map - ref_map) / ref_map

    return relint


def get_relint_err_map(data_map, ref_map, n_resamples=20):

    with np.errstate(invalid='ignore', divide='ignore'):
        relint_err = (data_map/ref_map) * np.sqrt(1/data_map + n_resamples/ref_map)

    return relint_err


def get_skymap_file(config, n_side, composition='all'):

    # Setup global path names
    mypaths = get_paths()

    skymap_file = os.path.join(mypaths.comp_data_dir,
                               config + '_data', 'anisotropy',
                               '{}_maps_nside_{}_{}.fits'.format(config,
                                                                 n_side,
                                                                 composition))

    return skymap_file


def get_map(name='relint', config='IC86.2012', composition='all', n_side=64,
            smooth=5.0, decmin=None, decmax=None, scale=None):

    if not isinstance(config, (str, tuple, list, np.array)):
        raise ValueError('config must be a string or array-like')
    if isinstance(config, str):
        config = [config]

    files = [get_skymap_file(c, n_side, composition) for c in config]
    data_map, ref_map, local_map = [], [], []
    for f in files:
        data, ref, local = hp.read_map(f, range(3), verbose=False)
        data_map.append(data)
        ref_map.append(ref)
        local_map.append(local)
    data_map = np.sum(data_map, axis=0)
    ref_map = np.sum(ref_map, axis=0)
    local_map = np.sum(local_map, axis=0)

    # If specified, smooth data and reference maps
    if smooth:
        data_map = smooth_map(data_map, smooth_rad_deg=smooth)
        ref_map = smooth_map(ref_map, smooth_rad_deg=smooth)

    if name == 'data':
        skymap = data_map
    elif name == 'ref':
        skymap = ref_map
    elif name == 'relint':
        skymap = get_relint_map(data_map, ref_map)
    elif name == 'relerr':
        skymap = get_relint_err_map(data_map, ref_map, n_resamples=20)
    elif name == 'sig':
        skymap = get_significance_map(data_map, ref_map)
    else:
        raise ValueError('Invalid name, {}, entered'.format(name))

    # Scale skymap
    if scale:
        skymap *= 10**scale

    # Mask skymap
    if decmin or decmax:
        skymap = mask_map(skymap, decmin=decmin, decmax=decmax)

    return skymap


def get_relint_diff(config='IC86.2012', n_side=64, smooth=5.0, decmin=None,
                    decmax=None, scale=None):

    rel_int_light = get_map(name='relint', composition='light', config=config,
                        n_side=n_side, smooth=smooth, decmin=decmin,
                        decmax=decmax, scale=scale)
    rel_int_heavy = get_map(name='relint', composition='heavy', config=config,
                        n_side=n_side, smooth=smooth, decmin=decmin,
                        decmax=decmax, scale=scale)

    rel_int_diff = mask_map(rel_int_light - rel_int_heavy, decmin=decmin, decmax=decmax)

    return rel_int_diff


def plot_skymap(skymap, smooth=None, decmax=None, scale=None, color_bins=40,
                color_palette='viridis', symmetric=False, cbar_min=None,
                cbar_max=None, cbar_title='Skymap', llabel=None, polar=False,
                fig=None, sub=None):

    cpalette = sns.color_palette(color_palette, color_bins)
    cmap = ListedColormap(cpalette.as_hex())
    cmap.set_under('white')
    cmap.set_bad('gray')

    # if cbar_max and cbar_max and symmetric and (cbar_max != -cbar_min):
    #     raise ValueError('The max/min colorbar values can\'t be symmetric')
    # elif cbar_max and cbar_max:
    #     pass
    # elif:
    #     skymap_min = skymap.min()
    #     skymap_max = skymap.max()
    #     maximum = np.max(np.abs([skymap_min, skymap_max]))
    #     cbar_min = np.sign(skymap_min) * maximum
    #     cbar_max = np.sign(skymap_max) * maximum
    # else:
    #     cbar_min = cbar_min if cbar_min else skymap.min()
    #     cbar_max = cbar_max if cbar_max else skymap.max()

    if polar:
        shrink = 0.6
        rot = [0,-90,180]
        hp.orthview(skymap, half_sky=True, rot=rot, coord='C', title='',
                    min=cbar_min, max=cbar_max, cbar=False, cmap=cmap,
                    fig=fig, sub=sub)
    else:
        shrink = 1.0
        hp.mollview(skymap, rot=180, coord='C', title='', min=cbar_min,
                    max=cbar_max, cbar=False, cmap=cmap, fig=fig, sub=sub)
    hp.graticule(verbose=False)

    fig = plt.gcf()
    ax = plt.gca()
    image = ax.get_images()[0]
    cbar = fig.colorbar(image, orientation='horizontal', aspect=50,
                        pad=0.01, fraction=0.1, ax=ax,
                        format=FormatStrFormatter('%g'),
                        shrink=shrink)
    if cbar_title:
        cbar.set_label(cbar_title, size=14)

    if not polar:
        ax.set_ylim(-1, 0.005)
        ax.annotate('0$^\circ$', xy=(1.8, -0.75), size=14)
        ax.annotate('360$^\circ$', xy=(-1.99, -0.75), size=14)

    if llabel:
        ax.annotate(llabel, xy=(-1.85,-0.24), size=20, color='white')

    return fig, ax
