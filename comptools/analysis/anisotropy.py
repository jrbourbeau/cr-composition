
from __future__ import division
import os
import glob
import numpy as np
import pandas as pd
import healpy as hp
import matplotlib.pyplot as plt
import seaborn.apionly as sns
from matplotlib.ticker import FormatStrFormatter
from matplotlib.colors import ListedColormap

from icecube import astro

from ..base import get_paths


def equatorial_to_healpy_theta_phi(ra, dec):

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
    hp_theta, hp_phi = equatorial_to_healpy_theta_phi(ra, dec)
    pix = hp.ang2pix(n_side, hp_theta, hp_phi)
    data_skymap[pix] += 1

    # Reference skymap
    rand_times = np.random.choice(times, size=n_resamples)
    ra, dec = astro.dir_to_equa(local_zenith, local_azimuth, rand_times)
    hp_theta, hp_phi = equatorial_to_healpy_theta_phi(ra, dec)
    ref_skymap = append_time_scrambled_events(
                                hp_theta, hp_phi, ref_skymap,
                                n_side=n_side, n_resamples=20)

    return


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
        hp_theta, hp_phi = equatorial_to_healpy_theta_phi(ra, dec)
        pix = hp.ang2pix(n_side, hp_theta, hp_phi)
        data_skymap[pix] += 1

        # Reference skymap
        rand_times = np.random.choice(times, size=n_resamples)
        ra, dec = astro.dir_to_equa(local_zenith, local_azimuth, rand_times)
        hp_theta, hp_phi = equatorial_to_healpy_theta_phi(ra, dec)
        ref_skymap = add_to_skymap(hp_theta, hp_phi, ref_skymap,
                                   n_side=n_side, weights=1/n_resamples)

    return data_skymap, ref_skymap, local_skymap


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


def get_rel_int_map(data_map, ref_map):

    with np.errstate(invalid='ignore', divide='ignore'):
        rel_int = (data_map - ref_map) / ref_map

    return rel_int


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
    data_map, ref_map, local_map = np.sum([hp.read_map(f, range(3), verbose=False) for f in files], axis=0)

    # If specified, smooth data and reference maps
    if smooth:
        data_map = smooth_map(data_map, smooth_rad_deg=smooth)
        ref_map = smooth_map(ref_map, smooth_rad_deg=smooth)

    if name == 'data':
        skymap = data_map
    elif name == 'ref':
        skymap = ref_map
    elif name == 'relint':
        skymap = get_rel_int_map(data_map, ref_map)
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
                cbar_max=None, cbar_title='Skymap', llabel=None, polar=False):

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
                    min=cbar_min, max=cbar_max, cbar=False, cmap=cmap)
    else:
        shrink = 1.0
        hp.mollview(skymap, rot=180, coord='C', title='', min=cbar_min,
                    max=cbar_max, cbar=False, cmap=cmap)
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
