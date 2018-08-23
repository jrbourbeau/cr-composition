
from __future__ import division
import numpy as np
import healpy as hp


def equatorial_to_healpy(ra, dec):

    hp_theta = np.pi/2 - dec
    hp_phi = ra

    return hp_theta, hp_phi


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
