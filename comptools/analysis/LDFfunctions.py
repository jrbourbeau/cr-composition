
from __future__ import division
import numpy as np
from scipy import optimize


def top_ldf_sigma(r, logq):
    a = [-.5519,-.078]
    b = [-.373, -.658, .158]
    trans = [.340, 2.077]

    if logq > trans[1]:
        logq = trans[1]
    if (logq < trans[0]):
        return 10**(a[0] + a[1] * logq)
    else:
        return 10**(b[0] + b[1] * logq + b[2]*logq*logq)


def DLP(dists, log_s125, beta):
    '''Double Logarithmic Parabola (DLP) function used to parameterize air shower events

    For a reference see IceCube internal report 200702001, 'A Lateral Distribution Function and Fluctuation Parametrisation for IceTop' by Stefan Klepser.

    Parameters
    ----------
    dists : float, array-like
        Tank distance(s), in meters, from the shower core.
    log_s125 : float
        Base-10 logarithm of the signal deposited 125m away from the shower core.
    beta : float
        Slope of the DLP function. Related to the air shower age.

    Returns
    -------
    float, numpy.array
        Returns the base-10 logarithm of the signal deposited at a distance, dists, away from the shower core.

    '''

    lS0 = log_s125
    lR = np.log10(dists)
    lR0 = np.log10(125.0)

    return -0.30264 * (lR-lR0)**2 - beta * (lR-lR0) + lS0


def fit_DLP_params(charges, distances, lap_log_s125, lap_beta):

    charges = np.asarray(charges)
    distances = np.asarray(distances)

    charge_sigmas = []
    for r, logq in zip(distances, np.log10(charges)):
        charge_sigmas.append(10**top_ldf_sigma(r, logq))
    charge_sigmas = np.asarray(charge_sigmas)

    popt, pcov = optimize.curve_fit(DLP, distances, np.log10(charges), sigma=charge_sigmas)
    # popt, pcov = optimize.curve_fit(LDF, distances, np.log10(charges),
    #     sigma=charge_sigmas, p0=lap_beta)
    # beta = popt[0]

    return popt
