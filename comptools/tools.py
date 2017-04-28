from icecube import icetray, dataclasses

def pulse_is_saturated(gcd, om_key, reco_pulse):
    """
    Check if a pulse is saturated.

    The threshold is automatically set depending on the gain type and the DOM calibration.
    Low gain saturation is nominally set to 90000 PEs while low gain saturation is set to
    the hglgCrossOver value in the calibration (it defaults to 3000 PEs if this fails).

    Parameters:
     - gcd: an object with an dataclasses.I3Calibration data member called calibration.
     - om_key: and object of type icetray.OMKey
     - reco_pulse: object of type dataclasses.I3RecoPulse

    Returns: Bool
    """
    from icecube.dataclasses import I3DOMStatus

    # nominal thresholds (magic numbers)
    SAT_LG = 90000.
    SAT_HG = 3000.

    gain = gcd.detector_status.dom_status[om_key].dom_gain_type
    vemCalib = gcd.calibration.vem_cal[om_key]
    pe_per_vem = vemCalib.pe_per_vem/vemCalib.corr_factor

    if gain == I3DOMStatus.High:
        threshold = vemCalib.hglg_cross_over/pe_per_vem
        if threshold <= 0:
            threshold = SAT_HG/pe_per_vem
    elif gain == I3DOMStatus.Low:
        threshold = SAT_LG/pe_per_vem
    else:
        raise Exception('Unknown gain type: %s'%gain)

    return reco_pulse.GetCharge() > threshold


def tank_geometry(geometry, om_key):
    """
    Returns the IceTop tank geometry object corresponding to a given DOM.
    """
    if om_key.om == 61 or om_key.om == 62:
        return geometry.stationgeo[om_key.string][0]
    elif om_key.om == 63 or om_key.om == 64:
        return geometry.stationgeo[om_key.string][1]
    return None


def to_shower_cs(fit):
    """
    Rotate to shower CS takes a fit (assumes fit.dir is set) and returns a rotation matrix.
    Requires numpy.
    """
    import numpy
    from math import cos, sin
    # counter-clockwise (pi + phi) rotation
    d_phi = numpy.matrix([ [ -cos(fit.dir.phi), -sin(fit.dir.phi), 0],
                           [  sin(fit.dir.phi), -cos(fit.dir.phi), 0],
                           [  0,                 0,                1] ])
    # clock-wise (pi - theta) rotation
    d_theta = numpy.matrix([ [  -cos(fit.dir.theta), 0, -sin(fit.dir.theta)],
                             [  0,                  1,  0,                ],
                             [  sin(fit.dir.theta), 0,  -cos(fit.dir.theta)] ])
    return d_theta*d_phi


def classify_from_seed(pulses, reco, geometry, min_time=-200, max_time=800):
    """
    Classify pulses according to their agreement in time with a shower axis reconstruction.
    The classification is done using a constant cut on the time difference of the pulse time
    and the arrival time of the plane front at the position of the DOM.
    Requires numpy.

    Arguments:
     - pulses (a list of I3RecoPulse)
     - reco (an I3Particle with position and direction set)
     - geometry (an I3Geometry instance)
     - min_time (float, optional)
     - max-time (float, optional)

    Returns:
      A dictionary with keys ('ok', 'rejected', 'after-pulse').
      The values of the dictionary are lists of (OMKey, launch number) pairs.
      The lists are disjoint.
    """
    import numpy
    stations = []
    keys = []
    for k in pulses.keys():
        for launch, p in enumerate(pulses[k]):
            keys.append((k, launch))
            stations.append((0.,
                             launch,
                             tank_geometry(geometry, k).position.x,
                             tank_geometry(geometry, k).position.y,
                             tank_geometry(geometry, k).position.z,
                             p.charge, p.time, p.width, p.flags))

    stations = numpy.array(stations)

    if len(stations) == 0:
        return {'ok': [], 'after-pulses': [], 'rejected': []}

    # Rotate position to shower coordinates according to reco
    M = to_shower_cs(reco)
    #print "stations:", stations
    stations[:,2:5] -= numpy.array([reco.pos.x, reco.pos.y, reco.pos.z])
    stations[:,2:5] = numpy.array(M*stations[:,2:5].transpose()).transpose()

    # subtract travelling time of plane front from station time
    stations[:,6] -= reco.time - stations[:,4]/0.29979

    ap = [(keys[i][0], keys[i][1]) for i,p in enumerate(stations.tolist()) if p[6] > 6500 and p[1] > 0.0]
    ok = [(keys[i][0], keys[i][1]) for i,p in enumerate(stations.tolist()) if not (keys[i][0], keys[i][1]) in ap and p[6] > min_time and p[6] < max_time]
    rej = [(keys[i][0], keys[i][1]) for i,p in enumerate(stations.tolist()) if not (keys[i][0], keys[i][1]) in ap and not (keys[i][0], keys[i][1]) in ok]

    return { 'ok':ok, 'after-pulses':ap, 'rejected':rej }


def get_barycenter(geometry, pulses):
    """
    This function returns the barycenter (I3Position) list of pulses.
    """
    t = 0
    x = 0
    y = 0
    z = 0
    w = 0
    for k in pulses.keys():
        for launch, p in enumerate(pulses[k]):
            x += tank_geometry(geometry, k).position.x
            y += tank_geometry(geometry, k).position.y
            z += tank_geometry(geometry, k).position.z
            t += p.time
            w += p.charge
    if w == 0:
        return None
    return dataclasses.I3Position(x/w, y/w, z/w), t/w

def get_direction(geometry, pulses, barycenter, barytime, front_time):
    import numpy
    from icecube.dataclasses import I3Constants
    from icecube.dataclasses import I3Direction

    sumw = 0
    sumx = 0
    sumy = 0
    sumx2 = 0
    sumy2 = 0
    sumxy = 0
    sumt = 0
    sumxt = 0
    sumyt = 0
    for k in pulses.keys():
        for launch, p in enumerate(pulses[k]):
            dx = tank_geometry(geometry, k).position.x - barycenter.x
            dy = tank_geometry(geometry, k).position.y - barycenter.y
            dt = p.time - barytime

            sumx += dx
            sumy += dy
            sumx2 += dx*dx
            sumy2 += dy*dy
            sumxy += dx*dy
            sumt += dt
            sumxt += dt*dx
            sumyt += dt*dy

    rxx = sumw * sumy2 - sumy*sumy;
    ryy = sumw * sumx2 - sumx*sumx;
    rxy = sumx * sumy - sumw * sumxy;
    rx = sumxy * sumy - sumx * sumy2;
    ry = sumxy * sumx - sumy * sumx2;
    rr = sumx2 * sumy2 - sumxy*sumxy;
    deter = sumx2 * rxx + sumxy * rxy + sumx * rx;

    if deter == 0 or deter is numpy.nan:
        return None

    u = (rxx * sumxt + rxy * sumyt + rx * sumt) / deter
    u *= -I3Constants.c
    v = (rxy * sumxt + ryy * sumyt + ry * sumt) / deter
    v *= -I3Constants.c
    t0 = (rx * sumxt + ry * sumyt + rr * sumt) / deter

    return I3Direction(u, v, numpy.sqrt(u*u+v*v))

def combinations2_(n, min_entries=0):
    # this was wrong, but I'm keeping it for now.
    import numpy
    selected = numpy.array([True for d in range(n)])
    if sum(selected) >= min_entries:
        yield selected
    for n in range(0,len(selected)):
        selected[:] = True
        selected[-n-1:] = False
        for c in combinations_(n):
            if n:
                selected[-n:] = c
            if sum(selected) >= min_entries:
                yield selected

def combinations_m_(n, m):
    import numpy
    selected = numpy.array([True for d in range(n)])
    if m == 0:
        yield selected
    else:
        for n in range(m, len(selected)+1):
            selected[:] = True
            selected[-n] = False
            for c in combinations_m_(n-1, m-1):
                if len(c):
                    selected[-n+1:] = c
                yield selected

def combinations_(n, min_entries=0):
    import numpy
    for i in range(n):
        for selected in combinations_m_(n, i):
            if sum(selected) >= min_entries:
                yield selected

def combinations(data, min_entries=0):
    import numpy
    data = numpy.array(data)
    for c in combinations_(len(data)):
        if sum(c) >= min_entries:
            yield data[c]
