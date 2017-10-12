
from __future__ import division
from collections import defaultdict
import numpy as np
import pandas as pd
from scipy import optimize
from I3Tray import NaN, Inf
from icecube import icetray, dataio, dataclasses, toprec, phys_services, recclasses
from icecube.icetop_Level3_scripts import icetop_globals
from icecube.icetop_Level3_scripts.functions import count_stations

from .analysis import fit_DLP_params


def add_opening_angle(frame, particle1='MCPrimary', particle2='Laputop', key='angle'):
    angle = phys_services.I3Calculator.angle(frame[particle1], frame[particle2])
    frame[key] = dataclasses.I3Double(angle)

def add_IceTop_quality_cuts(frame):
        passed = all(frame['IT73AnalysisIceTopQualityCuts'].values())
        frame['passed_IceTopQualityCuts'] = icetray.I3Bool(passed)

def add_InIce_quality_cuts(frame):
    passed = all(frame['IT73AnalysisInIceQualityCuts'].values())
    frame['passed_InIceQualityCuts'] = icetray.I3Bool(passed)
    # Add individual InIce quality cuts to frame
    for key, value in frame['IT73AnalysisInIceQualityCuts']:
        frame['passed_{}'.format(key)] = icetray.I3Bool(value)

def add_nstations(frame, pulses='SRTCoincPulses'):
    nstation = count_stations(dataclasses.I3RecoPulseSeriesMap.from_frame(frame, pulses))
    frame['NStations'] = icetray.I3Int(nstation)

def lap_fitstatus_ok(frame):
    status_ok = frame['Laputop'].fit_status == dataclasses.I3Particle.OK
    frame['lap_fitstatus_ok'] = icetray.I3Bool(status_ok)

def add_fraction_containment(frame, track):
    scaling = phys_services.I3ScaleCalculator(frame['I3Geometry'])

    icetop_containment = scaling.scale_icetop(frame[track])
    frame['FractionContainment_{}_IceTop'.format(track)] = dataclasses.I3Double(icetop_containment)

    inice_containment = scaling.scale_inice(frame[track])
    frame['FractionContainment_{}_InIce'.format(track)] = dataclasses.I3Double(inice_containment)


class AddFractionContainment(icetray.I3ConditionalModule):

    def __init__(self, context):
        icetray.I3ConditionalModule.__init__(self, context)
        self.AddParameter('track', 'Track to be used in fraction containment', 'Laputop')
        self.AddOutBox('OutBox')

    def Configure(self):
        self.track = self.GetParameter('track')
        pass

    def Geometry(self, frame):
        # print('Working on Geometry frame')
        self.geometry = frame['I3Geometry']
        self.scaling = phys_services.I3ScaleCalculator(self.geometry)
        self.PushFrame(frame)

    def Physics(self, frame):
        # print('Working on Physics frame')
        # print('track = {}'.format(self.track))
        # print('keys = {}'.format(frame.keys()))
        icetop_containment = self.scaling.scale_icetop(frame[self.track])
        frame['FractionContainment_{}_IceTop'.format(self.track)] = dataclasses.I3Double(icetop_containment)

        inice_containment = self.scaling.scale_inice(frame[self.track])
        frame['FractionContainment_{}_InIce'.format(self.track)] = dataclasses.I3Double(inice_containment)

        self.PushFrame(frame)

    def Finish(self):
        return


def add_IceTop_tankXYcharge(frame, pulses):

    frame['I3RecoPulseSeriesMap_union'] = dataclasses.I3RecoPulseSeriesMapUnion(frame, pulses)
    pulse_map = dataclasses.I3RecoPulseSeriesMap.from_frame(frame, 'I3RecoPulseSeriesMap_union')

    geomap = frame['I3Geometry'].omgeo
    tanks_x, tanks_y, tanks_charge = [], [], []
    for omkey, pulses in pulse_map:
        x, y, z = geomap[omkey].position
        tanks_x.append(x)
        tanks_y.append(y)
        charge = sum([pulse.charge for pulse in pulses])
        tanks_charge.append(charge)

    if tanks_x and tanks_y and tanks_charge:
        frame['tanks_x'] = dataclasses.I3VectorDouble(tanks_x)
        frame['tanks_y'] = dataclasses.I3VectorDouble(tanks_y)
        frame['tanks_charge'] = dataclasses.I3VectorDouble(tanks_charge)

    del frame['I3RecoPulseSeriesMap_union']


class AddIceTopTankXYCharge(icetray.I3ConditionalModule):

    def __init__(self, context):
        icetray.I3ConditionalModule.__init__(self, context)
        self.AddParameter('pulses', 'Pulses to caluclate distances to from track', 'SRTCoincPulses')
        self.AddOutBox('OutBox')

    def Configure(self):
        self.pulses = self.GetParameter('pulses')
        pass

    def Geometry(self, frame):
        self.geometry = frame['I3Geometry']
        self.geomap = self.geometry.omgeo
        self.PushFrame(frame)

    def Physics(self, frame):
        frame['I3RecoPulseSeriesMap_union'] = dataclasses.I3RecoPulseSeriesMapUnion(frame, self.pulses)
        pulse_map = dataclasses.I3RecoPulseSeriesMap.from_frame(frame, 'I3RecoPulseSeriesMap_union')

        tanks_x, tanks_y, tanks_charge = [], [], []
        for omkey, pulses in pulse_map:
            x, y, z = self.geomap[omkey].position
            tanks_x.append(x)
            tanks_y.append(y)
            charge = sum([pulse.charge for pulse in pulses])
            tanks_charge.append(charge)

        if tanks_x and tanks_y and tanks_charge:
            frame['tanks_x'] = dataclasses.I3VectorDouble(tanks_x)
            frame['tanks_y'] = dataclasses.I3VectorDouble(tanks_y)
            frame['tanks_charge'] = dataclasses.I3VectorDouble(tanks_charge)

        del frame['I3RecoPulseSeriesMap_union']
        self.PushFrame(frame)

    def Finish(self):
        return


class AddInIceMuonRadius(icetray.I3ConditionalModule):

    def __init__(self, context):
        icetray.I3ConditionalModule.__init__(self, context)
        self.AddParameter('track', 'Track to calculate distances from', 'Laputop')
        self.AddParameter('pulses', 'Pulses to caluclate distances to from track', 'CoincMuonReco_LineFit')
        self.AddParameter('min_DOM', 'Minimum DOM number to be considered', 1)
        self.AddParameter('max_DOM', 'Maximum DOM number to be considered', 60)
        self.AddOutBox('OutBox')

    def Configure(self):
        self.track = self.GetParameter('track')
        self.pulses = self.GetParameter('pulses')
        self.min_DOM = self.GetParameter('min_DOM')
        self.max_DOM = self.GetParameter('max_DOM')
        self.get_dist = phys_services.I3Calculator.closest_approach_distance
        pass

    def Geometry(self, frame):
        self.geometry = frame['I3Geometry']
        self.geomap = self.geometry.omgeo
        self.PushFrame(frame)

    def Physics(self, frame):
        track = frame[self.track]
        pulse_map = dataclasses.I3RecoPulseSeriesMap.from_frame(frame, self.pulses)
        dists, charges = [], []
        for omkey, pulses in pulse_map:
            # Throw out Deep Core strings (want homogenized total charge)
            if (omkey.string < 79) and (omkey.om >= self.min_DOM) and (omkey.om <= self.max_DOM):
                # Get distance of clostest approach to DOM from track
                dist = self.get_dist(track, self.geomap[omkey].position)
                dists.append(dist)
                # Get charge recorded in DOM
                charge = np.sum([pulse.charge for pulse in pulses])
                charges.append(charge)

        # Ensure that both dists and charges have non-zero size
        if dists and charges:
            frame['inice_dom_dists_{}_{}'.format(self.min_DOM, self.max_DOM)] = dataclasses.I3VectorDouble(dists)
            frame['inice_dom_charges_{}_{}'.format(self.min_DOM, self.max_DOM)] = dataclasses.I3VectorDouble(charges)

            dists = np.asarray(dists)
            charges = np.asarray(charges)

            avg_dist = np.average(dists)
            median_dist = np.median(dists)
            std_dists = np.std(dists)
            one_std_mask = (dists > avg_dist + std_dists) | (dists < avg_dist - std_dists)
            half_std_mask = (dists > avg_dist + 2*std_dists) | (dists < avg_dist - 2*std_dists)

            frac_outside_one_std = dists[one_std_mask].shape[0]/dists.shape[0]
            frac_outside_two_std = dists[half_std_mask].shape[0]/dists.shape[0]

            # Add variables to frame
            frame['avg_inice_radius'] = dataclasses.I3Double(avg_dist)
            frame['median_inice_radius'] = dataclasses.I3Double(median_dist)
            frame['std_inice_radius'] = dataclasses.I3Double(std_dists)
            frame['frac_outside_one_std_inice_radius'] = dataclasses.I3Double(frac_outside_one_std)
            frame['frac_outside_two_std_inice_radius'] = dataclasses.I3Double(frac_outside_two_std)

            # frame['qweighted_inice_radius_{}_{}'.format(self.min_DOM, self.max_DOM)] = dataclasses.I3Double(np.average(dists, weights=charges))
            #
            # frame['invqweighted_inice_radius_{}_{}'.format(self.min_DOM, self.max_DOM)] = dataclasses.I3Double(np.average(dists, weights=1/charges))

        self.PushFrame(frame)

    def Finish(self):
        return


class AddIceTopNNCharges(icetray.I3ConditionalModule):

    def __init__(self, context):
        icetray.I3ConditionalModule.__init__(self, context)
        self.AddParameter('pulses',
                          'Pulses to caluclate distances to from track',
                          'SRTCoincPulses')
        self.AddOutBox('OutBox')

    def Configure(self):
        self.pulses = self.GetParameter('pulses')
        pass

    def Geometry(self, frame):
        self.geometry = frame['I3Geometry']
        self.geomap = self.geometry.omgeo
        self.PushFrame(frame)

    def Physics(self, frame):
        union_key = 'I3RecoPulseSeriesMap_union'
        frame[union_key] = dataclasses.I3RecoPulseSeriesMapUnion(frame,
                                                                 self.pulses)
        pulse_map = dataclasses.I3RecoPulseSeriesMap.from_frame(frame,
                                                                union_key)
        # tanks_x, tanks_y, tanks_charge = [], [], []
        tank_charges = defaultdict(list)
        for omkey, omgeo in self.geomap:
            # Only interested in saving IceTop OM charges
            if omgeo.omtype.name != 'IceTop':
                continue
            # x, y, z = omgeo.position
            # tanks_x.append(x)
            # tanks_y.append(y)
            try:
                pulses = pulse_map[omkey]
                charge = sum([pulse.charge for pulse in pulses])
            except KeyError:
                charge = 0
            tank_charges[omkey].append(charge)

        # if tanks_x and tanks_y and tanks_charge:
        #     frame['tanks_x'] = dataclasses.I3VectorDouble(tanks_x)
        #     frame['tanks_y'] = dataclasses.I3VectorDouble(tanks_y)
        #     frame['tanks_charge'] = dataclasses.I3VectorDouble(tanks_charge)
        # self.tank_charges.append(pd.DataFrame(tank_charges))

        del frame[union_key]
        frame['NNcharges'] = dataclasses.I3MapKeyVectorDouble(tank_charges)

        self.PushFrame(frame)

    def Finish(self):
        # df_charges = pd.DataFrame(self.tank_charges)
        # columns = {c:'{}_{}_{}'.format(c.string, c.om, c.pmt) for c in df_charges.columns}
        # df_charges.rename(index=str, columns=columns, inplace=True)
        # with pd.HDFStore('test_charges_1.hdf') as output_store:
        #     output_store['dataframe'] = df_charges

        return


class AddIceTopChargeDistance(icetray.I3ConditionalModule):

    def __init__(self, context):
        icetray.I3ConditionalModule.__init__(self, context)
        self.AddParameter('track', 'Track to calculate distances from', 'Laputop')
        self.AddParameter('pulses', 'Pulses to caluclate distances to from track', 'SRTCoincPulses')
        self.AddOutBox('OutBox')

    def Configure(self):
        self.track = self.GetParameter('track')
        self.pulses = self.GetParameter('pulses')
        self.get_dist = phys_services.I3Calculator.closest_approach_distance
        pass

    def Geometry(self, frame):
        self.geometry = frame['I3Geometry']
        self.geomap = self.geometry.omgeo
        self.PushFrame(frame)


    def Physics(self, frame):
        track = frame[self.track]
        frame['I3RecoPulseSeriesMap_union'] = dataclasses.I3RecoPulseSeriesMapUnion(frame, self.pulses)
        pulse_map = dataclasses.I3RecoPulseSeriesMap.from_frame(frame, 'I3RecoPulseSeriesMap_union')

        tanks_dist, tanks_charge = [], []
        for omkey, pulses in pulse_map:
            # Get distance of clostest approach to DOM from track
            dist = self.get_dist(track, self.geomap[omkey].position)
            tanks_dist.append(dist)
            # Get charge recorded in DOM
            charge = sum([pulse.charge for pulse in pulses])
            tanks_charge.append(charge)
            # pair = dataclasses.make_pair(dist, charge)
            # tanks_charge_dist_pair.append(pair)

        # frame['tank_charge_dist_{}'.format(self.track)] = dataclasses.I3VectorDoubleDouble(tanks_charge_dist_pair)
        if tanks_dist and tanks_charge:
            frame.Put('tanks_charge_{}'.format(self.track), dataclasses.I3VectorDouble(tanks_charge))
            frame.Put('tanks_dist_{}'.format(self.track), dataclasses.I3VectorDouble(tanks_dist))
            # frame.Put('IceTop_charge', dataclasses.I3Double( np.sum(charges) ))

            # Convert to ndarrays for easy array manipulation
            tanks_dist = np.asarray(tanks_dist)
            tanks_charge = np.asarray(tanks_charge)
            distance_mask = tanks_dist > 175
            # charge_175m = np.sum(charges[distance_mask])
            # frame.Put('IceTop_charge_175m', dataclasses.I3Double(charge_175m))

            try:
                lap_params = frame['LaputopParams']
                lap_log_s125 = lap_params.value(recclasses.LaputopParameter.Log10_S125)
                lap_beta = lap_params.value(recclasses.LaputopParameter.Beta)
                tank_dist_mask = tanks_dist > 11
                # beta, log_s125 = fit_DLP_params(tanks_charge[distance_mask],
                #     tanks_dist[distance_mask], lap_log_s125, lap_beta)
                log_s125, beta = fit_DLP_params(tanks_charge[distance_mask],
                    tanks_dist[distance_mask], lap_log_s125, lap_beta)
                # print('lap_beta, refit_beta = {}, {}'.format(lap_beta, beta))
                # print('lap_log_s125, refit_log_s125 = {}, {}'.format(lap_log_s125, log_s125))
                # print('='*20)
            except Exception, e:
                print('Refitting shower to DLP didn\'t work out. '
                      'Setting to NaN...')
                print(e)
                log_s125, beta = NaN, NaN
                pass
            frame.Put('refit_beta', dataclasses.I3Double(beta))
            frame.Put('refit_log_s125', dataclasses.I3Double(log_s125))
            # print('='*20)

        del frame['I3RecoPulseSeriesMap_union']
        self.PushFrame(frame)

    def Finish(self):
        return


class AddInIceCharge(icetray.I3ConditionalModule):

    def __init__(self, context):
        icetray.I3ConditionalModule.__init__(self, context)
        self.AddParameter('pulses',
                          'I3RecoPulseSeriesMapMask to use for total charge',
                          'SRTCoincPulses')
        self.AddParameter('min_DOM', 'Minimum DOM number to be considered', 1)
        self.AddParameter('max_DOM', 'Maximum DOM number to be considered', 60)
        self.AddOutBox('OutBox')

    def Configure(self):
        self.pulses = self.GetParameter('pulses')
        self.min_DOM = self.GetParameter('min_DOM')
        self.max_DOM = self.GetParameter('max_DOM')
        pass

    def Physics(self, frame):
        q_tot = NaN
        n_channels = 0
        n_hits = 0
        max_qfrac = NaN
        pulse_map = dataclasses.I3RecoPulseSeriesMap.from_frame(frame, self.pulses)
        charge_list = []
        for omkey, pulses in pulse_map:
            # Throw out Deep Core strings (want homogenized total charge)
            if (omkey.string < 79) and (omkey.om >= self.min_DOM) and (omkey.om <= self.max_DOM):
                n_channels += 1
                for pulse in pulses:
                    charge_list += [pulse.charge]
        # Check to see if DOMs with signal are in the min_DOM to
        # max_DOM range
        if n_channels == 0:
            q_tot = NaN
            max_qfrac = NaN
        else:
            q_tot = np.sum(charge_list)
            n_hits = len(charge_list)
            max_qfrac = np.max(charge_list)/q_tot

        frame.Put('InIce_charge_{}_{}'.format(self.min_DOM, self.max_DOM),
                  dataclasses.I3Double(q_tot))
        frame.Put('NChannels_{}_{}'.format(self.min_DOM, self.max_DOM),
                  icetray.I3Int(n_channels))
        frame.Put('NHits_{}_{}'.format(self.min_DOM, self.max_DOM),
                  icetray.I3Int(n_hits))
        frame.Put('max_qfrac_{}_{}'.format(self.min_DOM, self.max_DOM),
                  dataclasses.I3Double(max_qfrac))
        self.PushFrame(frame)

    def Finish(self):
        return
