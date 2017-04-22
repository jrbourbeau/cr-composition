#!/usr/bin/env python

import numpy as np
from scipy import optimize
from I3Tray import NaN, Inf
from icecube import icetray, dataio, dataclasses, toprec, phys_services, recclasses
from icecube.icetop_Level3_scripts import icetop_globals
from icecube.icetop_Level3_scripts.functions import count_stations


def add_MC_Laputop_angle(frame):
    angle_MC_Laputop = NaN
    if ('MCPrimary' in frame) and ('Laputop' in frame):
        angle_MC_Laputop = phys_services.I3Calculator.angle(frame['MCPrimary'],
                                                         frame['Laputop'])
    frame.Put('angle_MC_Laputop', dataclasses.I3Double(angle_MC_Laputop))


class AddIceTopChargeDistance(icetray.I3Module):

    def __init__(self, context):
        icetray.I3Module.__init__(self, context)
        self.AddParameter('track', 'Track to calculate distances from', 'Laputop')
        self.AddParameter('pulses', 'Pulses to caluclate distances to from track', 'SRTCoincPulses')
        self.AddOutBox('OutBox')

    def Configure(self):
        self.track = self.GetParameter('track')
        self.pulses = self.GetParameter('pulses')
        self.get_dist = phys_services.I3Calculator.closest_approach_distance
        self.charge_vs_dist_list = []
        pass

    def Geometry(self, frame):
        self.geometry = frame['I3Geometry']
        self.geomap = self.geometry.omgeo
        self.PushFrame(frame)

    def fit_beta(self, charges, distances, log_s125, lap_beta):

        charges = np.asarray(charges)
        distances = np.asarray(distances)
        # print('charges = {}'.format(charges))
        # print('distances = {}'.format(distances))
        # print('log_s125 = {}'.format(log_s125))
        # print('lap_beta = {}'.format(lap_beta))

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

        charge_sigmas = []
        for r, logq in zip(distances, np.log10(charges)):
            charge_sigmas.append(10**top_ldf_sigma(r, logq))
        charge_sigmas = np.asarray(charge_sigmas)

        def LDF(dists, beta):
            return 10**log_s125 * (dists/125)**(-beta-0.303*np.log10(dists/125))
            # return log_s125 - beta*np.log10(dists/125) - 0.303*np.log10(np.log10(dists/125))

        # popt, pcov = optimize.curve_fit(LDF, distances, np.log10(charges))
        popt, pcov = optimize.curve_fit(LDF, distances, np.log10(charges),
            sigma=charge_sigmas, p0=lap_beta)
        beta = popt[0]

        return beta


    def Physics(self, frame):
        try:
            track = frame[self.track]
            frame['I3RecoPulseSeriesMap_union'] = dataclasses.I3RecoPulseSeriesMapUnion(frame, self.pulses)
            pulse_map = dataclasses.I3RecoPulseSeriesMap.from_frame(frame, 'I3RecoPulseSeriesMap_union')
        except:
            # icetray.logging.log_info('Frame doesn\'t contain MCPrimary')
            self.PushFrame(frame)
            return

        dists = []
        charges = []
        for omkey, pulses in pulse_map:
            # Get distance of clostest approach to DOM from track
            dist = self.get_dist(track, self.geomap[omkey].position)
            dists.append(dist)
            # Get charge recorded in DOM
            charge = 0.0
            for pulse in pulses:
                charge += pulse.charge
            charges.append(charge)

        # # Create charge vs distance histogram
        # dist_bins = np.logspace(-2, 4, 50)
        # charge_bins = np.logspace(0, 3, 50)
        # # dist_bins = np.linspace(1e-2, 1e4, 128)
        # # charge_bins = np.linspace(0, 1e3, 128)
        # hist, _ , _ = np.histogram2d(dists, charges, bins=(charge_bins, dist_bins))
        # print(hist.flatten().sum())

        if dists and charges:
            frame.Put('tank_charges', dataclasses.I3VectorDouble(charges))
            frame.Put('tank_dists', dataclasses.I3VectorDouble(dists))
            frame.Put('IceTop_charge', dataclasses.I3Double( np.sum(charges) ))

            # Convert to ndarrays for easy array manipulation
            dists = np.asarray(dists)
            charges = np.asarray(charges)
            distance_mask = dists > 175
            charge_175m = np.sum(charges[distance_mask])
            frame.Put('IceTop_charge_175m', dataclasses.I3Double(charge_175m))

            try:
                lap_params = frame['LaputopParams']
                log_s125 = lap_params.value(recclasses.LaputopParameter.Log10_S125)
                lap_beta = lap_params.value(recclasses.LaputopParameter.Beta)
                beta = self.fit_beta(charges, dists, log_s125, lap_beta)
                print('lap_beta = {}'.format(lap_beta))
                print('refit_beta = {}'.format(beta))
                print('='*20)
                frame.Put('refit_beta', dataclasses.I3Double(beta))
            except:
                # print('Didn\'t work out')
                pass
            # print('='*20)

        self.PushFrame(frame)

    def Finish(self):
        return

class AddMuonRadius(icetray.I3Module):

    def __init__(self, context):
        icetray.I3Module.__init__(self, context)
        self.AddParameter('track', 'Track to calculate distances from', 'Laputop')
        self.AddParameter('pulses', 'Pulses to caluclate distances to from track', 'SRTCoincPulses')
        self.AddParameter('min_DOM', 'Minimum DOM number to be considered', 1)
        self.AddParameter('max_DOM', 'Maximum DOM number to be considered', 60)
        self.AddOutBox('OutBox')

    def Configure(self):
        self.track = self.GetParameter('track')
        self.pulses = self.GetParameter('pulses')
        self.min_DOM = self.GetParameter('min_DOM')
        self.max_DOM = self.GetParameter('max_DOM')
        self.get_dist = phys_services.I3Calculator.closest_approach_distance
        self.get_time = phys_services.I3Calculator.time_residual
        pass

    def Geometry(self, frame):
        self.geometry = frame['I3Geometry']
        self.geomap = self.geometry.omgeo
        self.PushFrame(frame)

    def Physics(self, frame):
        try:
            track = frame[self.track]
            pulses = frame[self.pulses]
            if pulses.__class__ == dataclasses.I3RecoPulseSeriesMapMask:
                pulses = pulses.apply(frame)
        except KeyError:
            # icetray.logging.log_info('Frame doesn\'t contain MCPrimary')
            self.PushFrame(frame)
            return

        dists = []
        charges = []
        for omkey, pulses in pulses:
            # Throw out Deep Core strings (want homogenized total charge)
            if omkey.string >= 79:
                continue
            if (omkey.om >= self.min_DOM) and (omkey.om <= self.max_DOM):
                # Get distance of clostest approach to DOM from track
                dist = self.get_dist(track, self.geomap[omkey].position)
                dists.append(dist)
                # Get charge recorded in DOM
                charge = 0.0
                for pulse_idx, pulse in enumerate(pulses):
                    charge += pulse.charge
                charges.append(charge)

        # Ensure that both dists and charges have non-zero size
        if dists and charges:
            # Add variables to frame
            frame.Put('avg_inice_radius_{}_{}'.format(self.min_DOM, self.max_DOM),
                dataclasses.I3Double( np.mean(dists) ))
            frame.Put('max_inice_radius_{}_{}'.format(self.min_DOM, self.max_DOM),
                dataclasses.I3Double( np.amax(dists) ))

        self.PushFrame(frame)

    def Finish(self):
        return

def add_num_mil_particles(frame):
    n_particles = 0
    if 'Millipede_dEdX' in frame:
        for i3particle in frame['Millipede_dEdX']:
            n_particles += 1
    frame.Put('num_millipede_particles', icetray.I3Int(n_particles))

def addMCprimarykeys(frame):

    if 'MCPrimary' in frame:
        i3primary = frame['MCPrimary']
        frame.Put('MC_x', dataclasses.I3Double(i3primary.pos.x))
        frame.Put('MC_y', dataclasses.I3Double(i3primary.pos.y))
        frame.Put('MC_azimuth', dataclasses.I3Double(i3primary.dir.azimuth))
        frame.Put('MC_zenith', dataclasses.I3Double(i3primary.dir.zenith))
        frame.Put('MC_energy', dataclasses.I3Double(i3primary.energy))
        ts = i3primary.type_string
        print('type_string = {}'.format(ts))
        print('type(type_string) = {}'.format(type(ts)))
        frame.Put('MC_type', dataclasses.I3String(ts))

    return


""" Output number of stations triggered in IceTop """
def GetStations(frame, InputITpulses, output):
    nstation = 0
    if InputITpulses in frame:
        vemPulses = frame[InputITpulses]
        if vemPulses.__class__ == dataclasses.I3RecoPulseSeriesMapMask:
            vemPulses = vemPulses.apply(frame)
        stationList = set([pulse.key().string for pulse in vemPulses])
        nstation = len(stationList)
    frame[output] = icetray.I3Int(nstation)


class AddITContainment(icetray.I3Module):  # Kath's containment
    ''' Icetray module to determine if ShowerLLH reconstructions
        are contained in IceTop '''

    def __init__(self, context):
        icetray.I3Module.__init__(self, context)
        self.AddParameter('LLH_tables', 'LLH table dictionary', None)
        self.AddOutBox('OutBox')

    def Configure(self):
        self.LLH_tables = self.GetParameter('LLH_tables')
        pass

    def Geometry(self, frame):
        self.geometry = frame['I3Geometry']
        self.scaling = phys_services.I3ScaleCalculator(self.geometry)
        self.PushFrame(frame)

    def Physics(self, frame):
        for comp in self.LLH_tables.keys():
            ShowerLLH_particle = 'ShowerLLH_' + comp
            if ShowerLLH_particle in frame:
                frame.Put('ShowerLLH_FractionContainment_{}'.format(comp),
                          dataclasses.I3Double(self.scaling.scale_icetop(frame[ShowerLLH_particle])))

        self.PushFrame(frame)

    def Finish(self):
        return


class AddMCContainment(icetray.I3Module):  # Kath's containment
    ''' Icetray module to determine if ShowerLLH reconstructions
        are contained in IceCube '''

    def __init__(self, context):
        icetray.I3Module.__init__(self, context)
        self.AddOutBox('OutBox')

    def Configure(self):
        pass

    def Geometry(self, frame):
        self.geometry = frame['I3Geometry']
        self.scaling = phys_services.I3ScaleCalculator(self.geometry)
        self.PushFrame(frame)

    def Physics(self, frame):
        if 'MCPrimary' in frame:
            frame.Put('InIce_FractionContainment',
                      dataclasses.I3Double(self.scaling.scale_inice(frame['MCPrimary'])))
            frame.Put('IceTop_FractionContainment',
                      dataclasses.I3Double(self.scaling.scale_icetop(frame['MCPrimary'])))

        self.PushFrame(frame)

    def Finish(self):
        return


class AddInIceRecoContainment(icetray.I3Module):  # Kath's containment
    ''' Icetray module to determine if ShowerLLH reconstructions
        are contained in IceCube '''

    def __init__(self, context):
        icetray.I3Module.__init__(self, context)
        self.AddOutBox('OutBox')

    def Configure(self):
        pass

    def Geometry(self, frame):
        self.geometry = frame['I3Geometry']
        self.scaling = phys_services.I3ScaleCalculator(self.geometry)
        self.PushFrame(frame)

    def Physics(self, frame):
        if 'Laputop' in frame:
            lap_particle = frame['Laputop']
            if (lap_particle.fit_status == dataclasses.I3Particle.OK):
                frame.Put('Laputop_InIce_FractionContainment',
                        dataclasses.I3Double(self.scaling.scale_inice(lap_particle)))
                frame.Put('Laputop_IceTop_FractionContainment',
                        dataclasses.I3Double(self.scaling.scale_icetop(lap_particle)))
        if 'CoincMuonReco_LineFit' in frame:
            I3_particle = frame['CoincMuonReco_LineFit']
            frame.Put('LineFit_InIce_FractionContainment',
                      dataclasses.I3Double(self.scaling.scale_inice(I3_particle)))

        self.PushFrame(frame)

    def Finish(self):
        return


class AddInIceCharge(icetray.I3Module):

    def __init__(self, context):
        icetray.I3Module.__init__(self, context)
        self.AddOutBox('OutBox')
        self.AddParameter('inice_pulses',
                          'I3RecoPulseSeriesMapMask to use for total charge',
                          'SRTCoincPulses')
        self.AddParameter('min_DOM', 'Minimum DOM number to be considered', 1)
        self.AddParameter('max_DOM', 'Maximum DOM number to be considered', 60)

    def Configure(self):
        self.inice_pulses = self.GetParameter('inice_pulses')
        self.min_DOM = self.GetParameter('min_DOM')
        self.max_DOM = self.GetParameter('max_DOM')
        pass

    def Physics(self, frame):
        q_tot = NaN
        n_channels = 0
        n_hits = 0
        max_qfrac = NaN
        if self.inice_pulses in frame:
            VEMpulses = frame[self.inice_pulses]
            if VEMpulses.__class__ == dataclasses.I3RecoPulseSeriesMapMask:
                VEMpulses = VEMpulses.apply(frame)
                charge_list = []
                for omkey, pulses in VEMpulses:
                    # Throw out Deep Core strings (want homogenized total charge)
                    if omkey.string >= 79:
                        continue
                    if (omkey.om >= self.min_DOM) and (omkey.om <= self.max_DOM):
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

class AddIceTopCharge(icetray.I3Module):

    def __init__(self, context):
        icetray.I3Module.__init__(self, context)
        self.AddOutBox('OutBox')
        self.AddParameter('icetop_pulses',
                          'I3RecoPulseSeriesMapMask to use for total charge',
                          'IceTopHLCSeedRTPulses')

    def Configure(self):
        self.icetop_pulses = self.GetParameter('icetop_pulses')
        pass

    def Physics(self, frame):
        q_tot = NaN
        if self.icetop_pulses in frame:
            VEMpulses = frame[self.icetop_pulses]
            if VEMpulses.__class__ == dataclasses.I3RecoPulseSeriesMapMask:
                VEMpulses = VEMpulses.apply(frame)
                charge_list = []
                for omkey, pulses in VEMpulses:
                    for pulse in pulses:
                        charge_list.append(pulse.charge)
                q_tot = np.sum(charge_list)

        frame.Put('IceTop_charge', dataclasses.I3Double(q_tot))
        self.PushFrame(frame)

    def Finish(self):
        return


# ============================================================================
# Modules used for ShowerLLH cuts

""" Move MCPrimary from P to Q frame """


class moveMCPrimary(icetray.I3PacketModule):

    def __init__(self, ctx):
        icetray.I3PacketModule.__init__(self, ctx, icetray.I3Frame.DAQ)
        self.AddOutBox("OutBox")

    def Configure(self):
        pass

    def FramePacket(self, frames):
        qframe = frames[0]
        if len(frames) <= 1 or 'MCPrimary' in qframe:
            for frame in frames:
                self.PushFrame(frame)
            return

        # prim, prim_info = 0,0
        pframes = frames[1:]
        primary_found = False
        for frame in pframes:
            if 'MCPrimary' in frame:
                # if prim != 0:
                if primary_found:
                    raise RuntimeError("MCPrimary in more than one P frame!")
                prim = frame['MCPrimary']
                primary_found = True
                #prim_info = frame['MCPrimaryInfo']
                del frame['MCPrimary']
                del frame['MCPrimaryInfo']

        qframe['MCPrimary'] = prim

        self.PushFrame(qframe)
        for frame in pframes:
            self.PushFrame(frame)


""" Find the loudest station and station with loudest tank """


class FindLoudestStation(icetray.I3Module):

    def __init__(self, context):
        icetray.I3Module.__init__(self, context)
        self.AddParameter('InputITpulses', 'Which IceTop Pulses to use', 0)
        self.AddParameter('SaturationValue', 'Saturation value (VEM)', 600)
        self.AddParameter(
            'output', 'Name of vector with saturated stations', '')
        self.AddOutBox('OutBox')

    def Configure(self):
        self.pulses = self.GetParameter('InputITpulses')
        self.saturation = self.GetParameter('SaturationValue')
        self.outputName = self.GetParameter('output')

    def Physics(self, frame):

        if self.pulses not in frame:
            self.PushFrame(frame)
            return

        vem = frame[self.pulses]
        if vem.__class__ == dataclasses.I3RecoPulseSeriesMapMask:
            vem = vem.apply(frame)

        loudPulse, loudStaCharge, avStaCharge = 0, 0, 0
        loudStation1, loudStation2, loudStation3 = 0, 0, 0
        prevSta, staCharge = 0, 0
        sat_stations = []

        for key, series in vem:
            for tank in series:  # will be one waveform anyway

                # if NaN : rely on waveform in other tank, so skip
                if tank.charge != tank.charge:
                    continue

                # Keep track of largest single pulse
                if tank.charge > loudPulse:
                    loudPulse = tank.charge
                    loudStation1 = key.string

                # Calculate total station charge
                if key.string != prevSta:
                    staCharge = tank.charge
                else:
                    staCharge += tank.charge

                if staCharge > loudStaCharge:
                    loudStaCharge = staCharge
                    loudStation2 = key.string
                prevSta = key.string

                # LG saturaton bookkeeping :
                if tank.charge > self.saturation:
                    sat_stations.append(key.string)

        # Write to frame
        frame['StationWithLoudestPulse'] = dataclasses.I3Double(loudStation1)
        frame['LoudestStation'] = dataclasses.I3Double(loudStation2)
        # Option for writing saturated stations
        if self.outputName != '':
            sat_stations = set(sat_stations)
            sta_list = dataclasses.I3VectorInt()
            for sta in sat_stations:
                sta_list.append(sta)
            frame[self.outputName] = sta_list

        self.PushFrame(frame)

# Is the loudest station OR any saturated station on the edge


class LoudestStationOnEdge(icetray.I3Module):

    def __init__(self, context):
        icetray.I3Module.__init__(self, context)
        self.AddParameter('InputLoudestStation',
                          'Loudest Station (or I3Vector of saturated stations)', 0)
        self.AddParameter('config', 'Which detectorConfig? IT40/59/73/81? ', 0)
        self.AddParameter('output', 'Output bool in the frame', 0)
        self.AddOutBox('OutBox')

    def Configure(self):
        self.loudest = self.GetParameter('InputLoudestStation')
        self.config = self.GetParameter('config')
        self.outputName = self.GetParameter('output')
        IT40edges = [21, 30, 40, 50, 59, 67, 74,
                     73, 72, 78, 77, 76, 75, 68, 60, 52, 53]
        IT40edges += [44, 45, 46, 47, 38, 29]
        IT59edges = [2, 3, 4, 5, 6, 13, 21, 30, 40,
                     50, 59, 67, 74, 73, 72, 78, 77, 76, 75]
        IT59edges += [68, 60, 52, 53, 44, 45, 36, 26, 17, 9]
        IT73edges = [2, 3, 4, 5, 6, 13, 21, 30, 40,
                     50, 59, 67, 74, 73, 72, 78, 77, 76, 75]
        IT73edges += [68, 60, 51, 41, 32, 23, 15, 8]
        IT81edges = [2, 3, 4, 5, 6, 13, 21, 30, 40,
                     50, 59, 67, 74, 73, 72, 78, 77, 76, 75]
        IT81edges += [68, 60, 51, 41, 31, 22, 14, 7, 1]
        self.edgeDict = dict({'IT40': IT40edges, 'IT59': IT59edges})
        self.edgeDict.update({'IT73': IT73edges, 'IT81': IT81edges})

    def Physics(self, frame):

        if self.loudest not in frame:
            self.PushFrame(frame)
            return

        loud = frame[self.loudest]  # is an I3Double or I3VectorInt
        edge = False
        if self.config not in self.edgeDict.keys():
            raise RuntimeError('Unknown config, Please choose from IT40-IT81')

        edgeList = self.edgeDict[self.config]

        # Check if loudest station on edge
        if loud.__class__ == dataclasses.I3Double:
            if loud.value in edgeList:
                edge = True
        # Check if any saturated stations on edge
        elif loud.__class__ == dataclasses.I3VectorInt:
            for station in loud:
                if station in edgeList:
                    edge = True

        output = icetray.I3Bool(edge)
        frame[self.outputName] = output

        self.PushFrame(frame)


# Calculate the largest n pulses and neighbor to the largest one (Q1b)
class LargestTankCharges(icetray.I3Module):

    def __init__(self, context):
        icetray.I3Module.__init__(self, context)
        self.AddParameter('nPulses',
                          'Book largest N pulses for TailCut (+neighbor of largest)', 4)
        self.AddParameter('ITpulses', 'IT pulses Name', 0)
        self.AddOutBox('OutBox')

    def Configure(self):
        self.nPulses = self.GetParameter('nPulses')
        self.recoPulses = self.GetParameter('ITpulses')
        self.counter = 0

    def Physics(self, frame):

        if self.recoPulses not in frame:
            self.PushFrame(frame)
            return

        tank_map = frame[self.recoPulses]
        if tank_map.__class__ == dataclasses.I3RecoPulseSeriesMapMask:
            tank_map = tank_map.apply(frame)

        # Build list of charges and corresponding om's
        charge_map = []
        for om, pulses in tank_map:
            for wave in pulses:
                # If nan, use charge in other tank as best approximation
                if wave.charge != wave.charge:
                    # Get neighboring charge
                    omList = [om1 for om1, pulses in tank_map if om1 != om]
                    stringList = [om1.string for om1 in omList]
                    try:
                        index = stringList.index(om.string)
                    except ValueError:      # pulse cleaning removed one tank
                        continue
                    om_neighbor = omList[index]
                    pulses = tank_map[om_neighbor]
                    charge = pulses[0].charge
                else:
                    charge = wave.charge
                charge_map.append((charge, om))

        if len(charge_map) < 1:
            self.PushFrame(frame)
            return

        charge_map = sorted(charge_map, reverse=True)
        q1 = charge_map[0][0]
        q1_dom = charge_map[0][1]

        # Get charge in neighbor to largest pulse (Q1b)
        omList = [om1 for om1, pulses in tank_map if om1 != q1_dom]
        stringList = [om1.string for om1 in omList]
        if q1_dom.string in stringList:
            index = stringList.index(q1_dom.string)
            q1b_dom = omList[index]
            q1b = tank_map[q1b_dom][0].charge
            frame['Q1b'] = dataclasses.I3Double(q1b)

        # Write charges to frame
        bookedN = 0
        while (bookedN < self.nPulses) and (bookedN < charge_map.__len__()):
            name = 'Q%i' % (bookedN + 1)
            frame[name] = dataclasses.I3Double(charge_map[bookedN][0])
            bookedN += 1

        self.PushFrame(frame)
