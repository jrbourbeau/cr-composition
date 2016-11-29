#!/usr/bin/env python

import numpy as np
from I3Tray import NaN, Inf
from icecube import icetray, dataio, toprec, phys_services
from icecube import dataclasses as dc
from icecube.icetop_Level3_scripts import icetop_globals
from icecube.icetop_Level3_scripts.functions import count_stations

# ============================================================================
# Generally useful modules

""" Output number of stations triggered in IceTop """


def GetStations(frame, InputITpulses, output):
    nstation = 0
    if InputITpulses in frame:
        vemPulses = frame[InputITpulses]
        if vemPulses.__class__ == dc.I3RecoPulseSeriesMapMask:
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
                          dc.I3Double(self.scaling.scale_icetop(frame[ShowerLLH_particle])))

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
                      dc.I3Double(self.scaling.scale_inice(frame['MCPrimary'])))
            frame.Put('IceTop_FractionContainment',
                      dc.I3Double(self.scaling.scale_icetop(frame['MCPrimary'])))

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
        if 'CoincMuonReco_LineFit' in frame:
            I3_particle = frame['CoincMuonReco_LineFit']
            frame.Put('LineFit_InIce_FractionContainment',
                      dc.I3Double(self.scaling.scale_inice(I3_particle)))

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
        self.AddParameter('min_DOM',
                          'Minimum DOM number to be considered', 1)
        self.AddParameter('max_DOM',
                          'Maximum DOM number to be considered', 60)

    def Configure(self):
        self.inice_pulses = self.GetParameter('inice_pulses')
        self.min_DOM = self.GetParameter('min_DOM')
        self.max_DOM = self.GetParameter('max_DOM')
        pass

    def Physics(self, frame):
        q_tot = NaN
        n_channels = 0
        max_qfrac = NaN
        if self.inice_pulses in frame:
            VEMpulses = frame[self.inice_pulses]
            if VEMpulses.__class__ == dc.I3RecoPulseSeriesMapMask:
                VEMpulses = VEMpulses.apply(frame)
                # n_channels = 0
                charge_list = []
                for omkey, pulses in VEMpulses:
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
                    max_qfrac = np.max(charge_list)/q_tot

        frame.Put('InIce_charge_{}_{}'.format(self.min_DOM, self.max_DOM),
                  dc.I3Double(q_tot))
        frame.Put('NChannels_{}_{}'.format(self.min_DOM, self.max_DOM),
                  icetray.I3Int(n_channels))
        frame.Put('max_qfrac_{}_{}'.format(self.min_DOM, self.max_DOM),
                  dc.I3Double(max_qfrac))
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
        if vem.__class__ == dc.I3RecoPulseSeriesMapMask:
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
        frame['StationWithLoudestPulse'] = dc.I3Double(loudStation1)
        frame['LoudestStation'] = dc.I3Double(loudStation2)
        # Option for writing saturated stations
        if self.outputName != '':
            sat_stations = set(sat_stations)
            sta_list = dc.I3VectorInt()
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
        if loud.__class__ == dc.I3Double:
            if loud.value in edgeList:
                edge = True
        # Check if any saturated stations on edge
        elif loud.__class__ == dc.I3VectorInt:
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
        if tank_map.__class__ == dc.I3RecoPulseSeriesMapMask:
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
            frame['Q1b'] = dc.I3Double(q1b)

        # Write charges to frame
        bookedN = 0
        while (bookedN < self.nPulses) and (bookedN < charge_map.__len__()):
            name = 'Q%i' % (bookedN + 1)
            frame[name] = dc.I3Double(charge_map[bookedN][0])
            bookedN += 1

        self.PushFrame(frame)
