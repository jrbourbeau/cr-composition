
#Python Lib imports
import cPickle as pickle
import numpy as np

#IceTray imports
from icecube import icetray, dataio, dataclasses, phys_services, toprec
from I3Tray import *
from icecube.dataclasses import I3Constants
from icecube.icetray.i3logging import log_info, log_warn, log_debug, log_trace, log_error
from icecube.recclasses import I3LaputopParams, LaputopParameter

# Function/Global var imports
from llh_ratio_test_library import llh_ratio_test
from llh_ratio_test_globals import globalhistbins as histbins
from llh_ratio_test_globals import globalhistrange as histrange
from llh_ratio_test_globals import zenith_bins,s125_bins, correct_slc_time
from tools import to_shower_cs


class IceTop_LLH_Ratio(icetray.I3ConditionalModule):
    def __init__(self,ctx):
        icetray.I3ConditionalModule.__init__(self,ctx)
        self.AddParameter('Track','Name of Laputop Frame Obj',
                         'Laputop')
        self.AddParameter('SLCTankPulses','SLC Tank Pulses to be used',
                         'IceTopLaputopSeededSelectedSLC')
        self.AddParameter('HLCTankPulses','HLC Tank Pulses to be used',
                         'IceTopHLCSeedRTPulses')
        self.AddParameter('Output','o/p Frame obj name',
                         'IceTopLLHRatio')
        self.AddParameter('SLCTimeCorrectionPickle','Pickle that contains Time Correction Parametrization',
                          None)
        self.AddParameter('highEbins','If true: s125_bins upto 100 PeV will be taken into consideration. Else upto 10 PeV',
                          False)
        self.AddParameter('GeometryHDF5','I3Geometry booked in HDF file',
                          'geometry.h5')
        self.AddParameter('TwoDPDFPickleYear','Year to get pickle that contains 3 x 2D PDFs for Gamma Sim and Data for all s125/zen bins',
                          None)
        #self.AddOutbox('OutBox')

    def Configure(self):
        self.track = self.GetParameter('Track')
        self.slcpulses = self.GetParameter('SLCTankPulses')
        self.hlcpulses = self.GetParameter('HLCTankPulses')
        self.slc_time_corr = self.GetParameter('SLCTimeCorrectionPickle')
        self.objname = self.GetParameter('Output')
        self.geoh5 = self.GetParameter('GeometryHDF5')
        self.highE = self.GetParameter('highEbins')
        year = self.GetParameter('TwoDPDFPickleYear')

        if year == None:
            self.histpickle = None
        else:
            pf = '/data/user/hpandya/gamma_combined_scripts/resources/'
            pickles = {'2011':pf+'12622_2011GammaSim_BurnSample_2011.pickle',
                       '2012':pf+'12533_2012GammaSim_BurnSample_2012.pickle',
                       '2013':pf+'12612_2013GammaSim_BurnSample_2013.pickle',
                       '2014':pf+'12613_2014GammaSim_BurnSample_2014.pickle',
                       '2015':pf+'12614_2015GammaSim_BurnSample_2015.pickle'}
            self.histpickle = pickles[year]

        # Initiate LLH Ratio Test class
        self.llh = llh_ratio_test()
        self.llh.load_geometry(self.geoh5)
        self.llh.load_heatmaps(self.histpickle)
        self.llh.normalize_heatmaps_new(smearedthumbplot=False, normradialbins=False)

        # Load SLC Time Correction Pickle
        if self.slc_time_corr != None:
            f=open(self.slc_time_corr,'r')
            self.mean_slc_charge = np.array( pickle.load(f) )
            self.median_time_diff = np.array( pickle.load(f) )
            variance_time = np.array( pickle.load(f) )
            f.close()

    def Geometry(self,frame):
        self.geometry = frame['I3Geometry']
        self.PushFrame(frame)


    def Physics(self,frame):
        # Load reconstructed quantities
        laputop=frame[self.track]
        coszen = np.cos(laputop.dir.zenith)

        Par = LaputopParameter
        params = I3LaputopParams.from_frame(frame, self.track+'Params')
        logs125= params.value(Par.Log10_S125)

        # Locate the Zen/S125 bin to use appropriate 2D PDF
        for zen_high,zen_low in zenith_bins:
            if coszen < zen_high and coszen >= zen_low:
                break
            zen_high = None

        #if not self.highEbins:
        if not self.highE:
            s125_bins = np.array(zip(np.arange(-0.5,1,0.1),np.arange(-0.4,1.1,0.1)))
        else:
            s125_bins = np.array(zip(np.arange(-0.5,2.1,0.1),np.arange(-0.4,2.2,0.1)))

        for s125_low,s125_high in s125_bins:
            if logs125 < s125_high and logs125 >= s125_low:
                break
            s125_high = None

        # See whether Pulse Containers are found in frame
        good_event = self.slcpulses in frame or self.hlcpulses in frame
        if not good_event:
            log_info('Either %s or %s missing in frame. LLH Ratio not being calculated.'%(self.slcpulses,self.hlcpulses))

        # Store Nans if event out of S125/Zen bin or Pulse Containers not found
        if s125_high==None or zen_high==None or not good_event:
            dict={}
            for key in ['q_r','q_t','t_r']:
                dict['LLH_Hadron_%s'%key]=np.nan
                dict['LLH_Gamma_%s'%key]=np.nan
            dict['LLH_Ratio']=np.nan
            frame.Put(self.objname, dataclasses.I3MapStringDouble(dict))
            self.PushFrame(frame)
            return

        self.llh.events={}
        event_doms=[]

        axis=frame[self.track]
        rotation = to_shower_cs(axis)
        origin = np.array([[axis.pos.x], [axis.pos.y], [axis.pos.z]])
        self.llh.events['laputop_x']=[axis.pos.x] #len(self.events['laputop_x']) used for a loop later

        for pulsename,tag in [[self.slcpulses,'slc'], [self.hlcpulses,'hlc']]:

            if pulsename  not in frame:
                log_warn('%s not found in frame'%pulsename)
                continue

            self.llh.events[tag+'_rperp']=[]
            self.llh.events[tag+'_q']=[]
            self.llh.events[tag+'_t']=[]

            pulses = dataclasses.I3RecoPulseSeriesMap.from_frame(frame,pulsename)

            for k,m in pulses.iteritems():
                event_doms.append((k[0],k[1],k[2]))
                for i,pulse in enumerate(m):
                    if not k in self.geometry.omgeo:
                        log_fatal("OM {om} not in geometry!".format(om=k))
                        continue
                    position = self.geometry.omgeo[k].position
                    det_cs_position = np.array([[position.x],
                                      [position.y],
                                      [position.z]])
                    shower_cs_position = rotation*(det_cs_position - origin)
                    shower_cs_radius = np.sqrt(shower_cs_position[0]**2 + shower_cs_position[1]**2)
                    time = pulse.time - float(axis.time - shower_cs_position[2]/ I3Constants.c)

                    # Correct SLC Time Stamp if SLC Time Correction Pickle has been supplied
                    if self.slc_time_corr!=None and pulsename==self.slcpulses:
                        newtime = correct_slc_time(self.mean_slc_charge, self.median_time_diff, time, pulse.charge)
                        time = newtime

                    self.llh.events[tag+'_rperp'].append(np.log10(np.float(shower_cs_radius)))
                    self.llh.events[tag+'_q'].append(np.log10(pulse.charge))
                    self.llh.events[tag+'_t'].append(np.log10(time))

        nohit_t,nohit_q,nohit_rperp,nohit_om,nohit_string= self.llh.no_hit_doms(event_doms,axis.dir.azimuth,axis.dir.zenith,axis.pos.x,axis.pos.y)
        self.llh.events['nohit_t'] = nohit_t
        self.llh.events['nohit_q'] = nohit_q
        self.llh.events['nohit_rperp'] = nohit_rperp
        self.llh.events['nohit_om'] = nohit_om
        self.llh.events['nohit_string'] = nohit_string

        # need to convert each list to list of list.
        for key in self.llh.events.keys():
            self.llh.events[key] = [self.llh.events[key]]

        # print('s125_high = {}'.format(s125_high))
        # print('str(s125_high) = {}'.format(str(s125_high)))
        self.llh.calc_llh_values_new(histbins=histbins,
                                     histrange=histrange,
                                     s125=str(s125_high),
                                     zen=str(zen_high),
                                     ignorezerobinsinboth=True,
                                     generatepdf=False,
                                     pdffileinstance=None,
                                     trace_tanks=False)

        dict={}
        LLHRatio=0
        for key in ['q_r','q_t','t_r']:
            dict['LLH_Hadron_%s'%key]=self.llh.events['proton_like'][key][0]
            dict['LLH_Gamma_%s'%key]=self.llh.events['gamma_like'][key][0]
            LLHRatio += self.llh.events['proton_like'][key][0] - self.llh.events['gamma_like'][key][0]
        dict['LLH_Ratio']=LLHRatio
        frame.Put(self.objname, dataclasses.I3MapStringDouble(dict))

        # Event arrays need to be cleared out:
        del self.llh.events

        self.PushFrame(frame)
        return
