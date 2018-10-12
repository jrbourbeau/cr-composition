import os
import tables
import numpy as np
import random
import cPickle as pickle
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.backends.backend_pdf import PdfPages
from llh_ratio_test_globals import global3dbins, global3drange

class llh_ratio_test(object):


    def __init__(self):
        """ Does nothing"""
        self.tangerine = 'abba dabba chabba'
        return

    def load_geometry(self,geofile=os.getcwd()+'/geometry.h5'):
        """ From a geometry file, loads geometry of the detector"""
        geometry = tables.open_file(geofile)
        # geometry = tables.openFile(geofile)
        self.geo_doms=[]
        self.geo_X = []
        self.geo_Y = []
        self.geo_Z = []

        for i in range(324):
            j = 'geometry.root.mygeometry.cols.dom%i'%i
            #BEWARE: This is a DOM list not Tank list
            self.geo_doms.append((np.int(eval(j+'[0]')),np.int(eval(j+'[1]')), np.int(eval(j+'[2]'))))
            self.geo_X.append(eval(j+'[3]'))
            self.geo_Y.append(eval(j+'[4]'))
            self.geo_Z.append(eval(j+'[5]'))

        geometry.close()

        self.geo_X = np.array(self.geo_X)
        self.geo_Y = np.array(self.geo_Y)
        self.geo_Z = np.array(self.geo_Z)

        return

    def rotate_to_shower_cs(self,x,y,z,phi,theta,core_x,core_y,core_z):
        """ Input: X,Y,Z of the dom/particle; Phi, Theta, Core x,y ,z of the shower.
            Output: Radial distance from shower core in shower coordinate system."""

        # counter-clockwise (pi + phi) rotation
        d_phi = np.matrix([ [ -np.cos(phi), -np.sin(phi), 0],
        [  np.sin(phi), -np.cos(phi), 0],
        [  0,                 0,                1] ])
        # clock-wise (pi - theta) rotation
        d_theta = np.matrix([ [  -np.cos(theta), 0, -np.sin(theta)],
        [  0,                  1,  0,                ],
        [  np.sin(theta), 0,  -np.cos(theta)] ])
        rotation= d_theta*d_phi

        origin = np.array([[core_x], [core_y], [core_z]])

        det_cs_position = np.array([[x],[y],[z]])
        shower_cs_position = rotation*(det_cs_position - origin)
        shower_cs_radius = np.sqrt(shower_cs_position[0]**2 + shower_cs_position[1]**2)

        return np.float(shower_cs_radius)

    def no_hit_doms(self,event_doms, phi, theta, core_x, core_y):
        """ Input: Event Doms list
            Output: No Hit Doms Location in Shower Coordinate System and Their Charge 1e-3 VEM for underflow bin."""

        qnohit =[]
        rnohit = []
        omnohit= []
        stringnohit = []

        #print event_doms
        #print '----------'
        for m in range(len(self.geo_doms)):
            if self.geo_doms[m] not in event_doms:
                # Check if other DOM from tank is present:
                if self.geo_doms[m][1]==61 and (self.geo_doms[m][0],62,self.geo_doms[m][2]) in event_doms:
                    #print 'other dom present',geo_doms[m]
                    continue
                if self.geo_doms[m][1]==62 and (self.geo_doms[m][0],61,self.geo_doms[m][2]) in event_doms:
                    #print 'other dom present',geo_doms[m]
                    continue
                if self.geo_doms[m][1]==63 and (self.geo_doms[m][0],64,self.geo_doms[m][2]) in event_doms:
                    #print 'other dom present',geo_doms[m]
                    continue
                if self.geo_doms[m][1]==64 and (self.geo_doms[m][0],63,self.geo_doms[m][2]) in event_doms:
                    #print 'other dom present',geo_doms[m]
                    continue
                #print 'this tank is not present:',geo_doms[m]
                rnohit.append(self.rotate_to_shower_cs(self.geo_X[m], self.geo_Y[m], 0, phi, theta, core_x, core_y, 0))
                qnohit.append(1e-3)
                event_doms.append(self.geo_doms[m])
                omnohit.append(self.geo_doms[m][1])
                stringnohit.append(self.geo_doms[m][0])

        #print len(event_doms)
        qnohit = np.log10(qnohit)
        rnohit = np.log10(rnohit)
        tnohit = np.ones_like(rnohit)*-2
        omnohit = np.array(omnohit)
        stringnohit = np.array(stringnohit)

        return tnohit, qnohit,rnohit,omnohit,stringnohit

    def load_hdf_file(self,file,llh_calculated=False,skip_non_llh=False,skylab=False,isMC=False):
        """ Load all necessary variables from a hdf file. """

        self.f=tables.openFile(file)

        self.events={}
        self.hist={}

        self.events['cos_zen']=np.cos(self.f.root.Laputop.cols.zenith[:])
        self.events['log_s125']=np.log10(self.f.root.LaputopParams.cols.s125[:])
        self.events['beta']=self.f.root.LaputopParams.cols.beta[:]
        self.events['inice_count'] = np.zeros(len(self.events['beta']))#np.array(self.f.root.hlc_count_InIcePulses.cols.value[:])
        #self.events['inice_count'] += 0#np.array(self.f.root.slc_count_InIcePulses.cols.value[:])
        #self.events['thumb_q'] = self.f.root.ThumbRegion_2_0_0_7_0_4.cols.Sum_Q_Thumb[:]

        if skylab:
            from icecube import astro
            ra,dec= astro.tables_to_equa(self.f.root.Laputop,self.f.root.I3EventHeader)
            #using azimuth as ra (since azimuth is randomly sampled in simulation while I don't know time sampling)
            #another option is to pick mjd randomly from a year and use that to go from azi to ra.
            #which is more work.
            self.events['laputop_ra']=self.f.root.Laputop.cols.azimuth[:]
            self.events['laputop_dec']=dec
            self.events['run']=self.f.root.I3EventHeader.cols.Run[:]
            self.events['event']=self.f.root.I3EventHeader.cols.Event[:]
            self.events['time']= (self.f.root.I3EventHeader.cols.time_start_mjd_day[:]
                                + self.f.root.I3EventHeader.cols.time_start_mjd_sec[:]/86400.)

            if isMC:
                self.events['true_zen']=self.f.root.MCPrimary.cols.zenith[:]
                self.events['true_azimuth']=self.f.root.MCPrimary.cols.azimuth[:]
                self.events['true_E']=self.f.root.MCPrimary.cols.energy[:]
                ra,dec= astro.tables_to_equa(self.f.root.MCPrimary,self.f.root.I3EventHeader)
                #using azimuth as ra (since azimuth is randomly sampled in simulation while I don't know time sampling)
                #another option is to pick mjd randomly from a year and use that to go from azi to ra.
                #which is more work.
                self.events['true_ra']=self.f.root.MCPrimary.cols.azimuth[:]
                self.events['true_dec']=dec


        if not skip_non_llh:
            self.events['start_slc']=self.f.root.__I3Index__.IceTopLaputopSeededSelectedSLC.cols.start[:]
            self.events['stop_slc']=self.f.root.__I3Index__.IceTopLaputopSeededSelectedSLC.cols.stop[:]
            self.events['start_hlc']=self.f.root.__I3Index__.IceTopLaputopSeededSelectedHLC.cols.start[:]
            self.events['stop_hlc']=self.f.root.__I3Index__.IceTopLaputopSeededSelectedHLC.cols.stop[:]
            self.events['laputop_azi'] = self.f.root.Laputop.cols.azimuth[:]
            self.events['laputop_zen'] = self.f.root.Laputop.cols.zenith[:]
            self.events['laputop_x'] = self.f.root.Laputop.cols.x[:]
            self.events['laputop_y'] = self.f.root.Laputop.cols.y[:]
            self.events['laputop_z'] = self.f.root.Laputop.cols.z[:]

        if llh_calculated==True:
            self.events['Laputop_E']=self.f.root.Laputop_E.cols.value[:]
            for key in ['q_r','q_t','t_r']:
                self.events['LLH_Hadron_%s'%key]=eval('self.f.root.GammaRayAnalysis_LLH.cols.LLH_Hadron_%s[:]'%key)
                self.events['LLH_Gamma_%s'%key]=eval('self.f.root.GammaRayAnalysis_LLH.cols.LLH_Gamma_%s[:]'%key)
            self.events['LLH_Ratio']=eval('self.f.root.GammaRayAnalysis_LLH.cols.LLH_Ratio[:]')

            #self.events['hlc_count_InIcePulses']=self.f.root.hlc_count_InIcePulses.cols.value[:]
            #self.events['slc_count_InIcePulses']=self.f.root.slc_count_InIcePulses.cols.value[:]

            #self.events['hlc_count_SRTCoincPulses']=self.f.root.hlc_count_SRTCoincPulses.cols.value[:]
            #self.events['slc_count_SRTCoincPulses']=self.f.root.slc_count_SRTCoincPulses.cols.value[:]

            #self.events['hlc_count_CoincPulses']=self.f.root.hlc_count_CoincPulses.cols.value[:]
            #self.events['slc_count_CoincPulses']=self.f.root.slc_count_CoincPulses.cols.value[:]

        return

    def make_quality_cuts(self):
        """ Make Quality Cuts """

        #Check Trigger
        cut1 = self.f.root.IT73AnalysisIceTopQualityCuts.cols.IceTop_StandardFilter[:] == 1.0
        #Did Laputop Converge?
        cut5 = self.f.root.IT73AnalysisIceTopQualityCuts.cols.IceTop_reco_succeeded[:] == 1.0
        #Check Spectrum Paper Cuts - Bakhtiyars paper
        cut2 = self.f.root.IT73AnalysisIceTopQualityCuts.cols.IceTopMaxSignalInside[:] == 1.0
        cut4 = self.f.root.IT73AnalysisIceTopQualityCuts.cols.Laputop_FractionContainment[:] == 1.0 # i.e. fraction<0.96
        cut7 = self.f.root.IT73AnalysisIceTopQualityCuts.cols.IceTopMaxSignalAbove6[:] == 1.0
        cut6 = self.f.root.NStation.cols.value[:] >= 5.0

        #Beta cut also used now
        cut3 = self.f.root.IT73AnalysisIceTopQualityCuts.cols.BetaCutPassed[:] == 1.0

        bool = (cut1)&(cut2)&(cut3)&(cut4)&(cut5)&(cut6)&(cut7)

        for key in self.events:
            self.events[key]=self.events[key][bool]

        return

    def make_quality_cuts_zach(self):
        """ Make Quality Cuts same as Zach"""

        ic_containment = self.f.root.Laputop_inice_FractionContainment.cols.value[:]
        cut1 = ic_containment<=1.0

        laputop_E = self.events['Laputop_E']
        cut2 = np.log10(laputop_E)>5.65

        bool = (cut1)&(cut2)

        for key in self.events:
            self.events[key]=self.events[key][bool]

        return

    def make_cut(self,key,low, high):
        """ Make Log S125 Cut """

        cut1 = self.events[key] >= low
        cut2 = self.events[key] < high

        bool = (cut1)&(cut2)

        for key in self.events:
            self.events[key]=self.events[key][bool]

        return

    def make_s125_cut(self,low, high):
        """ Make Log S125 Cut """
        self.make_cut('log_s125',low,high)
        return

    def make_zenith_cut(self,low,high):
        """ Make Zenith Cut"""
        self.make_cut('cos_zen',low,high)
        return

    def check_survivors(self,var='laputop_x'):
        any_survivors = len(self.events[var])>0
        return any_survivors

    def delete_close(self):
        """ Deletes certain variables from memory and closes open hdf5 file"""

        try:
            del self.hist
        except:
            pass

        del self.events

        print '--close file'
        self.f.close()

        return

    def load_pulses(self,include_slc_time=False, exclude_hlc_time=False,slc_time_corrected=False):
        """ Run this module after cuts.
            It loads slc, hlc and no hit q and rperp containers"""

        if not self.check_survivors():
            print 'no surviving events after cuts'
            print 'not loading pulse containers'
            return

        self.events['slc_rperp']=[]
        self.events['slc_q']=[]
        self.events['slc_t']=[]
        self.events['slc_T']=[]
        self.events['slc_om']=[]
        self.events['slc_string']=[]
        self.events['hlc_rperp']=[]
        self.events['hlc_q']=[]
        self.events['hlc_t']=[]
        self.events['hlc_T']=[]
        self.events['hlc_om']=[]
        self.events['hlc_string']=[]
        self.events['nohit_rperp']=[]
        self.events['nohit_q']=[]
        self.events['nohit_t']=[]
        self.events['nohit_om']=[]
        self.events['nohit_string']=[]

        for i,j,k,l,eventid in zip(self.events['start_slc'],self.events['stop_slc'],
                           self.events['start_hlc'],self.events['stop_hlc'], range(len(self.events['start_slc']))):

            xslc=self.f.root.IceTopLaputopSeededSelectedSLC.cols.x[i:j]
            yslc=self.f.root.IceTopLaputopSeededSelectedSLC.cols.y[i:j]
            Xslc=self.f.root.IceTopLaputopSeededSelectedSLC.cols.X[i:j]
            Yslc=self.f.root.IceTopLaputopSeededSelectedSLC.cols.Y[i:j]
            rperpslc= np.log10(np.sqrt(xslc**2 + yslc**2))
            qslc=self.f.root.IceTopLaputopSeededSelectedSLC.cols.charge[i:j]
            qslc= np.log10(qslc)

            if slc_time_corrected:
                tslc=self.f.root.IceTopLaputopSeededSelectedSLC.cols.time_corrected[i:j]
                det_tslc=self.f.root.IceTopLaputopSeededSelectedSLC.cols.T_corrected[i:j]
            else:
                tslc=self.f.root.IceTopLaputopSeededSelectedSLC.cols.time[i:j]
                det_tslc=self.f.root.IceTopLaputopSeededSelectedSLC.cols.T[i:j]


            select = tslc<0
            if len(select[select])>0:
                print 'slc neg time',len(select[select]),'out of ',len(select)
                #print qslc[select],tslc[select],rperpslc[select]
                #print 'changing those to 10^-2'
                tslc[select]=0.01

            if include_slc_time:
                tslc= np.log10(tslc)
            else:
                tslc= np.ones_like(tslc)*-2

            self.events['slc_q'].append(qslc)
            self.events['slc_t'].append(tslc)
            self.events['slc_T'].append(det_tslc)
            self.events['slc_rperp'].append(rperpslc)

            xhlc=self.f.root.IceTopLaputopSeededSelectedHLC.cols.x[k:l]
            yhlc=self.f.root.IceTopLaputopSeededSelectedHLC.cols.y[k:l]
            rperphlc= np.log10(np.sqrt(xhlc**2 + yhlc**2))
            qhlc=self.f.root.IceTopLaputopSeededSelectedHLC.cols.charge[k:l]
            qhlc= np.log10(qhlc)
            thlc=self.f.root.IceTopLaputopSeededSelectedHLC.cols.time[k:l]

            select = thlc<0
            if len(select[select])>0:
                print 'hlc neg time',len(select[select]),'out of ',len(select)
                #print 'changing those to 10^-2'
                thlc[select]=0.01

            if exclude_hlc_time:
                thlc= np.ones_like(thlc)*-2
            else:
                thlc= np.log10(thlc)

            det_thlc=self.f.root.IceTopLaputopSeededSelectedHLC.cols.T[k:l]

            self.events['hlc_q'].append(qhlc)
            self.events['hlc_t'].append(thlc)
            self.events['hlc_T'].append(det_thlc)
            self.events['hlc_rperp'].append(rperphlc)

            stringslc=self.f.root.IceTopLaputopSeededSelectedSLC.cols.string[i:j]
            omslc=self.f.root.IceTopLaputopSeededSelectedSLC.cols.om[i:j]
            pmtslc=self.f.root.IceTopLaputopSeededSelectedSLC.cols.pmt[i:j]
            stringhlc=self.f.root.IceTopLaputopSeededSelectedHLC.cols.string[k:l]
            omhlc=self.f.root.IceTopLaputopSeededSelectedHLC.cols.om[k:l]
            pmthlc=self.f.root.IceTopLaputopSeededSelectedHLC.cols.pmt[k:l]

            self.events['hlc_om'].append(omhlc)
            self.events['hlc_string'].append(stringhlc)
            self.events['slc_om'].append(omslc)
            self.events['slc_string'].append(stringslc)

            event_doms = zip(stringslc,omslc,pmtslc)+zip(stringhlc,omhlc,pmthlc)

            phi = self.events['laputop_azi'][eventid]
            theta = self.events['laputop_zen'][eventid]
            core_x= self.events['laputop_x'][eventid]
            core_y= self.events['laputop_y'][eventid]

            tnohit, qnohit,rperpnohit,omnohit,stringnohit = self.no_hit_doms(event_doms,phi,theta,core_x,core_y)

            self.events['nohit_rperp'].append(rperpnohit)
            self.events['nohit_q'].append(qnohit)
            self.events['nohit_t'].append(tnohit)
            self.events['nohit_om'].append(omnohit)
            self.events['nohit_string'].append(stringnohit)

        i = int(random.random()*len(self.events['slc_rperp']))
        total_tanks = sum([len(self.events['slc_rperp'][i]), len(self.events['hlc_rperp'][i]), len(self.events['nohit_rperp'][i])])

        if total_tanks!=162:
            print ' WARNING: Total Tanks not adding up to 162'

        return

    def histogram2d(self,key,bins, range):
        """ Histogram HLC, SLC, No hit Tanks"""

        self.hist[key] = {}

        for var in ['nohit','hlc','slc']:

            r=np.concatenate(self.events[var+'_rperp'])
            q=np.concatenate(self.events[var+'_q'])
            t= np.concatenate(self.events[var+'_t'])

            x=eval(key[2])
            y=eval(key[0])

            self.hist[key][var], self.hist[key]['xedges'], self.hist[key]['yedges'] = np.histogram2d(x, y, bins = bins, range=range)

        return

    def histogram3d(self):
        """ Histogram HLC, SLC, No hit Tanks"""

        key='3d'
        self.hist[key] = {}

        for var in ['nohit','hlc','slc']:

            r=np.concatenate(self.events[var+'_rperp'])
            q=np.concatenate(self.events[var+'_q'])
            t= np.concatenate(self.events[var+'_t'])

            if len(r)==0:
                self.hist[key][var] = np.zeros(global3dbins)
                continue

            data=np.array(zip(r,q,t))
            bins = global3dbins
            range = global3drange

            self.hist[key][var], self.hist[key]['edges'] = np.histogramdd(data, bins = bins, range=range)

        return

    def load_heatmaps(self,inputpicklefile):
        """ Given a pickle file with appropriate structure,
            Loads Gamma, Proton heat map dicts with s125, zen keys"""

        self.heatmap={}
        nevents={}

        picklefile= open(inputpicklefile,'r')
        self.heatmap['proton'] = pickle.load(picklefile)
        nevents['proton'] = pickle.load(picklefile)

        self.heatmap['gamma'] = pickle.load(picklefile)
        nevents['gamma'] = pickle.load(picklefile)

        self.xedges = pickle.load(picklefile)
        self.yedges = pickle.load(picklefile)

        picklefile.close()

        return

    def load_heatmaps_new(self,inputpicklefile):
        """ FOR loading pickles with 3d heatmaps too
            Given a pickle file with appropriate structure,
            Loads Gamma, Proton heat map dicts with s125, zen keys"""

        self.heatmap={}
        nevents={}

        picklefile= open(inputpicklefile,'r')

        self.heatmap['proton'] = pickle.load(picklefile)
        nevents['proton'] = pickle.load(picklefile)
        self.heatmap['proton']['3d'] = pickle.load(picklefile)

        self.heatmap['gamma'] = pickle.load(picklefile)
        nevents['gamma'] = pickle.load(picklefile)
        self.heatmap['gamma']['3d'] = pickle.load(picklefile)

        xedges = pickle.load(picklefile)
        yedges = pickle.load(picklefile)

        picklefile.close()

        return


    def normalize_heatmaps_new(self,smearedthumbplot=False,normradialbins=False):

        for prim in self.heatmap.keys():
            self.heatmap[prim+'_norm']={}
            for var in self.heatmap[prim].keys():
                self.heatmap[prim+'_norm'][var]={}
                for s125 in self.heatmap[prim][var].keys():
                    self.heatmap[prim+'_norm'][var][s125]={}
                    for zen in self.heatmap[prim][var][s125].keys():
                        temp=self.heatmap[prim][var][s125][zen].copy()

                        if smearedthumbplot==True:
                            temp[temp<=1e-7]=0 # remove extremely small probability values

                        if normradialbins==True:
                            for i in range(len(temp)):
                                if sum(temp[i])!=0:
                                    temp[i]= temp[i]/sum(temp[i])
                        else:
                            temp=temp/np.sum(temp)

                        self.heatmap[prim+'_norm'][var][s125][zen]=temp
        return


    def return_hevent(self,Nevent,histbins,histrange,template_key):
        hist={}
        for key in ['nohit','hlc','slc']:
            r=np.array(self.events[key+'_rperp'][Nevent])
            q=np.array(self.events[key+'_q'][Nevent])
            t=np.array(self.events[key+'_t'][Nevent])

            x=eval(template_key[2])
            y=eval(template_key[0])

            hist[key], xedges, yedges = np.histogram2d(x, y, bins = histbins[template_key], range=histrange[template_key])

        hevent = hist['hlc'] + hist['slc'] + hist['nohit']

        return hevent,xedges,yedges

    def plot_heatmaps(self,s125,zen,template_key,pdffileinstance,xedges,yedges):
        pdf = pdffileinstance
        for key in ['gamma','proton']:
            temp = np.flipud(np.rot90(self.heatmap[key+'_norm'][template_key][s125][zen]))
            temp = np.ma.masked_where(temp==0,temp)

            plt.figure()
            plt.pcolormesh(xedges, yedges, temp, norm=LogNorm(vmin=temp.min(), vmax=temp.max()))
            plt.title('LogS125: %s CosZen: %s'%(s125,zen))

            if template_key == 'q_r':
                plt.xlabel('log r')
                plt.ylabel('log VEM')
            if template_key == 'q_t':
                plt.xlabel('log t')
                plt.ylabel('log VEM')
            if template_key == 't_r':
                plt.ylabel('log t')
                plt.xlabel('log r')

            cbar = plt.colorbar()
            cbar.ax.set_ylabel('Probability')

            if template_key == 'q_r':
                plt.xlim([0,4])
                plt.ylim([-3,4])
            if template_key == 'q_t':
                plt.xlim([-2,4])
                plt.ylim([-3,4])
            if template_key == 't_r':
                plt.xlim([0,4])
                plt.ylim([-2,4])

            pdf.savefig()
            plt.close()
        return

    def make_pdf(self,s125,zen,template_key,pdffileinstance,tempp,tempg,xedges,yedges,i,Tbins,Zbins_g,Zbins_p):
        pdf = pdffileinstance
        for key in ['gamma','proton']:
            temp = np.flipud(np.rot90(self.heatmap[key+'_norm'][template_key][s125][zen]))
            temp = np.ma.masked_where(temp==0,temp)

            plt.figure()
            plt.pcolormesh(xedges, yedges, temp, norm=LogNorm(vmin=temp.min(), vmax=temp.max()))

            if template_key == 'q_r':
                plt.xlabel('log r')
                plt.ylabel('log VEM')
            if template_key == 'q_t':
                plt.xlabel('log t')
                plt.ylabel('log VEM')
            if template_key == 't_r':
                plt.ylabel('log t')
                plt.xlabel('log r')

            cbar = plt.colorbar()
            cbar.ax.set_ylabel('Probability')

            for key1,marker,label in zip(['slc','hlc','nohit'],['o','s','^'],['SLC','HLC','No Hit']):
                r=np.array(self.events[key1+'_rperp'][i])
                q=np.array(self.events[key1+'_q'][i])
                t=np.array(self.events[key1+'_t'][i])

                x=eval(template_key[2])
                y=eval(template_key[0])

                plt.scatter(x,y,marker=marker,facecolor='None',edgecolor='k',s=30,label=label)

            if template_key == 'q_r':
                plt.xlim([0,4])
                plt.ylim([-3,4])
            if template_key == 'q_t':
                plt.xlim([-2,4])
                plt.ylim([-3,4])
            if template_key == 't_r':
                plt.xlim([0,4])
                plt.ylim([-2,4])

            plt.title('LogLp: %.2f, LogLg: %.2f , diff: %.2f  %i %i %i'%(tempp,tempg,tempp-tempg,Tbins,Zbins_g,Zbins_p))
            plt.legend()

            pdf.savefig()
            plt.close()
        return

    def tanks_on_zero_bins(self,protonmap,gammamap,hevent,xedges,yedges,template_key,eventno):
        """ Find which tanks are landing on zero bin in this event"""

        bins_assigned={}
        tanks_zerobins=[]
        tanks_nonzerobins=[]

        for key in ['nohit','hlc','slc']:
            om_non_zero_bins = []
            string_non_zero_bins = []
            om_zero_bins = []
            string_zero_bins = []

            if template_key[2]=='r':
                x=self.events[key+'_%sperp'%template_key[2]][eventno]
            else:
                x=self.events[key+'_%s'%template_key[2]][eventno]

            y=self.events[key+'_%s'%template_key[0]][eventno]
            om=self.events[key+'_om'][eventno]
            string=self.events[key+'_string'][eventno]

            bins_assigned[key]=[]
            for i in range(len(x)):
                for j in range(len(xedges)-1):
                    for k in range(len(yedges)-1):
                        if xedges[j] <= x[i] < xedges[j+1]:
                            if yedges[k] <= y[i] < yedges[k+1]:
                                bins_assigned[key].append((j,k))
                                if protonmap[j,k]!=0 and gammamap[j,k]!=0:
                                    om_non_zero_bins.append(om[i])
                                    string_non_zero_bins.append(string[i])
                                else:
                                    om_zero_bins.append(om[i])
                                    string_zero_bins.append(string[i])
            #om_zero_bins=np.array(om_zero_bins)
            #string_zero_bins=np.array(string_zero_bins)
            #om_non_zero_bins=np.array(om_non_zero_bins)
            #string_non_zero_bins=np.array(string_non_zero_bins)

            tanks_nonzerobins.extend(zip(string_non_zero_bins, om_non_zero_bins))
            tanks_zerobins.extend(zip(string_zero_bins, om_zero_bins))

            # rest of these 10 lines are for checking purposes
            #print 'key, len of tanks on non zerobins',key,len(tanks_nonzerobins[key])
            #print 'len of tanks on zerobins',key,len(tanks_zerobins[key])
            #print 'len of bins assigned',len(bins_assigned[key])
            #seen = set()
            #unique=[item for item in bins_assigned[key] if item not in seen and not seen.add(item)]
            #bins_assigned[key] = unique
            #print 'len of unique bins assigned' , len(unique)
            #print '---------------------'
        #print 'no of bins in hevent', len(hevent[hevent!=0])

        return tanks_zerobins, tanks_nonzerobins



    def calc_llh_values_new(self,histbins,histrange,s125,zen,ignorezerobinsinboth,generatepdf,pdffileinstance,trace_tanks=True):
        """ Calculate LLH values for events """

        self.events['proton_like']={}
        self.events['gamma_like']={}
        self.events['unused_proton']={}
        self.events['unused_gamma']={}
        self.events['N_unused_tanks']={}
        tanks_zerobins = {}
        tanks_nonzerobins = {}

        for template_key in ['q_r','q_t','t_r']:
            self.events['proton_like'][template_key]=[]
            self.events['gamma_like'][template_key]=[]
            self.events['unused_proton'][template_key]=[]
            self.events['unused_gamma'][template_key]=[]
            self.events['N_unused_tanks'][template_key]=[]
            tanks_zerobins[template_key]={}
            tanks_nonzerobins[template_key]={}
        self.events['N_unused_tanks']['Common']=[]

        for i in range(len(self.events['laputop_x'])):
            for template_key in ['q_r','q_t','t_r']:
                hevent, xedges, yedges = self.return_hevent(i, histbins, histrange, template_key)

                s125 = str(round(float(s125), 1))
                # print("self.heatmap['proton_norm'][template_key].keys() = {}".format(self.heatmap['proton_norm'][template_key].keys()))
                if s125 == '-0.0':
                    # print("self.heatmap['proton_norm'][template_key].keys() = {}".format(self.heatmap['proton_norm'][template_key].keys()))
                    # print('\ns125 (before) = {}'.format(s125))
                    s125 = '-1.11022302463e-16'
                    # print('s125 (after) = {}\n'.format(s125))
                tempp=hevent* self.heatmap['proton_norm'][template_key][s125][zen]
                tempg=hevent* self.heatmap['gamma_norm'][template_key][s125][zen]

                tanksloc = hevent!=0

                totalbins = float(len(tanksloc[tanksloc]))

                bins_proton = self.heatmap['proton_norm'][template_key][s125][zen].copy()[tanksloc]
                zerobins_proton = len(bins_proton[bins_proton==0])

                bins_gamma = self.heatmap['gamma_norm'][template_key][s125][zen].copy()[tanksloc]
                zerobins_gamma = len(bins_gamma[bins_gamma==0])

                #Find which tanks are landing on zero bins
                if trace_tanks:
                    tanks_zerobins[template_key], tanks_nonzerobins[template_key] = self.tanks_on_zero_bins(
                                        self.heatmap['proton_norm'][template_key][s125][zen].copy(),
                                        self.heatmap['gamma_norm'][template_key][s125][zen].copy(),
                                        hevent, xedges, yedges, template_key, eventno=i
                                        )
                self.events['unused_proton'][template_key].append(zerobins_proton/totalbins)
                self.events['unused_gamma'][template_key].append(zerobins_gamma/totalbins)

                if ignorezerobinsinboth==True:
                    bool = (tempp!=0)&(tempg!=0)
                else:
                    bool = tempp!=0

                tempp=np.sum(np.log10(tempp[bool]))
                tempg=np.sum(np.log10(tempg[bool]))

                self.events['proton_like'][template_key].append(tempp)
                self.events['gamma_like'][template_key].append(tempg)

                if generatepdf==True and random.random() > 0.95 :
                   self.make_pdf(s125,zen,template_key,pdffileinstance,tempp,tempg,xedges,yedges,i,
                                totalbins,zerobins_proton,zerobins_gamma)

            #Count N Tanks landing on zero bins in all 3 PDFs
            a=tanks_zerobins['q_r']
            self.events['N_unused_tanks']['q_r'].append(len(a))
            b=tanks_zerobins['t_r']
            self.events['N_unused_tanks']['t_r'].append(len(b))
            c=tanks_zerobins['q_t']
            self.events['N_unused_tanks']['q_t'].append(len(c))
            d=list(set(a) & set(b) & set(c))
            self.events['N_unused_tanks']['Common'].append(len(d))

        return

    def calc_llh_values_new_3d(self,s125,zen,ignorezerobinsinboth=True):
        """ Calculate LLH values for events """

        template_key='3d'

        for template in ['proton','gamma']:
            self.events[template+'_like'][template_key]=[]
            self.events['unused_'+template][template_key]=[]

            for i in range(len(self.events['laputop_x'])):
                hist={}
                for key in ['nohit','hlc','slc']:
                    r=np.array(self.events[key+'_rperp'][i])
                    q=np.array(self.events[key+'_q'][i])
                    t=np.array(self.events[key+'_t'][i])

                    if len(r)==0:
                        hist[key] = np.zeros(global3dbins)
                        continue

                    data=np.array(zip(r,q,t))
                    hbins = global3dbins
                    hrange = global3drange

                    hist[key], edges3d = np.histogramdd(data, bins = hbins, range=hrange)

                hevent = hist['hlc'] + hist['slc'] + hist['nohit']

                temp=hevent* self.heatmap[template+'_norm'][template_key][s125][zen]

                tanksloc = hevent!=0
                zerobins = self.heatmap[template+'_norm'][template_key][s125][zen].copy()[tanksloc]
                totalbins = float(len(zerobins))
                unusedbins = len(zerobins[zerobins==0])
                self.events['unused_'+template][template_key].append(unusedbins/totalbins)

                if ignorezerobinsinboth==True:
                    bool = (self.heatmap['proton_norm'][template_key][s125][zen]!=0)&(self.heatmap['gamma_norm'][template_key][s125][zen]!=0)&(temp!=0)
                else:
                    bool = temp!=0

                temp=np.log10(temp[bool])
                tempp=np.sum(temp)

                self.events[template+'_like'][template_key].append(tempp)
        return
