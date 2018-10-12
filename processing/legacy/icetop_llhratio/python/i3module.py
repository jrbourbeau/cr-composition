

# -*- coding: utf-8 -*-
#
## copyright  (C) 2018
# The Icecube Collaboration
#
# $Id$
#
# @version $Revision$
# @date $LastChangedDate$
# @author Hershal Pandya <hershal@udel.edu> Last changed by: $LastChangedBy$
#
from icecube import phys_services, dataclasses, icetray, recclasses
import numpy as np
import tables
from icecube.icetray.i3logging import log_fatal,log_warn
from llh_ratio_nd import get_slice_vector,log_likelihood_ratio

def signed_log(t):
     return np.sign(t)*np.log10(np.absolute(t)+1)

def log_plus_one(t):
    return np.log10(t+1)

def check_distinct_regions_add_up_to_full(distinct_regions_binedges,binedges,decimals=2):
    combine_edges=[]
    for i in range(len(distinct_regions_binedges)):
        for j in range(len(distinct_regions_binedges[i])):
            if i==0:
                combine_edges.append(distinct_regions_binedges[i][j])
            else:
                combine_edges[j]=np.unique(np.sort(np.concatenate((combine_edges[j],distinct_regions_binedges[i][j]))))

    for i in range(len(binedges)):
        are_equal=(np.round(binedges[i],decimals=decimals)==np.round(combine_edges[i],decimals=decimals)).all()
        if not are_equal:
            print 'DistinctRegionsBinEdges do not add up to binedges for this dimension'
            print combine_edges[i], binedges[i]
            raise Exception('Inconsistency found')

    for i in range(len(distinct_regions_binedges)):
        for j in range(len(distinct_regions_binedges[0])):
            if i<len(distinct_regions_binedges)-1:
                next_one=i+1
            else:
                next_one=0
            intersection=np.intersect1d(np.round(distinct_regions_binedges[i][j],decimals=decimals),np.round(distinct_regions_binedges[next_one][j],decimals=decimals))
            if len(intersection)>1 and len(intersection)!=len(binedges[j]):
                print 'comparing "Distinct" regions %i and %i, dimension %i'%(i,next_one,j)
                print 'These regions Intersect'
                print 'binedges of region1',distinct_regions_binedges[i]
                print 'binedges of region2',distinct_regions_binedges[next_one]
                raise Exception('Inconsistency found')
    return

class IceTop_LLHRatio(icetray.I3ConditionalModule):
    """
    Input takes I3VectorShieldHitRecords with following members:
    distance
    residual_time
    charge
    """

    def __init__(self,ctx):
        icetray.I3ConditionalModule.__init__(self, ctx)

        #common inputs
        self.AddParameter('Hits_I3VectorShieldHitRecord',
                          'Shield applied to Pulses Using a reco',
                          None)
        self.AddParameter('Unhits_I3VectorShieldHitRecord',
                          'Unhits from Shield and Charge/Time assigned false values',
                          None)
        self.AddParameter('Excluded_I3VectorShieldHitRecord',
                          'Containing Dist of Excluded Tanks and Charge/time assigned false values',
                          None)
        self.AddParameter('AngularReco_I3Particle',
                          'I3Particle from which cosZenith is to be drawn',
                          None)
        self.AddParameter('EnergyReco_I3Particle',
                          'I3Particle from which logEnergy is to be drawn',
                          None)
        self.AddParameter('LaputopParamsName',
                          'LaputopParams from which logS125 is to be drawn only accepted if EnergyReco_I3Particle not provided',
                          None)
        self.AddParameter('RunMode','Options: GeneratePDF / CalcLLHR',None)
        self.AddParameter('Output','Name of the output container','IceTopLLHR')

        # inputs for RunMode CalcLLHR
        self.AddParameter('OutputFileName','',None)
        self.AddParameter('BinEdges5D','[logE_edges, cosZen_edges, logQ_edges, signed_logT_edges, logRplusone_edges]',[])
        self.AddParameter('DistinctRegionsBinEdges3D',
                          'Disjoint Regions in Q, T, R PDF. e.g.Unhits/Excluded. [3dEdges1,3dEdges2,..]',
                          [])

        # inputs for RunMode GeneratePDF
        self.AddParameter('SigPDFInputFileName',
                          'Path to input file (Sig) made using GeneratePDF method in the previous run',None)
        self.AddParameter('BkgPDFInputFileName',
                          'Path to input file (Bkg) made using GeneratePDF method in the previous run',None)
        self.AddParameter('DecimalsForSanityCheck',
                          'Consistency checks will compare values rounded to these N decimals.Default:2',2)
        self.AddParameter('SubtractEventFromPDF',
                          'subtract the event from the PDF if it was used for generating the PDF. Default:None',None)
        return

    def Configure(self):
        self.HitsName = self.GetParameter('Hits_I3VectorShieldHitRecord')
        self.UnhitsName = self.GetParameter('Unhits_I3VectorShieldHitRecord')
        self.ExcludedName = self.GetParameter('Excluded_I3VectorShieldHitRecord')
        self.AngularRecoName = self.GetParameter('AngularReco_I3Particle')
        self.EnergyRecoName = self.GetParameter('EnergyReco_I3Particle')
        self.LaputopParamsName = self.GetParameter('LaputopParamsName')
        self.RunMode = self.GetParameter('RunMode')
        self.Decimals= self.GetParameter('DecimalsForSanityCheck')

        if self.RunMode=='GeneratePDF':
            self.OutputName = self.GetParameter('OutputFileName')
            self.binedges = self.GetParameter('BinEdges5D')
            self.distinct_regions_binedges = self.GetParameter('DistinctRegionsBinEdges3D')

            # make sure distinct regions binedges make sense
            if len(self.distinct_regions_binedges)==0:
                # give the whole region as a distinct single region
                self.distinct_regions_binedges = [self.binedges[2:]]
            else:
                #check that each distinct region binedge is same shape as self.binedges i.e.
                for i in self.distinct_regions_binedges:
                    if np.shape(i)!=np.shape(self.binedges[2:]):
                        print 'shape of self.binedges[2:] :',np.shape(self.binedges[2:])
                        print 'shape of self.distinct_regions_binedges',np.shape(self.distinct_regions_binedges)
                        log_fatal('DistinctRegionBinEdges and BinEdges* not compatible')

                #check that joining all distinct regions gives total binedges
                check_distinct_regions_add_up_to_full(self.distinct_regions_binedges, self.binedges[2:],decimals=self.Decimals)

            self.labels = ['logE', 'cosZ', 'logQ', 'signedlogT', 'logRplusOne']

            #creates the self.hist
            self._init_hist()
        elif self.RunMode=='CalcLLHR':
            self.SigPDFInputName = self.GetParameter('SigPDFInputFileName')
            self.BkgPDFInputName = self.GetParameter('BkgPDFInputFileName')
            # this one should create self.bkg_hist, self.sig_hist, self.binedges, self.labels, self.distinct_regions_binedges
            self._load_PDF_from_file()
            self.SubtractEventFromPDF= self.GetParameter('SubtractEventFromPDF')
            self.objname = self.GetParameter('Output')

        return

    def Physics(self,frame):
        if self.RunMode=='GeneratePDF':
            self._GenPDFsPhysics(frame)
        elif self.RunMode=='CalcLLHR':
            self._CalcLLHRPhysics(frame)
        else:
            log_fatal('RunMode can only accept one these two inputs: GeneratePDF / CalcLLHR')
        self.PushFrame(frame)
        return

    def Finish(self):
        if self.RunMode=='GeneratePDF':
            # generate the outputfile. save histogram.
            f=tables.open_file(self.OutputName,'w')
            f.create_carray('/', 'hist', obj=self.hist,filters=tables.Filters(complib='blosc:lz4hc', complevel=1))

            for i in range(len(self.binedges)):
                f.create_carray('/', 'binedges_%i'%i,
                                obj=self.binedges[i],
                                filters=tables.Filters(complib='blosc:lz4hc',
                                complevel=1))

            for i in range(len(self.distinct_regions_binedges)):
                for j in range(len(self.distinct_regions_binedges[0])):
                    f.create_carray('/', 'region_%i_binedges_%i'%(i,j),
                                    obj=self.distinct_regions_binedges[i][j],
                                    filters=tables.Filters(complib='blosc:lz4hc',
                                    complevel=1))

            f.create_carray('/', 'labels', obj=self.labels,filters=tables.Filters(complib='blosc:lz4hc', complevel=1))
            f.create_carray('/', 'n_events', obj=[self.n_events],filters=tables.Filters(complib='blosc:lz4hc', complevel=1))
            f.close()

        return

    def _load_PDF_from_file(self):
        '''
        this part is hard wired for 5 dimensional PDFs
        '''

        f=tables.open_file(self.SigPDFInputName,'r')
        self.sig_hist = f.root.hist[:]
        self.binedges = [ f.root.binedges_0[:], f.root.binedges_1[:], f.root.binedges_2[:],  f.root.binedges_3[:] , f.root.binedges_4[:]]
        self.distinct_regions_binedges = [ ]
        for r in range(1):
            region_binedges=[]
            for i in range(3):
                temp=eval('f.root.region_%i_binedges_%i[:]'%(r,i))
                region_binedges.append(temp)
            self.distinct_regions_binedges.append(region_binedges)
        self.labels = f.root.labels[:]
        f.close()

        f=tables.open_file(self.BkgPDFInputName,'r')
        self.bkg_hist = f.root.hist[:]
        binedges = [ f.root.binedges_0[:], f.root.binedges_1[:], f.root.binedges_2[:],  f.root.binedges_3[:] , f.root.binedges_4[:]]
        labels = f.root.labels[:]
        f.close()

        if np.shape(self.sig_hist)!=np.shape(self.bkg_hist):
            print 'sig hist, bkg hist shapes dont match'
            print 'sig hist shape',np.shape(sig_hist)
            print 'bkg hist shape',np.shape(bkg_hist)
            raise Exception('Inconsistency found')

        for i in range(len(binedges)):
            are_equal=(np.round(binedges[i],decimals=self.Decimals)==np.round(self.binedges[i],decimals=self.Decimals)).all()
            if not are_equal:
                print 'sig binedges dim %i'%i, self.binedges[i]
                print 'bkg binedges dim %i'%i, binedges[i]
                raise Exception('Sig and Bkg binedges are not equal')

        if (labels!=self.labels).any():
            print 'labels for sig and bkg are not same'
            print 'are you sure you are loading correct sig/bkg pdfs?'

        return

    def _init_hist(self):
        histogram_shape= np.array([len(i)-1 for i in self.binedges])
        self.hist=np.zeros(histogram_shape)
        self.n_events=0
        return

    def _fill(self,sample):
        h,edges=np.histogramdd(sample,self.binedges)

        if np.shape(h)!=np.shape(self.hist):
            log_fatal('initialized histogram and fill histogram dont match in shape')

        self.hist+= h
        self.n_events+=1
        return

    def _GenPDFsPhysics(self,frame):
        in_array=self._create_in_array(frame)
        self._fill(in_array)
        return

    def _CalcLLHRPhysics(self,frame):

        d={}
        d['llh_ratio']= 0.
        d['n_extrapolations_sig_PDF'] = 0.
        d['n_extrapolations_bkg_PDF'] = 0.
        d['llh_sig'] = 0.
        d['llh_bkg'] = 0.
        d['isGood'] = 0.

        # load event information
        in_array = self._create_in_array(frame)
        logE=in_array[0][0]
        coszen=in_array[0][1]

        # select Q, T, R dimensions, generate event histogram
        in_array = (in_array.T[2:]).T
        binedges = self.binedges[2:]
        event_hist,temp = np.histogramdd(in_array, binedges)

        # check if event logE and coszen lies within range of binedges
        if logE>self.binedges[0][-1] or logE<self.binedges[0][0]:
            frame.Put(self.objname,dataclasses.I3MapStringDouble(d))
            return
        if coszen>self.binedges[1][-1] or coszen<self.binedges[1][0]:
            frame.Put(self.objname,dataclasses.I3MapStringDouble(d))
            return

        # find the logE and coszen bins select those bins in sig/bkg pdfs
        logEbincenters = np.array((self.binedges[0][1:] + self.binedges[0][:-1] )/2.)
        coszenbincenters = np.array((self.binedges[1][1:] + self.binedges[1][:-1] )/2.)

        dE = np.absolute(logEbincenters - logE)
        Ebin=np.where(np.amin(dE)==dE)[0][0]

        dcZ = np.absolute(coszenbincenters - coszen)
        cZbin = np.where(np.amin(dcZ)==dcZ)[0][0]

        sig_hist = self.sig_hist[Ebin][cZbin]
        bkg_hist = self.bkg_hist[Ebin][cZbin]
        # subtract the event from the PDF if it was used for generating the PDF
        if self.SubtractEventFromPDF:
            if self.SubtractEventFromPDF=='Sig':
                sig_hist = sig_hist - event_hist
                if (sig_hist<0).any():
                    log_fatal('Event subtraction led to negative values')

            if self.SubtractEventFromPDF=='Bkg':
                bkg_hist = bkg_hist - event_hist
                if (bkg_hist<0).any():
                    log_fatal('Event subtraction led to negative values')

        # normalize histogram, obtain PDFs
        sig_pdf = sig_hist/ np.sum(sig_hist)
        bkg_pdf = bkg_hist/ np.sum(bkg_hist)

        # calculate llh ratio for each region separately and add it up
        # separate calculation is done to avoid one region influencing
        # extrapolated values of empty pixels in the PDF in another region

        llh_map_sig=np.zeros_like(sig_hist)
        llh_map_bkg=np.zeros_like(bkg_hist)
        d['isGood']=1.
        for region_edges in self.distinct_regions_binedges:
            # obtain slice vector for the region of the PDF
            region_range = [ [i[0],i[-1]] for i in region_edges]
            slice_vector= get_slice_vector(binedges,region_range)
            temp = log_likelihood_ratio(heatmap1=sig_pdf[slice_vector],
                                        heatmap2=bkg_pdf[slice_vector],
                                        event_hist = event_hist[slice_vector])

            d['llh_ratio'] += temp[0]
            # all the rest are debugging variables. some will be stored in I3VectorMap.
            # not storing any histograms as output. Just numbers.
            d['n_extrapolations_sig_PDF'] += temp[1]
            d['n_extrapolations_bkg_PDF'] += temp[2]
            d['llh_sig'] += temp[5]
            d['llh_bkg'] += temp[6]

            extrapolated_sig_PDF = temp[3]
            extrapolated_bkg_PDF = temp[4]
            llh_map_sig[slice_vector]=temp[7]
            llh_map_bkg[slice_vector]=temp[8]

        frame.Put(self.objname,dataclasses.I3MapStringDouble(d))
        return

    def _create_in_array(self,frame):

        if self.EnergyRecoName:
            En = np.log10(frame[self.EnergyRecoName].energy)
        elif self.LaputopParamsName:
            En = frame[self.LaputopParamsName].value(recclasses.LaputopParameter.Log10_S125)
            # En = np.log10(frame[self.LaputopParamsName].s125)
        else:
            log_fatal('One of EnergyRecoName_I3Particle or LaputopParamsName needs to be given')

        ze = np.cos(frame[self.AngularRecoName].dir.zenith)

        hits = frame[self.HitsName]
        unhits = frame[self.UnhitsName]
        excluded = frame[self.ExcludedName]

        #hits_t, hits_q, hits_r = np.array([[signed_log(hit.time_residual),np.log10(hit.charge), log_plus_one(hit.distance)] for hit in hits]).T

        hits_t = signed_log(np.array([hit.time_residual for hit in hits]))
        hits_q = np.log10(np.array([hit.charge for hit in hits]))
        hits_r = log_plus_one(np.array([hit.distance for hit in hits]))
        hits_E = np.ones_like(hits_r)*En
        hits_z = np.ones_like(hits_r)*ze


        #unhits_t, unhits_q, unhits_r = np.array([[signed_log(hit.time_residual),np.log10(hit.charge), log_plus_one(hit.distance)] for hit in unhits]).T
        unhits_t = signed_log(np.array([hit.time_residual for hit in unhits]))
        unhits_q = np.log10(np.array([hit.charge for hit in unhits]))
        unhits_r = log_plus_one(np.array([hit.distance for hit in unhits]))
        unhits_E = np.ones_like(unhits_r)*En
        unhits_z = np.ones_like(unhits_r)*ze

        #excluded_t, excluded_q, excluded_r = np.array([[signed_log(hit.time_residual),np.log10(hit.charge), log_plus_one(hit.distance)] for hit in excluded]).T
        excluded_t = signed_log(np.array([hit.time_residual for hit in excluded]))
        excluded_q = np.log10(np.array([hit.charge for hit in excluded]))
        excluded_r = log_plus_one(np.array([hit.distance for hit in excluded]))
        excluded_E = np.ones_like(excluded_r)*En
        excluded_z = np.ones_like(excluded_r)*ze

        # ready data for entry to 5D  hist
        t = np.concatenate( (hits_t, unhits_t, excluded_t) )
        q = np.concatenate( (hits_q, unhits_q, excluded_q) )
        r = np.concatenate( (hits_r, unhits_r, excluded_r) )
        E = np.concatenate( (hits_E, unhits_E, excluded_E) )
        z = np.concatenate( (hits_z, unhits_z, excluded_z) )

        if len(t)!=162 or len(q)!=162 or len(r)!=162:
            print 'N_t %s N_q %s N_r %s'%(len(t),len(q),len(r))
            log_fatal('Total Tanks in Event not 162')

        if np.isnan(t).any() or np.isnan(q).any() or np.isnan(r).any():
            print 't',t
            print 'q',q
            print 'r',r
            log_warn('signed_time/logq/logr have nans')

        in_array=np.vstack([E,z,q,t,r]).T

        return in_array
