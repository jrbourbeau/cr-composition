#ifndef __milCorsikaInterface_hh__
#define __milCorsikaInterface_hh__

#include <fstream>
#include <vector>
#include <stdio.h>
#include <string>
#include <iostream>
#include <zlib.h>

using namespace std;

///Holds details about one particle from the corsika shower
typedef struct {
    int      id;  ///<  corsika particle ID
    double px;  ///< px in GeV
    double py;  ///< py in GeV
    double pz;  ///< pz in GeV
    double t0;  ///< time in ns
    double x0;  ///<X position in cm 
    double y0;  ///<Y position in cm
} CParticle;

///Gets information from Corsika file and holds some other nice variables                     
class milCorsikaInterface
{

private:
  milCorsikaInterface(int nsub=21, int lsub=273);
  static milCorsikaInterface * theCorsikaFile;
  ~milCorsikaInterface();  

public:
  static inline milCorsikaInterface & GetCorsikaFile();
  static inline milCorsikaInterface * GetCorsikaFilePtr();

    void OpenInputFile(const string filename);
    int NextEvent();

    void PrintRunHeader();
    void PrintRunEnd();
                                                                                                                                                                                                     
    double *GetRunHeader()   { return fRunHeader;} 
    double *GetRunEnd()      { return fRunEnd;}
    double *GetEventHeader() { return fEventHeader;}
    double *GetEventEnd()    { return fEventEnd;}
    inline void SetOutputFile(string f) {filename=f;}  ///<Stores the output file location
    inline string GetOutputFile() {return filename;}   ///<Returns the output file location

    typedef std::vector<CParticle> CParticleList;   
    typedef CParticleList::const_iterator CIterator; 

    CParticleList GetCPlist() {return CPlist;} ///<Gets the information for all shower particles reaching the ground
    

  private:
 
    CParticleList CPlist;    ///<Hold information of all shower particles from corsika file                                                                                                                                                     
    int RecordType(float f); ///< Check the data type acoording to the first word of the sub-block.
      // Return value:
       //   1: run header
       //   2: event header
       //   3: end of event
       //   4: end of run
    int ReadBuff();       ///< Read one record from disk file
    int GetSubblock();   ///< get one sub block from the data buff
                                                                                                                                                                                                     
    void FillRunHeader(float *buff);
    void FillRunEnd(float *buff);
    void FillEventHeader(float *buff);
    void FillEventEnd(float *buff);
                                                                                                                                                                                                     
    void FillEvent(float *buff); ///< stores the particle information from one sub-block

                                                                                                                                                                                                     
//    std::ifstream inputFile;      ///< input stream
    gzFile inputFile;
                                                                                                                                                                                                     
    int   NumberOfSubblock;  ///< number of sub-blocks in a record (21)
    int   LengthOfSubblock;  ///< length of a sub-block (27words)
    int   RecordLength;      ///< length of the record in words
    int   BuffLength;        ///< data buffer length in bytes
    int   BuffPos;           ///< current position in data buff
    float* databuff;         ///< data buff to store one record
    float *buff;             ///< a pointer to a sub-block in data buff
    int   BuffRead;          ///< flag if a record in read
                                                                                                                                                                                                     
    double fRunHeader[273];   ///< run header information
    double fRunEnd[273];      ///< end of run information
    double fEventHeader[273]; ///< event header block
    double fEventEnd[273];    ///< event end block
    CParticle cp; 
    string filename;

};


milCorsikaInterface *milCorsikaInterface::GetCorsikaFilePtr()
{
  if (theCorsikaFile == NULL)
    theCorsikaFile= new milCorsikaInterface();
  return theCorsikaFile;
}

milCorsikaInterface &milCorsikaInterface::GetCorsikaFile()
{
  if (theCorsikaFile == NULL)
    theCorsikaFile= new milCorsikaInterface();
  return *theCorsikaFile;
}


#endif

