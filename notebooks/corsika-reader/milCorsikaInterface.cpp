#include "milCorsikaInterface.hh"
#include <string.h>
#include <stdlib.h>


milCorsikaInterface* milCorsikaInterface::theCorsikaFile=NULL;

///////////////////////////////////////////////////////////////////////////////////////////////////////

milCorsikaInterface::milCorsikaInterface(int nsub,int lsub) {
 NumberOfSubblock = nsub;
 LengthOfSubblock = lsub;
 RecordLength = NumberOfSubblock * LengthOfSubblock;
 BuffLength = RecordLength*sizeof(float);
 databuff = new float[RecordLength];
}
 
///////////////////////////////////////////////////////////////////////////////////////////////////////
void milCorsikaInterface::OpenInputFile(const string inputfile){

 inputFile = gzopen(inputfile.c_str(),"r");
 if(!inputFile) {
    cout<<" <milCorsikaInterface>: Can not open file: "<<inputfile<<endl;
    exit(1);
 }
 BuffRead = 0;
 BuffPos  = 0;
 if(GetSubblock() == 0)    {
   if(RecordType(buff[0]) == 1)  FillRunHeader(buff);
   else {
      cout<< " <milCorsikaInterface>: data Error: no run header block!"<<endl;
//      exit(1);
   }
 }
 else    { 
   cout<<" <milCorsikaInterface>: data Error: no data read!"<<endl;
//   exit(1);
 }
}


///////////////////////////////////////////////////////////////////////////////////////////////////////

milCorsikaInterface::~milCorsikaInterface() {
 delete [] databuff; 
 if(inputFile) gzclose(inputFile);
}
 
///////////////////////////////////////////////////////////////////////////////////////////////////////

int milCorsikaInterface::ReadBuff() {

 // read in one record from a disk file. 
 // Note: the data file was created by a Fortran program. For each record,
 // there are 4 byte at the begining and end of the record to mark the
 // begining and end of the record. We have to skip these data.
 //
     float temp;
 
     if(gzeof(inputFile) != 0) return 0;
     gzread(inputFile,(char *)&temp,sizeof(float)); // skip the first 4 byte
     gzread(inputFile,(char *)databuff,BuffLength);
     gzread(inputFile,(char *)&temp,sizeof(float));//  skip the last 4 byte
     BuffRead = 1;
     BuffPos = 0;
     return 1;
}   

/////////////////////////////////////////////////////////////////////////////////////////////////////// 

int milCorsikaInterface::GetSubblock() {

 // check if there are data in the data buff
    if(!BuffRead || BuffPos >= RecordLength) {
      if(ReadBuff() <= 0 ) {
        cout<< "<milCorsikaInterface::GetSubblock> No Data left!"<<endl;
        return 1;
      }
    }
    buff = &databuff[BuffPos];
    BuffPos += LengthOfSubblock;
    return 0;
 }

///////////////////////////////////////////////////////////////////////////////////////////////////////    

int milCorsikaInterface::RecordType(float f)
 {
     union {
        float f;
        char  c[4];
     } temp;
     char str[5];
     temp.f = f;
     strncpy(str,temp.c,4);
     str[4] = 0;
     if(!strcmp(str,"RUNH"))return 1;
     if(!strcmp(str,"EVTH"))return 2;
     if(!strcmp(str,"EVTE"))return 3;
     if(!strcmp(str,"RUNE"))return 4;
     return 0;
 }
 
///////////////////////////////////////////////////////////////////////////////////////////////////////

int milCorsikaInterface::NextEvent()
 {
   CPlist.clear();

   if(GetSubblock() != 0) return 1;

 //check if we reach end of run
   if(RecordType(buff[0]) == 4) {
    FillRunEnd(buff);
    PrintRunEnd();
     return 4;
   }

 // the first block shuld be the event header
   if(RecordType(buff[0]) == 2) {
               FillEventHeader(buff);
   } else {  
     cout<< "<milCorsikaInterface::ReadOneEvent>data Error: no Event header block!"<<endl;
//     exit(3);
   }

 // following subblocks are the particle information until an end of event subblock
   if(GetSubblock() != 0) return 1;
   while(RecordType(buff[0]) != 3) {
        FillEvent(buff);
     if(GetSubblock() != 0) return 1;
   } 

 // end of event block should be reached here
   if(RecordType(buff[0]) == 3) FillEventEnd(buff);
   else {
     cout<<" <milCorsikaInterface::ReadOneEvent>Data Error: no end of event block!"<<endl;
//     exit(3);
   }
 
   return 0;
 }
 
///////////////////////////////////////////////////////////////////////////////////////////////////////

void milCorsikaInterface::FillEvent(float *buff)
 {
   int i = 0;
   while(buff[7 * i] != 0.0 && i < 39) {
     cp.id = (int)   buff[7*i+0];     
     cp.px = (double)buff[7*i+1]; 
     cp.py = (double)buff[7*i+2]; 
     cp.pz = (double)buff[7*i+3]; 
     cp.x0 = (double)buff[7*i+4];  
     cp.y0 = (double)buff[7*i+5];
     cp.t0 = (double)buff[7*i+6]; //time since first interaction
     i++;
    CPlist.push_back(cp);
   }
 
 }
 
/////////////////////////////////////////////////////////////////////////////////////////////////////// 

void milCorsikaInterface::FillRunHeader(float *buff)  {   for(int i = 0; i < LengthOfSubblock; i++) fRunHeader[i] = buff[i]; }

void milCorsikaInterface::FillRunEnd(float *buff)     {   for(int i = 0; i < LengthOfSubblock; i++) fRunEnd[i] = buff[i]; }

///////////////////////////////////////////////////////////////////////////////////////////////////////
void milCorsikaInterface::FillEventHeader(float *buff)
{ 
      for(int i = 0; i < LengthOfSubblock; i++) fEventHeader[i] = buff[i];  
/*
//Assign some units and do some other stuff..
      fEventHeader[3]*=GeV;  //Total Energy
      fEventHeader[6]*=cm;   //Height of first interaction in cm
      fEventHeader[7]*=GeV;  //Primary px
      fEventHeader[8]*=GeV;  //Primary py
      fEventHeader[9]*=GeV;  //Primary pz
*/
}

///////////////////////////////////////////////////////////////////////////////////////////////////////

void milCorsikaInterface::FillEventEnd(float *buff)   { for(int i = 0; i < LengthOfSubblock; i++) fEventEnd[i] = buff[i]; }

///////////////////////////////////////////////////////////////////////////////////////////////////////
 
void milCorsikaInterface::PrintRunHeader()
{

//    FILE* opf=fopen(filename,"a");
//    fclose(opf);

 }

/////////////////////////////////////////////////////////////////////////////////////////////////////// 

void milCorsikaInterface::PrintRunEnd()
{
    // cout<<"====== End Of Run #"<<(int)fRunEnd[1]<<" ======"<<endl;
    // cout<<"  Number Of Event Processed: "<<(int)fRunEnd[2]<<endl;
}
 
