CPP=g++

CFLAGS= -O2 -Wall 
ROOTFLAGS= `root-config --cflags` `root-config --libs`

all: ldf_proton

ldf_proton: ldf_proton.cpp milCorsikaInterface.cpp
	$(CPP) -o ldf_proton ldf_proton.cpp milCorsikaInterface.cpp $(CFLAGS) $(ROOTFLAGS)

%.o: %.cpp %.h
	$(CPP) $(CFLAGS) -c $< -o $@ $(ROOTFLAGS) 

.PHONY: clean

clean:
	rm -vf ldf_proton *.oa

