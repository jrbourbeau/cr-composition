
# level 2 GCD
# GCDL2=/data/sim/IceTop/2011/filtered/CORSIKA-ice-top/level2/12518/00000-00999/GeoCalibDetectorStatus_2012.56063_V1_OctSnow.i3.gz
GCDL2=/data/sim/sim-new/downloads/GCD_31_08_11/GeoCalibDetectorStatus_IC79.55380_L2a.i3.gz
GCDL3=`basename $GCDL2`

# make new GCD
python $I3_BUILD/icetop_Level3_scripts/resources/scripts/MakeL3GCD_MC.py --MCgcd $GCDL2 --output $GCDL3
