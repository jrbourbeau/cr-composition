#!/bin/sh


input=(/data/sim/IceTop/2010/filtered/level2a/CORSIKA-ice-top/7241/00000-00999/Level2a_IC79_corsika_icetop.007241.000001.i3.bz2 /data/sim/IceTop/2010/filtered/level2a/CORSIKA-ice-top/7241/00000-00999/Level2a_IC79_corsika_icetop.007241.000002.i3.bz2 /data/sim/IceTop/2010/filtered/level2a/CORSIKA-ice-top/7241/00000-00999/Level2a_IC79_corsika_icetop.007241.000003.i3.bz2 /data/sim/IceTop/2010/filtered/level2a/CORSIKA-ice-top/7241/00000-00999/Level2a_IC79_corsika_icetop.007241.000004.i3.bz2 /data/sim/IceTop/2010/filtered/level2a/CORSIKA-ice-top/7241/00000-00999/Level2a_IC79_corsika_icetop.007241.000005.i3.bz2 /data/sim/IceTop/2010/filtered/level2a/CORSIKA-ice-top/7241/00000-00999/Level2a_IC79_corsika_icetop.007241.000006.i3.bz2 /data/sim/IceTop/2010/filtered/level2a/CORSIKA-ice-top/7241/00000-00999/Level2a_IC79_corsika_icetop.007241.000007.i3.bz2 /data/sim/IceTop/2010/filtered/level2a/CORSIKA-ice-top/7241/00000-00999/Level2a_IC79_corsika_icetop.007241.000008.i3.bz2 /data/sim/IceTop/2010/filtered/level2a/CORSIKA-ice-top/7241/00000-00999/Level2a_IC79_corsika_icetop.007241.000009.i3.bz2 /data/sim/IceTop/2010/filtered/level2a/CORSIKA-ice-top/7241/00000-00999/Level2a_IC79_corsika_icetop.007241.000010.i3.bz2 /data/sim/IceTop/2010/filtered/level2a/CORSIKA-ice-top/7241/00000-00999/Level2a_IC79_corsika_icetop.007241.000011.i3.bz2 /data/sim/IceTop/2010/filtered/level2a/CORSIKA-ice-top/7241/00000-00999/Level2a_IC79_corsika_icetop.007241.000012.i3.bz2 /data/sim/IceTop/2010/filtered/level2a/CORSIKA-ice-top/7241/00000-00999/Level2a_IC79_corsika_icetop.007241.000013.i3.bz2 /data/sim/IceTop/2010/filtered/level2a/CORSIKA-ice-top/7241/00000-00999/Level2a_IC79_corsika_icetop.007241.000014.i3.bz2 /data/sim/IceTop/2010/filtered/level2a/CORSIKA-ice-top/7241/00000-00999/Level2a_IC79_corsika_icetop.007241.000015.i3.bz2 /data/sim/IceTop/2010/filtered/level2a/CORSIKA-ice-top/7241/00000-00999/Level2a_IC79_corsika_icetop.007241.000016.i3.bz2 /data/sim/IceTop/2010/filtered/level2a/CORSIKA-ice-top/7241/00000-00999/Level2a_IC79_corsika_icetop.007241.000017.i3.bz2 /data/sim/IceTop/2010/filtered/level2a/CORSIKA-ice-top/7241/00000-00999/Level2a_IC79_corsika_icetop.007241.000018.i3.bz2 /data/sim/IceTop/2010/filtered/level2a/CORSIKA-ice-top/7241/00000-00999/Level2a_IC79_corsika_icetop.007241.000019.i3.bz2 /data/sim/IceTop/2010/filtered/level2a/CORSIKA-ice-top/7241/00000-00999/Level2a_IC79_corsika_icetop.007241.000020.i3.bz2)
# input=/data/sim/IceTop/2011/filtered/CORSIKA-ice-top/level2/11644/00000-00999/Level2_IC86_corsika_icetop.010042.000828.i3.gz
output=test.i3.gz

python level3_IceTop_InIce.py ${input[*]} \
       --isMC --do-inice --dataset 7241 --det IC79 \
       --waveform -o $output
    #    --L2-gcdfile /data/sim/sim-new/downloads/GCD_31_08_11/GeoCalibDetectorStatus_IC79.55380_L2a.i3.gz \
    #    --L3-gcdfile GeoCalibDetectorStatus_IC79.55380_L2a.i3.gz \
