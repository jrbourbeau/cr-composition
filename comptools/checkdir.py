#!/usr/bin/env python

import sys
import os

def checkdir(outfile):

    outdir, basename = os.path.split(outfile)
    if not os.path.exists(outdir):
        print('\nThe path {} doesn\'t exist. Creating it...\n'.format(outfile))
        os.makedirs(outdir)

    return
