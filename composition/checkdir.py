#!/usr/bin/env python

import sys
import os

def checkdir(outfile):
    outdir = os.path.dirname(outfile)
    if outdir == '':
        outdir = os.getcwd()
    if not os.path.isdir(outdir):
        print('\nThe directory {} doesn\'t exist...'.format(outdir)\
            + 'creating it...\n')
        os.makedirs(outdir)
    return
