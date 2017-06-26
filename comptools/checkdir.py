#!/usr/bin/env python

import os

def check_output_dir(outfile, makedirs=True):

    outdir, basename = os.path.split(outfile)

    if outdir == '':
        return
    elif not os.path.exists(outdir):
        if makedirs:
            print('The directory {} doesn\'t exist. '
                  'Creating it...'.format(outdir))
            os.makedirs(outdir)
        else:
            raise IOError('The directory {} doesn\'t exist'.format(outdir))

    return
