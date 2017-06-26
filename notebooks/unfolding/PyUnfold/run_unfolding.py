#!/usr/bin/env python

import os

if __name__ == '__main__':

    priors_list = ['h3a', 'antih3a', 'h4a', 'Hoerandel5', 'antiHoerandel5', 'uniform', 'alllight', 'allheavy']
    for prior in priors_list:
        f = 'unfolded_output_{}.root'.format(prior)
        if os.path.exists(f):
            print('Removing unfolded_output_{}.root...'.format(prior))
            os.remove(f)
        os.system('python Unfold.py -c config_{}.cfg --fluxmodel {}'.format(prior, prior))
