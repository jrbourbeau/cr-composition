#!/usr/bin/env python

from __future__ import division, print_function
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt

import comptools as comp

if __name__ == '__main__':

    description = 'Plots response matrix used in unfolding'
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('-c', '--config', dest='config',
                        default='IC86.2012',
                        choices=comp.simfunctions.get_sim_configs(),
                        help='Detector configuration')
    parser.add_argument('--num_groups', dest='num_groups', type=int,
                        default=4, choices=[2, 3, 4],
                        help='Number of composition groups')
    args = parser.parse_args()

    config = args.config
    num_groups = args.num_groups

    # Load response matrix from disk
    response_file = os.path.join(comp.paths.comp_data_dir, config,
                                 'unfolding',
                                 'response_{}-groups.txt'.format(num_groups))
    response = np.loadtxt(response_file)
    response_err_file = os.path.join(comp.paths.comp_data_dir, config,
                                 'unfolding',
                                 'response_err_{}-groups.txt'.format(num_groups))
    response_err = np.loadtxt(response_err_file)

    # Plot response matrix
    fig, ax = plt.subplots()
    plt.imshow(response, origin='lower', cmap='viridis')
    ax.plot([0, response.shape[0]-1], [0, response.shape[1]-1],
            marker='None', ls=':', color='C1')
    ax.set_xlabel('True bin')
    ax.set_ylabel('Reconstructed bin')
    ax.set_title('Response matrix')
    plt.colorbar(label='$\mathrm{P(E_i|C_{\mu})}$')

    response_plot_outfile = os.path.join(
        comp.paths.figures_dir, 'unfolding', config, 'response_matrix',
        'response-matrix_{}-groups.png'.format(num_groups))
    comp.check_output_dir(response_plot_outfile)
    plt.savefig(response_plot_outfile)

    # Plot response matrix error
    fig, ax = plt.subplots()
    plt.imshow(response_err, origin='lower', cmap='viridis')
    ax.plot([0, response_err.shape[0]-1], [0, response_err.shape[1]-1],
            marker='None', ls=':', color='C1')
    ax.set_xlabel('True bin')
    ax.set_ylabel('Reconstructed bin')
    ax.set_title('Response matrix')
    plt.colorbar(label='$\mathrm{\delta P(E_i|C_{\mu})}$')

    response_plot_outfile = os.path.join(
        comp.paths.figures_dir, 'unfolding', config, 'response_matrix',
        'response_err-matrix_{}-groups.png'.format(num_groups))
    comp.check_output_dir(response_plot_outfile)
    plt.savefig(response_plot_outfile)
