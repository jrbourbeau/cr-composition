
from __future__ import division
import numpy as np

import comptools as comp

def get_random_subsample(dataset, size=1000, frac=0.5, composition=None):

    # Get composition mask, if necessary
    if composition is not None:
        # encoded_comp = dataset.le.transform([composition])[0]
        # comp_mask = dataset.y == encoded_comp
        comp_mask = dataset.labels == composition
    else:
        comp_mask = np.array([True]*len(dataset))

    dataset = dataset[comp_mask]

    # Assuming fraction of total_events events
    subsample_size = int(frac*size)
    mask = np.random.choice(len(dataset), subsample_size, replace=False)

    return dataset[mask]
