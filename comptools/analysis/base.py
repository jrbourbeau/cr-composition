
from collections import namedtuple
import numpy as np

class DataSet(object):

    def __init__(self, name=None, X=None, y=None, le=None):
        self.name = name
        self.X = X  # features array
        self.y = y  # targets array
        self.le = le # target LabelEncoder, if exists
        # If both y and a le are provided, store the
        # LabelEncoder inverse_transform-ed array as 'labels'
        if all(item is not None for item in [y, le]):
            self.labels = le.inverse_transform(y)

    def __repr__(self):
        output = 'DataSet(name={},\nX={},\ny={})'.format(self.name, self.X, self.y)
        return output

    def __len__(self):
        if self.X is not None:
            return self.X.shape[0]
        else:
            return 0

    def __getitem__(self, sliced):
        sliced_dataset = DataSet(name=self.name, le=self.le)
        for attr, value in self.__dict__.iteritems():
            if attr in ['name', 'le']: continue
            try:
                sliced_value = value[sliced]
                setattr(sliced_dataset, attr, sliced_value)
            except:
                setattr(sliced_dataset, attr, value)

        return sliced_dataset

    def __add__(self, other):
        concat_dataset = DataSet(name=self.name, le=self.le)
        for attr, value in self.__dict__.iteritems():
            if attr in ['name', 'le']: continue
            other_value = getattr(other, attr)
            concat_value = np.concatenate((value, other_value))
            setattr(concat_dataset, attr, concat_value)

        return concat_dataset

    def isnull(self):
        attributes = []
        for attr, value in self.__dict__.iteritems():
            attributes.append(value)
        any_attr_true = any(attributes)
        is_null = not any_attr_true

        return is_null


def get_energybins():

    # Create EnergyBins object to store all realted information
    EnergyBins = namedtuple('EnergyBins', ['energy_min', 'energy_max',
        'energy_bins', 'energy_midpoints', 'energy_bin_widths',
        'log_energy_min', 'log_energy_max', 'log_energy_bin_width',
        'log_energy_bins', 'log_energy_midpoints'])

    # # Define full energy range
    # log_energy_min_full = 5.0
    # log_energy_max_full = 9.5
    # energy_min = 10**log_energy_min_full
    # energy_max = 10**log_energy_max_full
    # # Define energy binning for this analysis
    # log_energy_bin_width = 0.1
    # log_energy_bins_full = np.arange(log_energy_min_full,
    #     log_energy_max_full+log_energy_bin_width, log_energy_bin_width)
    # # log_energy_midpoints_full = (log_energy_bins[1:] + log_energy_bins[:-1]) / 2
    #
    # energy_bins_full = 10**log_energy_bins
    # energy_midpoints_full = 10**log_energy_midpoints
    # energy_bin_widths_full = energy_bins[1:] - energy_bins[:-1]

    # Define energy range for this analysis
    log_energy_min = 6.4
    log_energy_break = 8.0
    log_energy_max = 9.0
    energy_min = 10**log_energy_min
    energy_max = 10**log_energy_max
    # Define energy binning for this analysis
    log_energy_bin_width = 0.1
    # log_energy_small_bins = np.arange(log_energy_min,
    #     log_energy_max+log_energy_bin_width, log_energy_bin_width)
    log_energy_small_bins = np.arange(log_energy_min, log_energy_break, log_energy_bin_width)
    log_energy_large_bins = np.arange(log_energy_break,
        log_energy_max+2*log_energy_bin_width, 2*log_energy_bin_width)
    log_energy_bins = np.append(log_energy_small_bins, log_energy_large_bins)
    log_energy_midpoints = (log_energy_bins[1:] + log_energy_bins[:-1]) / 2

    energy_bins = 10**log_energy_bins
    energy_midpoints = 10**log_energy_midpoints
    energy_bin_widths = energy_bins[1:] - energy_bins[:-1]

    # Create instance of EnergyBins with appropriate binning
    energybins = EnergyBins(energy_min=energy_min,
                            energy_max=energy_max,
                            energy_bins=energy_bins,
                            energy_midpoints=energy_midpoints,
                            energy_bin_widths=energy_bin_widths,
                            log_energy_min=log_energy_min,
                            log_energy_max=log_energy_max,
                            log_energy_bin_width=log_energy_bin_width,
                            log_energy_bins=log_energy_bins,
                            log_energy_midpoints=log_energy_midpoints)

    return energybins


def get_color_dict():
    color_dict = {'light': 'C0', 'heavy': 'C1', 'total': 'C2',
                 'P': 'C0', 'He': 'C1', 'O': 'C3', 'Fe':'C4',
                 'data': 'k'}

    return color_dict

def cast_to_ndarray(input):
    if isinstance(input, np.ndarray):
        return input
    else:
        try:
            return np.array(input)
        except:
            raise TypeError('Input wasn\'t able to be cast to a numpy.ndarray. Got type {}.'.format(type(input)))
