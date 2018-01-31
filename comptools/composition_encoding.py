
from __future__ import print_function, division
from collections import OrderedDict
import numpy as np


_two_group_labels = OrderedDict()
_two_group_labels['PPlus'] = 'light'
_two_group_labels['He4Nucleus'] = 'light'
_two_group_labels['O16Nucleus'] = 'heavy'
_two_group_labels['Fe56Nucleus'] = 'heavy'

_three_group_labels = OrderedDict()
_three_group_labels['PPlus'] = 'light'
_three_group_labels['He4Nucleus'] = 'light'
_three_group_labels['O16Nucleus'] = 'intermediate'
_three_group_labels['Fe56Nucleus'] = 'heavy'


def composition_group_labels(compositions, num_groups=2):
    if num_groups == 2:
        comp_to_group = _two_group_labels
    elif num_groups == 3:
        comp_to_group = _three_group_labels
    elif num_groups == 4:
        return compositions
    else:
        raise ValueError('Invalid number of groups entered. '
                         'Must be 2, 3, or 4.')

    return [comp_to_group[c] for c in compositions]


_two_group_encoding = OrderedDict()
_two_group_encoding['light'] = 0
_two_group_encoding['heavy'] = 1

_three_group_encoding = OrderedDict()
_three_group_encoding['light'] = 0
_three_group_encoding['intermediate'] = 1
_three_group_encoding['heavy'] = 2

_four_group_encoding = OrderedDict()
_four_group_encoding['PPlus'] = 0
_four_group_encoding['He4Nucleus'] = 1
_four_group_encoding['O16Nucleus'] = 2
_four_group_encoding['Fe56Nucleus'] = 3


def _get_group_encoding_dict(num_groups=2):
    if num_groups == 2:
        group_to_label = _two_group_encoding
    elif num_groups == 3:
        group_to_label = _three_group_encoding
    elif num_groups == 4:
        group_to_label = _four_group_encoding
    else:
        raise ValueError('Invalid number of groups entered. '
                         'Must be 2, 3, or 4.')

    return group_to_label


def encode_composition_groups(groups, num_groups=2):
    group_to_label = _get_group_encoding_dict(num_groups=num_groups)
    return [group_to_label[g] for g in groups]


def decode_composition_groups(labels, num_groups=2):
    group_to_label = _get_group_encoding_dict(num_groups=num_groups)
    label_to_group = {value: key for key, value in group_to_label.items()}
    try:
        groups = np.empty_like(labels, dtype=object)
        for idx, label in enumerate(labels):
            groups[idx] = label_to_group[label]
        return groups
        # return np.array([label_to_group[l] for l in labels], dtype=str)
    except KeyError:
        raise KeyError('Incorrect label entered')


def get_comp_list(num_groups=2):
    group_to_label = _get_group_encoding_dict(num_groups=num_groups)
    return list(group_to_label.keys())


comp_to_label_dict = {'light': 0, 'heavy': 1}


def comp_to_label(composition):
    try:
        return comp_to_label_dict[composition]
    except KeyError:
        raise KeyError('Incorrect composition ({}) entered'.format(composition))


def label_to_comp(label):
    label_to_comp_dict = {value: key for key, value in comp_to_label_dict.items()}
    try:
        return label_to_comp_dict[label]
    except KeyError:
        raise KeyError('Incorrect label ({}) entered'.format(label))
