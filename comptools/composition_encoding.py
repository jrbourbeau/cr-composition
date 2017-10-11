
from __future__ import print_function, division
from collections import OrderedDict


_two_group_labels = OrderedDict(PPlus='light', He4Nucleus='light',
                                O16Nucleus='heavy', Fe56Nucleus='heavy')
_three_group_labels = OrderedDict(PPlus='light', He4Nucleus='light',
                                  O16Nucleus='intermediate',
                                  Fe56Nucleus='heavy')


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


_two_group_encoding = OrderedDict(light=0, heavy=1)
_three_group_encoding = OrderedDict(light=0, intermediate=1, heavy=2)
_four_group_encoding = OrderedDict(PPlus=0, He4Nucleus=1, O16Nucleus=2,
                                   Fe56Nucleus=3)


def encode_composition_groups(groups, num_groups=2):
    if num_groups == 2:
        group_to_label = _two_group_encoding
    elif num_groups == 3:
        group_to_label = _three_group_encoding
    elif num_groups == 4:
        group_to_label = _four_group_encoding
    else:
        raise ValueError('Invalid number of groups entered. '
                         'Must be 2, 3, or 4.')

    return [group_to_label[g] for g in groups]


def get_comp_list(num_groups=2):
    if num_groups == 2:
        group_to_label = _two_group_encoding
    elif num_groups == 3:
        group_to_label = _three_group_encoding
    elif num_groups == 4:
        group_to_label = _four_group_encoding

    return list(group_to_label.keys())


comp_to_label_dict = {'light': 0, 'heavy': 1}


def comp_to_label(composition):
    try:
        return comp_to_label_dict[composition]
    except KeyError:
        raise KeyError('Incorrect composition ({}) entered'.format(composition))

    
def label_to_comp(label):
    label_to_comp_dict = {value: key for key, value in comp_to_label_dict.iteritems()}
    try:
        return label_to_comp_dict[label]
    except KeyError:
        raise KeyError('Incorrect label ({}) entered'.format(label))
