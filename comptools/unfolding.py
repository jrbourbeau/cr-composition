
import numpy as np

from .composition_encoding import get_comp_list


def unfolded_counts_dist(unfolding_df, iteration=-1, num_groups=4):
    """
    Convert unfolded distributions DataFrame from PyUnfold counts arrays
    to a dictionary containing a counts array for each composition.

    Parameters
    ----------
    unfolding_df : pandas.DataFrame
        Unfolding DataFrame returned from PyUnfold.
    iteration : int, optional
        Specific unfolding iteration to retrieve unfolded counts
        (default is -1, the last iteration).
    num_groups : int, optional
        Number of composition groups (default is 4).

    Returns
    -------
    counts : dict
        Dictionary with composition-counts key-value pairs.
    counts_sys_err : dict
        Dictionary with composition-systematic error key-value pairs.
    counts_stat_err : dict
        Dictionary with composition-statistical error key-value pairs.
    """
    comp_list = get_comp_list(num_groups=num_groups)

    df_iter = unfolding_df.iloc[iteration]

    counts, counts_sys_err, counts_stat_err = {}, {}, {}
    for idx, composition in enumerate(comp_list):
        counts[composition] = df_iter['n_c'][idx::num_groups]
        counts_sys_err[composition] = df_iter['sys_err'][idx::num_groups]
        counts_stat_err[composition] = df_iter['stat_err'][idx::num_groups]

    counts['total'] = np.sum([counts[composition] for composition in comp_list], axis=0)
    counts_sys_err['total'] = np.sqrt(np.sum([counts_sys_err[composition]**2 for composition in comp_list], axis=0))
    counts_stat_err['total'] = np.sqrt(np.sum([counts_stat_err[composition]**2 for composition in comp_list], axis=0))

    return counts, counts_sys_err, counts_stat_err
