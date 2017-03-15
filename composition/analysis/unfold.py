
from __future__ import division
import numpy as np
from sklearn.metrics import confusion_matrix


class Unfolder(object):

    def __init__(self):
        return

    def unfold(self, true_MC_comp=None, reco_MC_comp=None,
               observed_comp=None, priors=None, labels=None):
        self.reco_MC_comp = reco_MC_comp
        self.true_MC_comp = true_MC_comp
        self.observed_comp = np.array(observed_comp)
        self.labels = labels
        self.priors = np.array(priors)
        self.num_comps = len(labels)

        response_mat = self._get_response_matrix()
        normalized_probs = self._get_normalize_probs(response_mat)
        unfolding_mat = self._get_unfolding_matrix(response_mat, normalized_probs)
        unfolded_events = self._get_unfolded_events(unfolding_mat)

        return unfolded_events

    def _get_response_matrix(self):
        response_mat = confusion_matrix(self.true_MC_comp, self.reco_MC_comp,
            labels=self.labels)
        response_mat = response_mat.T
        # Normalize response matrix to be the P(reco|true) instead of just number of events
        response_mat = response_mat.astype(float)/response_mat.sum(axis=0, keepdims=True)

        return response_mat

    def _get_normalize_probs(self, response_mat):
        normalized_probs = []
        for idx in range(self.num_comps):
            normalized_prob = np.sum([response_mat[idx, k]*self.priors[k] for k in range(self.num_comps)])
            normalized_probs.append(normalized_prob)

        return normalized_probs

    def _get_unfolding_matrix(self, response_mat, normalized_probs):
        unfolding_mat = np.zeros_like(response_mat)
        for row_idx in range(unfolding_mat.shape[0]):
            for col_idx in range(unfolding_mat.shape[1]):
                unfolding_mat[row_idx, col_idx] = response_mat[col_idx, row_idx]*self.priors[row_idx]/normalized_probs[col_idx]

        return unfolding_mat

    def _get_unfolded_events(self, unfolding_mat):

        observed = []
        for label in self.labels:
            label_mask = self.observed_comp == label
            observed.append(label_mask.sum())
            print('{} of label {}'.format(label_mask.sum(), label))

        unfolded_events = []
        for i in range(self.num_comps):
            unfolded_event = np.sum([unfolding_mat[i, k] * observed[k] for k in range(unfolding_mat.shape[1])])
            unfolded_events.append(unfolded_event)

        return unfolded_events
