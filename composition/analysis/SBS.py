#!/usr/bin/env python3

from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.base import clone
from itertools import combinations
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

class SBS():
    def __init__(self, estimator, k_features, scoring=accuracy_score,
                 test_size=0.25, random_state=1):
        self.scoring = scoring
        self.estimator = clone(estimator)
        self.k_features = k_features
        self.test_size = test_size
        self.random_state = random_state

    def fit(self, X, y):

        X_train, X_test, y_train, y_test = \
            train_test_split(X, y, test_size=self.test_size,
                             random_state=self.random_state)

        dim = X_train.shape[1]
        self.indices_ = tuple(range(dim))
        self.subsets_ = [self.indices_]
        score = self._calc_score(X_train, y_train,
                                 X_test, y_test, self.indices_)
        self.scores_ = [score]

        while dim > self.k_features:
            print('dim = {}'.format(dim))
            scores = []
            subsets = []

            for p in combinations(self.indices_, r=dim - 1):
                score = self._calc_score(X_train, y_train,
                                         X_test, y_test, p)
                scores.append(score)
                subsets.append(p)

            best = np.argmax(scores)
            self.indices_ = subsets[best]
            self.subsets_.append(self.indices_)
            dim -= 1

            self.scores_.append(scores[best])
        self.k_score_ = self.scores_[-1]

        return self

    def transform(self, X):
        print('transforming to feature subset = {}'.format(self.indices_))
        return X[:, self.indices_]

    def _calc_score(self, X_train, y_train, X_test, y_test, indices):
        self.estimator.fit(X_train[:, indices], y_train)
        y_pred = self.estimator.predict(X_test[:, indices])
        score = self.scoring(y_test, y_pred)
        return score

if __name__ == '__main__':

    classifier_dict = {'KNN':KNeighborsClassifier(n_neighbors=3),
        'RF':RandomForestClassifier(n_estimators=200),
        'LDA':LinearDiscriminantAnalysis()}
    for label, classifier in classifier_dict.items():
        X_train_std, X_test_std, y_train, y_test = get_train_test_sets()
        # selecting features
        sbs = SBS(classifier, k_features=150)
        sbs.fit(X_train_std, y_train)

        # plotting performance of feature subsets
        k_feat = [len(k) for k in sbs.subsets_]

        plt.plot(k_feat, sbs.scores_, marker='.')
        plt.ylim([0.5, 1.1])
        plt.ylabel('Accuracy [\%]')
        plt.xlabel('Number of features')
        plt.title(label + ' classifier')
        plt.grid()
        plt.tight_layout()
        plt.savefig('../plots/sbs-{}.png'.format(label), dpi=300)
        plt.close()
