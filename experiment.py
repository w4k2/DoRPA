import os
import pickle
import sys

import smote_variants as sv
from sklearn.base import clone
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.svm import SVC
from imblearn.over_sampling import RandomOverSampler, SMOTE

clo_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'modules', 'clo')
sys.path.append(clo_path)

from clo import SamplingTypeEstimator
from clo.base_samplers import RO as sampling_RO, SMOTE as sampling_SMOTE
from data import find_datasets, load_dataset, prepare_X_y

RANDOM_STATE = 0
RESULTS_DIR = 'results'

if not os.path.isdir(RESULTS_DIR):
    os.mkdir(RESULTS_DIR)

FOLDING = RepeatedStratifiedKFold(n_repeats=5, n_splits=2, random_state=90210)

CLASSIFIERS = [
    ('SVM', SVC(kernel='sigmoid', random_state=0)),
]

PROCESSING = [
    ("None", None),
    # ("RO", RandomOverSampler(random_state=RANDOM_STATE)),
    # ("SMOTE", SMOTE(random_state=RANDOM_STATE)),
    # ("DB-SMOTE", sv.DBSMOTE(random_state=RANDOM_STATE)),
    # ("SMOTE TL", sv.SMOTE_TomekLinks(random_state=RANDOM_STATE)),
    # ("SMOTE ENN", sv.SMOTE_ENN(random_state=RANDOM_STATE)),
    # ("CCR", sv.CCR()),
]

def lo_processing(base_classifier):
    return [
        # ("LO RO", SamplingTypeEstimator(sampling_RO(), base_classifier)),
        # ("LO SMOTE", SamplingTypeEstimator(sampling_SMOTE(), base_classifier)),
    ]

def main():
    for ds_name, ds_path in find_datasets():
        results = []
        X, y = prepare_X_y(*load_dataset(ds_path))

        for fold_idx, (train_index, test_index) in enumerate(FOLDING.split(X, y)):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            # Non-optimized methods
            for prc_name, prc in PROCESSING:
                if prc is None:
                    X_, y_ = X_train, y_train
                else:
                    X_, y_ = prc.fit_resample(X_train, y_train)

                for clf_name, clf in CLASSIFIERS:
                    print(f"[{ds_name} - {fold_idx}]{clf_name} + {prc_name}")
                    clf_ = clone(clf)
                    clf_.fit(X_, y_)
                    y_pred = clf_.predict(X_test)
                    cm = confusion_matrix(y_test, y_pred)

                    results.append({
                        "dataset": ds_name,
                        "fold": fold_idx,
                        "prc_name": prc_name,
                        "clf_name": clf_name,
                        "cm": cm.tolist(),
                    })

            # Optimized methods
            for clf_name, clf in CLASSIFIERS:
                for prc_name, prc in lo_processing(clf):
                    print(f"[{ds_name} - {fold_idx}]{clf_name} + {prc_name}")

                    prc_ = clone(prc)
                    prc_.fit(X_train, y_train)
                    y_pred = prc_.predict(X_test)
                    cm = confusion_matrix(y_test, y_pred)

                    results.append({
                        "dataset": ds_name,
                        "fold": fold_idx,
                        "prc_name": prc_name,
                        "clf_name": clf_name,
                        "cm": cm.tolist(),
                    })

        with open(os.path.join(RESULTS_DIR, ds_name), 'wb') as fp:
            pickle.dump(results, fp)


if __name__ == '__main__':
    main()
