import os
import pandas as pd
import numpy as np
from collections import Counter

from sklearn.compose import make_column_selector
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder

DATA_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'data')

def find_datasets():
    for f_name in filter(lambda _: _.endswith('.csv'), os.listdir(DATA_DIR)):
        ds_name = f_name.split('.')[0]
        ds_path = os.path.join(DATA_DIR, f_name)
        yield ds_name, ds_path


def load_dataset(ds_path):
    df = pd.read_csv(ds_path, delimiter=';', decimal=',', low_memory=False)
    return df.iloc[:,:-1], df.iloc[:,-1:]


def prepare_X_y(X, y):
    X = StandardScaler().fit_transform(X.to_numpy())
    return X, LabelEncoder().fit_transform(y.to_numpy().ravel())


def main():
    table = []

    for ds_name, ds_path in find_datasets():
        X, y = load_dataset(ds_path)
        X, y = prepare_X_y(X, y)
        table.append({
            "Dataset": ds_name,
            "Samples": len(X),
            "Attributes": len(X.T),
            **dict(Counter(y)),
        })

    print(pd.DataFrame(table).set_index("Dataset").to_markdown())

    table = []
    for ds_name, ds_path in find_datasets(multiclass=True):
        X, y = prepare_X_y(*load_dataset(ds_path))
        table.append({
            "Dataset": ds_name,
            "Samples": len(X),
            "Attributes": len(X.T),
            **dict(Counter(y)),
        })

        print(pd.DataFrame(table).set_index("Dataset").to_markdown())

if __name__ == '__main__':
    main()
