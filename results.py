import os
import pickle
import pandas as pd

from experiment import RESULTS_DIR

def main():
    if not os.path.isdir(RESULTS_DIR):
        return 1

    table = []

    for f_name in os.listdir(RESULTS_DIR):
        with open(os.path.join(RESULTS_DIR, f_name), 'rb') as fp:
            results = pickle.load(fp)
            table.extend(results)

    df = pd.DataFrame.from_records(table)
    df.to_csv('results.csv')

if __name__ == '__main__':
    main()
