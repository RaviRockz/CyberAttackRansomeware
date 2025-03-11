import warnings

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler

plt.rcParams["font.family"] = "IBM Plex Mono"
warnings.filterwarnings("ignore", category=Warning)


def load_data():
    dp = "Data/data_file.csv"
    print("[INFO] Loading Data From :: {0}".format(dp))
    df = pd.read_csv(dp)
    df.columns = df.columns.values.tolist()[:-1] + ['Class']
    print("[INFO] Data Shape :: {0}".format(df.shape))
    return df


def preprocess_data(df: pd.DataFrame):
    ucols = ['FileName', 'md5Hash']
    print(f'[INFO] Dropping Unwanted Columns :: {ucols}')
    df.drop(ucols, axis=1, inplace=True)
    df = pd.concat([df[df['Class'] == 0].head(10000), df[df['Class'] == 1].head(10000)]).sample(frac=1)
    x = df.values[:, :-1]
    print('[INFO] Scaling Data Using Z-Score Normalization')
    scaler = StandardScaler()
    x = scaler.fit_transform(x)
    df[df.columns.values[:-1]] = x
    sp = "Data/preprocessed.csv"
    print("[INFO] Saving Preprocessed Data To :: {0}".format(sp))
    df.to_csv(sp, index=False)
    print("[INFO] Data Shape :: {0}".format(df.shape))
    return df


if __name__ == "__main__":
    preprocess_data(load_data())
