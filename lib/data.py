import pandas as pd

def load_data(csv_path):
    df = pd.read_csv(csv_path)
    assert_right_columns(df)

    df = limit_excessive_samples(df)
    df = balance_dataset(df)
    df = limit_dataset(df)

    return to_X_y(df)



def to_X_y(df):
    X = df.drop(['DatasetID', 'y1', 'y2'], axis=1).fillna(0).to_numpy()
    y = df[['y1', 'y2']].to_numpy()
    return X, y



def split_train_test(X, y, test_split=0.2):
    test_split = 0.2
    test_size = int(X.shape[0] * test_split)
    X_train, X_test = X[:-test_size], X[-test_size:]
    y_train, y_test = y[:-test_size], y[-test_size:]

    return X_train, X_test, y_train, y_test


def limit_dataset(df, limit=50000):
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    return df[:limit]


def limit_excessive_samples(df, max_size=50000):
    def limit_size(df, DatasetID, limit):
        df_ds = df[df.DatasetID == DatasetID][:limit]
        df_other = df[~(df.DatasetID == DatasetID)]
        return pd.concat([df_ds, df_other], axis=0)

    df = limit_size(df, 1, max_size)
    df = limit_size(df, 2, max_size)

    return df


def balance_dataset(df):
    d1_size = df[df.DatasetID == 1].shape[0]
    d2_size = df[df.DatasetID == 2].shape[0]

    def upsample(df, DatasetID, num):
        print(f'Sampling from dataset {DatasetID} {num} samples')
        return pd.concat([df, df[df.DatasetID==DatasetID].sample(n=num, replace=True)], axis=0)

    if (d1_size > d2_size):
        return upsample(df, 2, d1_size-d2_size)
    return upsample(df, 1, d2_size-d1_size)


def assert_right_columns(df):
    expected_cols = {'DatasetID', 'x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'y1', 'y2', 'z'}
    if not set(df.columns.values) == expected_cols:
        raise ValueError(f'Expected columns: {expected_cols}')