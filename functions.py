import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import time


def encodeing_df(df):
    col_name = []
    label_encoder = LabelEncoder()
    for (colname, colval) in df.iteritems():
        if colval.dtype == 'object':
            col_name.append(colname)
    for col in col_name:
        df[col] = label_encoder.fit_transform(df[col])
    return df


def replace_null(df):
    col_nan = []
    for (colname, colval) in df.iteritems():
        if df[colname].isnull().values.any() == True:
            col_nan.append(colname)

    for col in col_nan:
        mean_value = df[col].mean()
        df[col].fillna(value=mean_value, inplace=True)

    return df


def scaling(df):
    x = df.iloc[:, :-1]  # Using all column except for the last column as X
    y = df.iloc[:, -1]  # Selecting the last
    df_norm = (x - x.min()) / (x.max() - x.min())
    df_norm = pd.concat((df_norm, y), 1)
    print(df_norm)
    return df_norm


def split_data(df, split_size):
    X = df.iloc[:, :-1]  # Using all column except for the last column as X
    Y = df.iloc[:, -1]  # Selecting the last column as Y
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=(100-split_size)/100)

    return X_train, X_test, Y_train, Y_test


def train_model(model, x_train, y_train, x_test, y_test):
    t0 = time.time()
    model.fit(x_train, y_train)
    duration = time.time() - t0
    y_train_pred = model.predict(x_train)
    y_test_pred = model.predict(x_test)

    train_accuracy = np.round(accuracy_score(y_train, y_train_pred), 3)
    train_f1 = np.round(f1_score(y_train, y_train_pred, average="weighted"), 3)

    test_accuracy = np.round(accuracy_score(y_test, y_test_pred), 3)
    test_f1 = np.round(f1_score(y_test, y_test_pred, average="weighted"), 3)

    return model, train_accuracy, train_f1, test_accuracy, test_f1, duration
