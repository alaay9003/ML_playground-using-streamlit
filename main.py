import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import re
from sklearn.metrics import accuracy_score, f1_score
import numpy as np
from sklearn.preprocessing import LabelEncoder

from ui import (
    model_selector,
    upload_data,
    sidebar_controllers,

)
from functions import *


def build_model(df, model):
    st.write("Data Set After preprossing ")
    st.write(df.head())
    X = df.iloc[:, :-1]  # Using all column except for the last column as X
    Y = df.iloc[:, -1]  # Selecting the last column as Y

    with st.sidebar.header('2. Set Parameters'):
        split_size = st.sidebar.slider(
            'Data split ratio (% for Training Set)', 10, 90, 80, 5)

    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=(100-split_size)/100)

    st.markdown('** Data splits**')
    st.write('Training set')
    st.info(X_train.shape)
    st.write('Test set')
    st.info(X_test.shape)

    st.markdown('** Variable details**:')
    st.write('X variable')
    st.info(list(X.columns))
    st.write('Y variable')
    st.info(Y.name)

    model.fit(X_train, Y_train)
    st.write('Model Info')
    st.info(model)

    st.markdown('** Training set**')
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    train_accuracy = np.round(accuracy_score(Y_train, y_train_pred), 3)
    train_f1 = np.round(f1_score(Y_train, y_train_pred, average="weighted"), 3)

    test_accuracy = np.round(accuracy_score(Y_test, y_test_pred), 3)
    test_f1 = np.round(f1_score(Y_test, y_test_pred, average="weighted"), 3)
    st.write('train_accuracy')
    st.info(train_accuracy)
    st.write('train_f1')
    st.info(train_f1)
    st.write('test_accuracy')
    st.info(test_accuracy)
    st.write('test_f1')
    st.info(test_f1)

    st.write('Model Parameters')
    st.write(model.get_params())


def encodeing_df(df):
    col_name = []
    label_encoder = LabelEncoder()
    # st.write(df.head())
    for (colname, colval) in df.iteritems():
        if colval.dtype == 'object':
            col_name.append(colname)
            # st.info(colname)
    # st.info(col_name)
    for col in col_name:
        df[col] = label_encoder.fit_transform(df[col])
    # st.write(df.head())
    return df


if __name__ == "__main__":

    (
        data_set,
        model_type,
        model,
        degree,

    ) = sidebar_controllers()


if data_set is not None:
    df = pd.read_csv(data_set)
    st.subheader(' Glimpse of dataset')
    st.write(df.head())
    df = replace_null(df)
    df = encodeing_df(df)
    df = scaling(df)
    (X_train, X_test, Y_train, Y_test) = split_data(df, 80)
    (model, train_accuracy, train_f1, test_accuracy, test_f1, duration) = train_model(
        model, X_train, Y_train, X_test, Y_test)
else:
    st.info('Awaiting for CSV file to be uploaded.')
