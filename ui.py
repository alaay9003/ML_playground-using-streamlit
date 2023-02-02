import numpy as np
import streamlit as st

from models.NaiveBayes import nb_param_selector
from models.NeuralNetwork import nn_param_selector
from models.RandomForet import rf_param_selector
from models.DecisionTree import dt_param_selector
from models.LogisticRegression import lr_param_selector
from models.KNearesNeighbors import knn_param_selector
from models.SVC import svc_param_selector
from models.GradientBoosting import gb_param_selector


def upload_data():
    with st.sidebar.header('1. Upload your CSV data'):
        uploaded_file = st.sidebar.file_uploader(
            "Upload your input CSV file", type=["csv"])
    return uploaded_file


def model_selector():
    model_training_container = st.sidebar.expander("Train a model", True)
    with model_training_container:
        model_type = st.selectbox(
            "Choose a model",
            (
                "Logistic Regression",
                "Decision Tree",
                "Random Forest",
                "Gradient Boosting",
                "Neural Network",
                "K Nearest Neighbors",
                "Gaussian Naive Bayes",
                "SVC",
            ),
        )

        if model_type == "Logistic Regression":
            model = lr_param_selector()

        elif model_type == "Decision Tree":
            model = dt_param_selector()

        elif model_type == "Random Forest":
            model = rf_param_selector()

        elif model_type == "Neural Network":
            model = nn_param_selector()

        elif model_type == "K Nearest Neighbors":
            model = knn_param_selector()

        elif model_type == "Gaussian Naive Bayes":
            model = nb_param_selector()

        elif model_type == "SVC":
            model = svc_param_selector()

        elif model_type == "Gradient Boosting":
            model = gb_param_selector()

    return model_type, model


def sidebar_controllers():
    data_set = upload_data()
    model_type, model = model_selector()
    st.sidebar.header("Feature engineering")
    degree = polynomial_degree_selector()
    return (
        data_set,
        model_type,
        model,
        degree,
    )
