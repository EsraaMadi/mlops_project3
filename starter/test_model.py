import pandas as pd
import pytest
import os
import pickle

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from .ml.model import train_model, compute_model_metrics, inference
from .ml.data import process_data


@pytest.fixture
def data():
#     # load data 
#     df = pd.read_csv("../data/census_clean_data.csv")
   
#     # load model
#     model = os.path.join(os.getcwd(), "../model/model.pkl")
#     with open(model, "rb") as f:
#         model = pickle.load(f)

#     # load encoder
#     encoder = os.path.join(os.getcwd(), "../model/encoder.pkl")
#     with open(encoder, "rb") as f:
#         encoder = pickle.load(f)

#     # load lb
#     lb = os.path.join(os.getcwd(), "../model/lb.pkl")
#     with open(lb, "rb") as f:
#         lb = pickle.load(f)

    # load data
    df = pd.read_csv("data/census_clean_data.csv", index=0)
    
    # load model
    model_path = "model/model.pkl"
    with open(model_path, "rb") as f:
        model = pickle.load(f)

    # load encoder
    encoder_path = "model/encoder.pkl"
    with open(encoder_path, "rb") as f:
        encoder = pickle.load(f)
        
    # load lb
    lb_path = "model/lb.pkl"
    with open(lb_path, "rb") as f:
        lb = pickle.load(f)

    return df, model, encoder, lb

@pytest.fixture
def train_test(data):
    
    df, _, encoder, lb = data
    # split data
    train, test = train_test_split(df, test_size=0.20, random_state=42)
    
    cat_features = [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country",
    ]

    X_train, y_train, _, _ = process_data(
        train,
        categorical_features=cat_features,
        label="salary",
        training=True,
        encoder=encoder,
        lb=lb,
    )

    X_test, y_test, _, _ = process_data(
        test,
        categorical_features=cat_features,
        label="salary",
        training=False,
        encoder=encoder,
        lb=lb,
    )
    print(X_train.shape)
    return train, X_train, y_train, test, X_test, y_test

def test_train_model(train_test):
    _, X_train, y_train, _, _, _ = train_test
    
    model = train_model(X_train, y_train)
    assert isinstance(model, RandomForestClassifier)


def test_compute_model_metrics(data, train_test):
    _, model, _, _ = data
    _, X_train, y_train, _, X_test, y_test= train_test

    y_train_pred = inference(model, X_train)
    y_test_pred = inference(model, X_test)

    precision_train, recall_train, fbeta_train = compute_model_metrics(y_train,
                                                                       y_train_pred
                                                                      )
    precision_test, recall_test, fbeta_test = compute_model_metrics(y_test,
                                                                    y_test_pred
                                                                   )
    assert isinstance(precision_train, float)
    assert isinstance(precision_test, float)
    assert isinstance(recall_train, float)
    assert isinstance(recall_test, float)
    assert isinstance(fbeta_train, float)
    assert isinstance(fbeta_test, float)



def test_inference(data, train_test):
    _, model, _, _= data
    _, _, _, _, X_test, y_test= train_test
    
    
    y_pred = inference(model, X_test)
    assert len(y_pred) == X_test.shape[0]
    assert len(y_pred) > 0