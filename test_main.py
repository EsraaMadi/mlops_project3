from fastapi.testclient import TestClient
from main import app
import pandas as pd

client = TestClient(app)


def test_welcome():
    r = client.get("/")
    assert r.status_code == 200
    assert r.json() == {"Home": "Welcome I am prediction app"}


def test_predict_salary_positive():
    df = pd.read_csv('data/census_clean_data.csv')
    df_dict = df.to_dict(orient='records')
    row = {'age': 37,
           'workclass': ' Private',
           'fnlgt': 284582,
           'education': ' Masters',
           'education-num': 14,
           'marital-status': ' Married-civ-spouse',
           'occupation': ' Exec-managerial',
           'relationship': ' Wife',
           'race': ' White',
           'sex': ' Female',
           'capital-gain': 0,
           'capital-loss': 0,
           'hours-per-week': 40,
           'native-country': ' United-States'
          }
    req = client.post("/predict", json=row)
    assert req.status_code == 200
    assert req.json() == 1


def test_predict_salary_negative():
    row = {'age': 31, 'workclass': 'Private',
            'fnlgt': 45781, 'education': 'Masters',
            'education-num': 14, 'marital-status': 'Never-married',
            'occupation': 'Prof-specialty', 'relationship': 'Not-in-family',
            'race': 'White', 'sex': 'Female',
            'capital-gain': 2174, 'capital-loss': 0,
            'hours-per-week': 50, 'native-country': 'United-States'
           }
    req = client.post("/predict", json=row)
    assert req.status_code == 200
    assert req.json() == 0