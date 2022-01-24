import requests
import json

# get request
print('Get request::')
response = requests.get('https://live-pred-app.herokuapp.com/')
print('---', 'response code: ', response.status_code)
print('---', 'response body: ', response.json())


# post request
print('Post request::')
row = {'age': 50, 'workclass': 'Self-emp-not-inc',
           'fnlgt': 83311, 'education': 'Bachelors',
           'education-num': 13, 'marital-status': 'Married-civ-spouse',
           'occupation': 'Exec-managerial', 'relationship': 'Husband',
           'race': 'White', 'sex': 'Male',
           'capital-gain': 0, 'capital-loss': 0,
           'hours-per-week': 13, 'native-country': 'United-States'
          }

response = requests.post('https://live-pred-app.herokuapp.com/predict', data=json.dumps(row))

print('---', 'response code: ', response.status_code)
print('---', 'response body: ', response.json())