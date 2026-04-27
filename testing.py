import joblib

cases=joblib.load("sample_test.pkl")

import requests


url='http://127.0.0.1:8000/predict'

for case in cases:
    response=requests.post(url,json=case)
    print(response.json())