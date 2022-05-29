from main import app
from fastapi.testclient import TestClient
import json

client = TestClient(app)


def test_hello_world():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == ["Hello World"]


def test_infer():
    sample = {
        'age': 39,
        'workclass': 'State-gov',
        'fnlgt': 77516,
        'education': 'Bachelors',
        'education-num': 13,
        'marital-status': 'Never-married',
        'occupation': 'Adm-clerical',
        'relationship': 'Not-in-family',
        'race': 'White',
        'sex': 'Male',
        'capital-gain': 2174,
        'capital-loss': 0,
        'hours-per-week': 40,
        'native-country': 'United-States'}
    response = client.post("/infer/", data=json.dumps(sample))
    assert response.status_code == 200
    assert response.json() == ['<=50k']


def test_inference_sample_2():
    sample = {
        'age': 76,
        'workclass': 'Private',
        'fnlgt': 124191,
        'education': 'Masters',
        'education-num': 14,
        'marital-status': 'Married-civ-spouse',
        'occupation': 'Exec-managerial',
        'relationship': 'Husband',
        'race': 'White',
        'sex': 'Male',
        'capital-gain': 0,
        'capital-loss': 0,
        'hours-per-week': 40,
        'native-country': 'United-States'}
    response = client.post("/infer/", data=json.dumps(sample))
    assert response.status_code == 200
    assert response.json() == ['>50k']