
import requests
import json

response = requests.get('https://c3project-udacity.herokuapp.com/')

print(response.status_code)
print(response.json())

sample = {"age": "32", "workclass": "Private", "fnlgt": "116138",
              "education": "Masters", "education-num": "14",
              "marital-status": "Never-married", "occupation": "Tech-support",
              "relationship": "Not-in-family", "race": "Asian-Pac-Islander",
              "sex": "Male", "capital-gain": "0", "capital-loss": "0",
              "hours-per-week": "11", "native-country": "Taiwan"}

response = requests.post('https://c3project-udacity.herokuapp.com/infer/', data=json.dumps(sample))

print(response.status_code)
print(response.json())