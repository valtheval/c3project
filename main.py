import pickle
from fastapi import FastAPI
from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel, Field
import dvc.api
import pandas as pd


# Instantiate app
app = FastAPI()

# Load model
with dvc.api.open(
        'data/inference_model.pkl',
        remote='s3remote',
        mode='rb'
        ) as fd:
    model = pickle.load(fd)
print("model loaded: ", model)

# Define objects
class Item(BaseModel):
    age: int
    workclass: str 
    fnlgt: int
    education: str
    education_num: int = Field(alias="education-num")
    marital_status: str = Field(alias="marital-status")
    occupation: str
    relationship: str
    race: str
    sex: str
    capital_gain: int = Field(alias="capital-gain")
    capital_loss: int = Field(alias="capital-loss")
    hours_per_week: int = Field(alias="hours-per-week")
    native_country: str = Field(alias="native-country")


# Define roots
@app.get("/")
def hello():
    return {"Hello World"}


@app.post("/infer/")
async def infer(item: Item):
    print(item)
    df_item = pd.DataFrame(jsonable_encoder(item), index=[0])
    pred = model.predict(df_item)[0]
    if pred == 0 :
        return {"<=50k"}
    else :
        return {">50k"}



