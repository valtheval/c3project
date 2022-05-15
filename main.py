import pickle
from fastapi import FastAPI
from pydantic import BaseModel
import dvc.api



# Instantiate app
app = FastAPI()


# Load model
with dvc.api.open(
        'data/model.pkl',
        remote='s3remote',
        mode='rb'
        ) as fd:
    model = pickle.load(fd)

# Define objects
class Item(BaseModel):
    age: int
    workclass: str 
    fnlgt: int
    education: str
    education_num: int
    marital_status: str
    occupation: str
    relationship: str
    race: str
    sex: str
    capital_gain: int
    capital_loss: int
    hours_per_week: int
    native_country: str


# Define roots
@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.post("/infer/")
async def create_item(item: Item):
    return item



