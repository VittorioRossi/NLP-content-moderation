from fastapi import FastAPI, Response
from pydantic import BaseModel
import spacy, gdown, os
from configparser import ConfigParser

model_dir = "./best-model-CNN-v1"
model_source = "https://drive.google.com/drive/folders/1Poi1kOOB0Yf7fmKG2vMZOEaYBo8lh1OE?usp=sharing"


gdown.download_folder(model_source, 
                      quiet = True, 
                      use_cookies = False)


nlp_model = spacy.load(model_dir)

app = FastAPI(tags=["sentence"])

# defin
class Input(BaseModel):
    sentence: str

class Output(BaseModel):
    quality:float
    spam:float
    toxic:float

@app.get("/")
def main():
    return {"response":'Working...'}


@app.post("/predict")
def predict(input:Input):
    print(input.sentence)
    res  = nlp_model(input.sentence).cats
    print(res)
    return res


