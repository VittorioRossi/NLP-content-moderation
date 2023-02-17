from fastapi import FastAPI
from pydantic import BaseModel
import gdown, os

from preprocessing import preprocess_text
from keras.models import load_model
import uvicorn

app = FastAPI(tags=["sentence"])


print("Running...")
model_dir = "./gloveBiGRU"
model_source = "https://drive.google.com/drive/folders/1-1XCaCdyiCiAgZQQGz3R1144r79kbpcX?usp=sharing"


if not os.path.exists("./gloveBiGRU"):
    print("Downloading the model")
    gdown.download_folder(model_source, 
                        quiet = True, 
                        use_cookies = False)


nlp_model = load_model(model_dir)

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
    real_input = preprocess_text(input.sentence)
    print(real_input)
    res  = nlp_model.predict(real_input)
    print(res)
    return res
