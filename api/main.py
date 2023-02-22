from fastapi import FastAPI
from pydantic import BaseModel
import gdown, os
import numpy as np

from preprocessing import preprocess_string
import uvicorn
import tensorflow as tf

MODEL_PATH = "./api/model.tflite"

app = FastAPI(tags=["sentence"])

print("Loading model...")
# loading the model
interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
print("Model loadded!")


#if not os.path.exists("./gloveBiGRU"):
#   print("Downloading the model")
#   gdown.download_folder(model_source, 
#                       quiet = True, 
#                       use_cookies = False)



class Input(BaseModel):
    sentence: str

class Output(BaseModel):
    quality:float
    toxic:float
    spam:float

def predict_tflite(inp:Input):
    string = inp.sentence
    string = preprocess_string(string)
    string = np.expand_dims([string], axis=0).astype(str)
    interpreter.set_tensor(input_details[0]['index'], string)

    interpreter.invoke()

    output_data = interpreter.get_tensor(output_details[0]['index'])[0].astype(float)
    data = {}
    data["quality"] = output_data[0]
    data["toxic"] = output_data[1]
    data["spam"] = output_data[2]

    print(output_data)
    print(data)

    return data

    
@app.get("/")
def main():
    return {"response":'Working...'}


@app.post("/predict")
def predict(input:Input):
    return predict_tflite(input)

if __name__ == "__main__":
    print("Running...")
    uvicorn.run("main:app",host='0.0.0.0', port=8080, reload=True)