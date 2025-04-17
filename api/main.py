from fastapi import FastAPI, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import requests

import numpy as np
import tensorflow as tf
import logging
from PIL import Image
from io import BytesIO

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app=FastAPI()

origins = [
    "http://localhost",
    "http://localhost:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

endpoint="http://localhost:8501/v1/models/apple_model:predict"

class_names=["Apple__Healthy","Aplle_Rotten"]

@app.get('/ping')
async def ping():
    return "Hello World!"

@app.post('/predict')
async def predict(file: UploadFile):
    try:
        image_arr=np.array(Image.open(BytesIO(await file.read())))
        image_batch=np.expand_dims(image_arr,0)

        json_data = {
            "instances":image_batch.tolist()
        }

        response=requests.post(endpoint,json=json_data)
        prediction=response.json()["predictions"][0]

        predicted_class = class_names[np.argmax(prediction)]
        confidence = float(np.max(prediction))

        return {
            'class':predicted_class,
            'confidence':confidence
        }

    except Exception as e:
        logger.error(f"Error processing image: {e}")



if __name__=="__main__":
    uvicorn.run(app,host='localhost',port=8000)
