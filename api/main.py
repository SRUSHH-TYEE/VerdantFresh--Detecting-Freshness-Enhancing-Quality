from fastapi import FastAPI, File, UploadFile
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
endpoint="http://localhost:8501/v1/models/apple_model:predict"

# model=tf.keras.models.load_model("../models/1.keras")
# tf.saved_model.save(
#     model,
#     export_dir="../models/1",
# )

class_names=["Apple__Healthy","Aplle_Rotten"]

@app.get('/ping')
async def ping():
    return "Hello World!"

@app.post('/predict')
async def predict(file: UploadFile):
    try:
        image_arr=np.array(Image.open(BytesIO(await file.read())))
        image_batch=np.expand_dims(image_arr,0)
        # pred = model.predict(image_batch)
        # logger.info(pred)

        json_data = {
            "instances":image_batch.tolist()
        }

        tfc.initialize_all_variables().run()
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
