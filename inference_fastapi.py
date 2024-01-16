from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
import tensorflow as tf
from typing import List

app = FastAPI()


loaded_model = tf.saved_model.load('/Users/chensitong/MLOps/LFM/MLOps_Group18/checkpoint/final_model/')
inference = loaded_model.signatures["serving_default"]

class PredictionInput(BaseModel):
    input_1: List[float]
    input_2: List[float]


@app.post('/predict')
async def predict(request: Request, input_data: PredictionInput):
    try:
        input_1 = tf.constant(input_data.input_1, dtype=tf.float32)
        input_2 = tf.constant(input_data.input_2, dtype=tf.float32)

        # Inference
        prediction = inference(input_1=input_1, input_2=input_2)
        output_tensor = prediction['output_1']
        predicted_value = int(output_tensor.numpy()[0][0])

        return {'prediction_star': predicted_value}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
