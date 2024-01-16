import pickle
from datetime import datetime
import tensorflow as tf
from evidently.metric_preset import (
    DataDriftPreset,
    DataQualityPreset,
    TargetDriftPreset,
)
from evidently.report import Report
from fastapi import BackgroundTasks, FastAPI
from fastapi.responses import HTMLResponse
from typing import List
from pydantic import BaseModel
import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, os.pardir))
sys.path.append(project_root)

app = FastAPI()

loaded_model = tf.saved_model.load(os.path.join(project_root, "checkpoint/final_model/"))
inference = loaded_model.signatures["serving_default"]


class PredictionInput(BaseModel):
    input_1: List[float]
    input_2: List[float]


@app.post("/lfm_v1/")
def lfm_inference_v1(input_data: PredictionInput):
    """Version 1 of the iris inference endpoint."""
    input_1 = tf.constant(input_data.input_1, dtype=tf.float32)
    input_2 = tf.constant(input_data.input_2, dtype=tf.float32)
    prediction = inference(input_1=input_1, input_2=input_2)
    output_tensor = prediction["output_1"]
    predicted_value = int(output_tensor.numpy()[0][0])

    return {"prediction_star": predicted_value}


with open(os.path.join(project_root, "drift/prediction_database.csv"), "w") as file:
    file.write("time, user, item, rate\n")


def add_to_database(
    now: str,
    input_1: int,
    input_2: int,
    prediction_star: int,
):
    """Simple function to add prediction to database."""

    try:
        with open(os.path.join(project_root, "drift/prediction_database.csv"), "a") as file:
            file.write(f"{now}, {input_1}, {input_2} , {prediction_star}\n")
    except Exception as e:
        print(f"Error adding to database: {str(e)}")


@app.post("/lfm_v2/")
async def iris_inference_v2(
    input_data: PredictionInput,
    background_tasks: BackgroundTasks,
):
    """Version 2 of the lfm inference endpoint."""
    input_1 = tf.constant(input_data.input_1, dtype=tf.float32)
    input_2 = tf.constant(input_data.input_2, dtype=tf.float32)
    prediction = inference(input_1=input_1, input_2=input_2)
    output_tensor = prediction["output_1"]
    predicted_value = int(output_tensor.numpy()[0][0])

    now = str(datetime.now())
    # add_to_database(now, input_data.input_1[0], input_data.input_2[0], predicted_value)

    background_tasks.add_task(
        add_to_database(now, input_data.input_1[0], input_data.input_2[0], predicted_value),
        input_1.numpy().item(),
        input_2.numpy().item(),
        predicted_value,
    )

    return {"prediction_value": predicted_value}


@app.get("/lfm_monitoring/", response_class=HTMLResponse)
async def lfm_monitoring():
    """Simple get request method that returns a monitoring report."""
    with open(os.path.join(project_root, "data/processed/train.pickle"), "rb") as file:
        lfm_dataframe = pickle.load(file)

    data_drift_report = Report(
        metrics=[
            DataDriftPreset(),
            DataQualityPreset(),
            TargetDriftPreset(),
        ]
    )

    data_drift_report.run(
        current_data=lfm_dataframe.iloc[:60],
        reference_data=lfm_dataframe.iloc[60:],
        column_mapping=None,
    )
    data_drift_report.save_html("./drift/monitoring.html")

    with open(os.path.join(project_root, "drift/monitoring.html"), "r", encoding="utf-8") as f:
        html_content = f.read()

    return HTMLResponse(content=html_content, status_code=200)


@app.get("/tasks/")
async def read_tasks():
    return {"message": "Background tasks completed successfully"}
