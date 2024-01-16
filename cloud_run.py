
from flask import Flask, jsonify, request
import tensorflow as tf
from google.cloud import storage
import os
from gcsfs import GCSFileSystem

app = Flask(__name__)

BUCKET_NAME = 'lfm_model'
MODEL_PATH = 'lfm_model/final_model/'

client = storage.Client()
bucket = client.get_bucket(BUCKET_NAME)

loaded_model = tf.saved_model.load("gs://lfm_model/final_model", tags=["serve"])
inference = loaded_model.signatures["serving_default"]

@app.route('/predict', methods=['POST'])
def predict():
    try:
        input_1 = request.json['input_1']
        input_2 = request.json['input_2']

      
        input_1 = tf.constant(input_1, dtype=tf.float32)
        input_2 = tf.constant(input_2, dtype=tf.float32)

        # inference
        prediction = inference(input_1=input_1, input_2=input_2)
        output_tensor = prediction['output_1']
        predicted_value = int(output_tensor.numpy()[0][0])

        return jsonify({'prediction star': predicted_value})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=int(os.environ.get('PORT', 8080)))
