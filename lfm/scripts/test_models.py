import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, os.pardir))
sys.path.append(project_root)
import tensorflow as tf
import pickle
from base.dataloader import DataSamplerForTest
from logs.logger import logger
from sklearn.metrics import mean_squared_error

model_path = "./checkpoint/final_model/"


def test(model_path):
    save_path = "./data/processed/"
    with open(save_path + "test.pickle", "rb") as file:
        df_test = pickle.load(file)

    test_data = DataSamplerForTest([df_test["user"], df_test["item"], df_test["rate"]], batch_size=-1)

    loaded_model = tf.saved_model.load(model_path)
    inference = loaded_model.signatures["serving_default"]
    for user, item, rate in test_data:
        user = tf.constant(user, dtype=tf.float32)
        item = tf.constant(item, dtype=tf.float32)
        rate = tf.constant(rate, dtype=tf.float32)

    output_star = inference(input_1=user, input_2=item)
    rate = tf.reshape(rate, output_star["output_1"].shape)
    mse = mean_squared_error(rate, output_star["output_1"])
    logger.info(f"Mean Squared Error (MSE): {mse}")


if __name__ == "__main__":
    test(model_path)
