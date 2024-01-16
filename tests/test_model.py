import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, os.pardir))
sys.path.append(project_root)

from models.LFM import SVDModel
import tensorflow as tf
import pytest
import hydra
from omegaconf import DictConfig


def test_model():
    @hydra.main(config_path="../config", config_name="main.yaml", version_base="1.3.2")
    def _main(cfg: DictConfig):
        enc = hydra.utils.instantiate(cfg)
        model = SVDModel(enc.models)

        user_input = tf.constant([1.0, 2.0, 3.0, 4.0])
        item_input = tf.constant([4.0, 5.0, 6.0, 7.0])
        output_star = model([user_input, item_input])


if __name__ == "__main__":
    test_path = os.path.join(project_root, "tests")
    pytest.main(["-v", test_path])
