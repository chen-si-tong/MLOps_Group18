import sys
import os
import pytest
import numpy as np
import tensorflow as tf
from omegaconf import OmegaConf, DictConfig
import hydra

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, os.pardir))
sys.path.append(project_root)

from models.LFM import SVDModel
from scripts.base.epoch_step import train_step, validation_step

@pytest.fixture
def _main(cfg: DictConfig):
    enc = hydra.utils.instantiate(cfg)
    model = SVDModel(enc)
    return model

@pytest.fixture
def mock_data():
    users = np.random.randint(1, 100, size=100)
    items = np.random.randint(1, 50, size=100)
    rates = np.random.randint(1, 5, size=100)


    return users, items, rates

@pytest.fixture
def cfg():
    config_path = os.path.join(project_root, 'config/test_config.yaml')
    return OmegaConf.load(config_path)

@pytest.mark.usefixtures("_main")
def test_train_step(_main, mock_data, cfg):
    model = _main
    users, items, rates = mock_data
    loss_object = tf.keras.losses.MeanSquaredError()
    optimizer = tf.keras.optimizers.Adam()
    mae_metric = tf.keras.metrics.MeanAbsoluteError()

    loss, mae = train_step(users, items, rates, model, loss_object, optimizer, mae_metric)
    assert loss >= 0  
    assert mae >= 0

if __name__ == "__main__":
    test_path = os.path.join(project_root, 'tests')
    pytest.main(["-v", test_path])
