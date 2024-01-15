import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, os.pardir))
sys.path.append(project_root)

from base.dataloader import *
from base.epoch_step import train_step,validation_step
import pickle
import hydra
from omegaconf import DictConfig
from models.LFM import SVDModel
import tensorflow as tf
import time
import wandb 

@hydra.main(config_path="../config", config_name="main.yaml", version_base="1.3.2")
def _main(cfg: DictConfig,train_step=train_step,validation_step=validation_step):
    wandb.init(
    # set the wandb project where this run will be logged
    project="MLops_LFM", 
    entity="s230027", 
    group="LFM", 
    name="mse", 
    # track hyperparameters and run metadata
    config={
    "learning_rate": 0.1,
    "architecture": "LFM",
    "dataset": "m1-1m",
    "epochs":500,
    'dim': 6,
    'reg_strength': 0.1,
    'clipvalue': 2,
    'batch_size':2000
    }
)
    enc = hydra.utils.instantiate(cfg)
    model = SVDModel(enc.models)

    save_path = './data/processed/'
    with open(save_path+'train.pickle', 'rb') as file:
        df_train = pickle.load(file)
    with open(save_path+'validation.pickle', 'rb') as file:
        df_validation = pickle.load(file)
    train_data = DataSampler([df_train["user"],
                                    df_train["item"],
                                    df_train["rate"]],
                                    batch_size=enc.epoch['batch_size'])
    validation_data = DataSampler([df_validation["user"],
                                    df_validation["item"],
                                    df_validation["rate"]],
                                    batch_size=enc.epoch['batch_size'])
    train_validaiton(enc.epoch,train_data,validation_data,model,train_step,validation_step)


def train_validaiton(cfg,train_data,validation_data,model,train_step,validation_step):
    draw_num = 10
    iter = []
    train_loss,train_mae=[],[]
    valid_loss, valid_mae=[],[]
    train_iter,validation_iter = [],[]
    train_loss_batch_record, train_mae_batch_record=[],[]
    valid_loss_batch_record, valid_mae_batch_record=[],[]
    loss_object = tf.keras.losses.MeanSquaredError()
    optimizer = tf.keras.optimizers.Adam(learning_rate=cfg.learning_rate, clipvalue=cfg.clipvalue)
    mae_metric = tf.keras.metrics.MeanAbsoluteError()
    for epoch in range(cfg.num_epochs):
        #train data 
        mae_metric.reset_states() 
        train_loss_batch, train_mae_batch= [], [] 
        start = time.time()
        for i, (users, items, rates) in enumerate(train_data):
            loss, mae = train_step(users, items, rates,model,loss_object,optimizer,mae_metric)
            train_loss_batch.append(loss.numpy())
            train_mae_batch.append(mae.numpy())
            train_loss_batch_record.append(loss.numpy())
            train_mae_batch_record.append(mae.numpy())
            train_iter.append(i+1)
            if i+1 >= cfg.batch_size:
                break
        train_loss.append(np.mean(train_loss_batch))
        train_mae.append(np.mean(train_mae_batch))
        wandb.log({'train_loss':np.mean(train_loss_batch), 'step':epoch+1})
        wandb.log({'train_mae':np.mean(train_mae_batch), 'step':epoch+1})
    
        #validation data
        valid_loss_batch, valid_mae_batch = [],[]
        for j, (users, items, rates) in enumerate(validation_data):
            val_loss, val_mae = validation_step(users, items, rates,model,loss_object,mae_metric)
            valid_loss_batch.append(val_loss.numpy())
            valid_mae_batch.append(val_mae.numpy())        
            valid_loss_batch_record.append(val_loss.numpy())
            valid_mae_batch_record.append(val_mae.numpy())
            validation_iter.append(j+1)
            if j+1 >= cfg.batch_size:
                break 
        valid_loss.append(np.mean(valid_loss_batch))
        valid_mae.append(np.mean(valid_mae_batch))
        iter.append(epoch+1)
        wandb.log({'validation_loss':np.mean(valid_loss_batch), 'step':epoch+1})
        wandb.log({'validation_mae':np.mean(valid_mae_batch), 'step':epoch+1})
        if epoch % draw_num == 0:
            end = time.time()
            logger.info(f"Epoch {epoch+1}: Train Loss = {np.mean(train_loss_batch)}, Train MAE = {np.mean(train_mae_batch)},Time Consuming = {round((end - start),3)}(s)")
            logger.info(f"Epoch {epoch+1}: Validation Loss = {np.mean(valid_loss_batch)}, Validation MAE = {np.mean(valid_mae_batch)},Time Consuming = {round((end - start),3)}(s)")
    logger.info("Finished training.")
    wandb.finish()

    # save model 
    tf.saved_model.save(model, './checkpoint/final_model')


if __name__ == "__main__":
    from logs.logger import logger
    _main()