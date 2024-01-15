import sys
sys.path.append("C:\\Users\\HC\\project\\LFM")
from logs.logger import logger
import pandas as pd
import numpy as np
import pickle

# function about reading  and prcoessing the movie data 
def read_data_and_process(filname, sep="\t"):
    col_names = ["user", "item", "rate", "st"] # 'st' is the length of the movie 
    df = pd.read_csv(filname, sep=sep, header=None, names=col_names, engine='python') 
    df["user"] -= 1  # user in the raw data starts from 1
    df["item"] -= 1  # item in the raw data starts from 1
    for col in ("user", "item"): 
        df[col] = df[col].astype(np.int32)
    df["rate"] = df["rate"].astype(np.float32)
    #process the null 
    if df.isnull().any().any():
        print("The document contained null values, and these null values were deleted.")
        df = df.dropna()
        df = df.reset_index(drop=True)
    else:
        print("The document doesnot contain null values")
    return df


def split_data(read_path,save_path):
    df = read_data_and_process(read_path, sep="::")
    rows = len(df)
    df = df.iloc[np.random.permutation(rows)].reset_index(drop=True)
    split_index_train_val = int(rows * 0.8)
    slpit_index_val_test = int(rows*0.9)
    df_train = df.iloc[:split_index_train_val,:]  
    df_validation = df.iloc[split_index_train_val:slpit_index_val_test,:]
    df_test = df.iloc[slpit_index_val_test:,:] 
    logger.info("The number of train data is %d" % len(df_train))
    logger.info("The number of validation data is %d" % len(df_validation))
    logger.info("The number of test data is %d" % len(df_test))

    with open(save_path+'train.pickle', 'wb') as file:
        pickle.dump(df_train, file)

    with open(save_path+'validation.pickle', 'wb') as file:
        pickle.dump(df_validation, file)    
    
    with open(save_path+'test.pickle', 'wb') as file:
        pickle.dump(df_test, file)  


if __name__ == '__main__':
    read_path = './data/raw/ratings.dat'
    save_path = './data/processed/'
    split_data(read_path,save_path)
    print ('finish procesing data!')