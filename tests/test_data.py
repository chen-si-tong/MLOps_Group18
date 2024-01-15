import pickle
import pytest
import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, os.pardir))
sys.path.append(project_root)

def test_data():
    save_path = os.path.abspath(os.path.join(project_root,'data/processed/'))
    with open(os.path.abspath(os.path.join(save_path,'train.pickle')), 'rb') as file:
        df_train = pickle.load(file)
    with open(os.path.abspath(os.path.join(save_path,'validation.pickle')), 'rb') as file:
        df_validation = pickle.load(file)
    with open(os.path.abspath(os.path.join(save_path,'test.pickle')), 'rb') as file:
        df_test = pickle.load(file)


    assert df_train.isnull().any().any() == False
    assert df_validation.isnull().any().any() == False
    assert df_test.isnull().any().any() == False


if __name__ == "__main__":
    test_path = os.path.join(project_root, 'tests')
    pytest.main(["-v", test_path])