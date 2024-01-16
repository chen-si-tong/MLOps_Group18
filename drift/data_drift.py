import pandas as pd
import pickle   
from sklearn import datasets
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, DataQualityPreset
import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, os.pardir))
sys.path.append(project_root)



path = os.path.join(project_root,'drift/')

current_data = pd.read_csv(os.path.join(path,'prediction_database.csv'))
current_data = current_data.rename(columns={' user': 'user', ' item': 'item',' rate':'rate'})
current_data = current_data[['user','item','rate']]

with open(os.path.join(project_root,'data/processed/train.pickle'), "rb") as file:
    reference_data = pickle.load(file)
    
reference_data = reference_data[['user','item','rate']]


report = Report(metrics=[DataDriftPreset(), DataQualityPreset()])
report.run(reference_data=reference_data, current_data=current_data)
report.save_html(os.path.join(path,'drift_report.html'))