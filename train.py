import sys
import os
working_directory = "/home/group2/group1/github_test"
os.chdir(working_directory)
sys.path.append(working_directory + "/models")

from models.task1_and_task3 import task1_task3_train
from models.task2 import task2_train

data_path = r"preprocess/Generate_process/Upsample_data.csv"

task1_phobert = task1_task3_train(data_path, r"task1", "vinai/phobert-base")
task3_phobert = task1_task3_train(data_path, r"task3", "vinai/phobert-base")

task2_phobert = task2_train(data_path, r"task2/phobert", "vinai/phobert-base")
task2_electra = task2_train(data_path, r"task2/electra", "NlpHUST/ner-vietnamese-electra-base")
task2_bert = task2_train(data_path, r"task2/bert", "bert-base-multilingual-cased")
