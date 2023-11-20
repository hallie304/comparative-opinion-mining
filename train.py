import sys
sys.path.append(r"C:\Users\Public\VLSP23-Comparative-Opinion-Mining\models")

from models.task1_and_task3 import task1_task3_train
from models.task2 import task2_train

task1_phobert = task1_task3_train(r"C:\Users\Public\VLSP23-Comparative-Opinion-Mining\data\all_task_ver3_combine_as_pred.csv", r"C:\Users\Public\VLSP23-Comparative-Opinion-Mining\model_outputs\task1", "vinai/phobert-base")

task2_phobert = task2_train(r"C:\Users\Public\VLSP23-Comparative-Opinion-Mining\data\all_task_ver3_combine_as_pred.csv", r"C:\Users\Public\VLSP23-Comparative-Opinion-Mining\model_outputs\task2\phobert", "vinai/phobert-base")
task2_electra = task2_train(r"C:\Users\Public\VLSP23-Comparative-Opinion-Mining\data\all_task_ver3_combine_as_pred.csv", r"C:\Users\Public\VLSP23-Comparative-Opinion-Mining\model_outputs\task2\electra", "NlpHUST/ner-vietnamese-electra-base")
task2_bert = task2_train(r"C:\Users\Public\VLSP23-Comparative-Opinion-Mining\data\all_task_ver3_combine_as_pred.csv", r"C:\Users\Public\VLSP23-Comparative-Opinion-Mining\model_outputs\task2\bert", "bert-base-multilingual-cased")

task3_phobert = task1_task3_train(r"C:\Users\Public\VLSP23-Comparative-Opinion-Mining\data\all_task_ver3_combine_as_pred.csv", r"C:\Users\Public\VLSP23-Comparative-Opinion-Mining\model_outputs\task3", "vinai/phobert-base")
