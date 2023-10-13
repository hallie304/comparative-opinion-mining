import ast
import os
import shutil
import pandas as pd
import numpy as np
# from tqdm.auto import tqdm
# from tqdm.notebook import tqdm
from tqdm import tqdm
from time import sleep
from copy import deepcopy
import torch
import transformers
from transformers import AutoModel, AutoTokenizer, AutoModelForPreTraining, AutoFeatureExtractor
from transformers import DataCollatorForTokenClassification
from transformers import AutoModelForTokenClassification, TrainingArguments, Trainer
import evaluate
import py_vncorenlp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
# from underthesea import ner
from torch.optim import AdamW
from transformers import get_scheduler
import matplotlib.pyplot as plt


curDir = os.getcwd()
print("Current directory: %s" % (curDir))

#ALL LABELS OF <NOCLASS> CAN BE REPLACED BY -100 (NEED EXPERIMENT)
labelDict = {'subject': 0, 'object': 1, 'aspect': 2, 'predicate': 3, '<noClass>': 4}
label2id = labelDict
id2label = {0: 'subject', 1: 'object', 2: 'aspect', 3: 'predicate', 4: '<noClass>'}

comparisonLabel2id = {'DIF': 0, 'EQL': 1, 'SUP': 2, 'SUP+': 3, 'SUP-': 4, 'COM': 5, 'COM+': 6, 'COM-': 7, '<noClass>': 8}
id2comparisonLabel = {0: 'DIF', 1: 'EQL', 2: 'SUP', 3: 'SUP+', 4: 'SUP-', 5: 'COM', 6: 'COM+', 7: 'COM-', 8: '<noClass>'}