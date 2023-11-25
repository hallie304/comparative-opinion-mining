import py_vncorenlp
import os
segmenter = py_vncorenlp.VnCoreNLP(annotators=["wseg"], save_dir = os.getcwd() + "/vncorenlp")
working_directory = "/home/group2/group1/github_test"
os.chdir(working_directory)
import torch
from transformers.file_utils import is_tf_available, is_torch_available, is_torch_tpu_available
from transformers import AutoModelForSequenceClassification, AutoTokenizer, TrainingArguments, Trainer, AutoModel, AutoConfig
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
import ast
import random
import pandas as pd
import evaluate

def task1_task3_train(data_path, output_path, model_path):
    def set_seed(seed: int):
        random.seed(seed)
        np.random.seed(seed)
        if is_torch_available():
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            # ^^ safe to call this function even if cuda is not available
        if is_tf_available():
            import tensorflow as tf

            tf.random.set_seed(seed)

    set_seed(1)
    id2label = {0: "None", 1: "DIF", 2: "EQL", 3: "SUP+", 4: "SUP-", 5: "SUP", 6: "COM+", 7: "COM-", 8: "COM"}
    label2id = {"None": 0, "DIF": 1, "EQL": 2, "SUP+": 3, "SUP-": 4, "SUP": 5, "COM+": 6, "COM-": 7, "COM": 8}

    df = pd.read_csv(data_path)
    df = df[(df["content"] != " ") & (df["content"] != "")].sample(frac=1).reset_index(drop=True)
    df["label"] = df["label"].apply(ast.literal_eval).apply(lambda x: x[0] if len(x) >= 1 else 0)

    phobert = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=9, id2label=id2label,
                                                                 label2id=label2id)
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    class DatasetSequenceVLSP(torch.utils.data.Dataset):
        def __init__(self, encoding, labels):
            self.encoding = encoding
            self.labels = labels

        def __len__(self):
            return len(self.encoding['input_ids'])

        def __getitem__(self, idx):
            item = {k: torch.tensor(v[idx]) for k, v in self.encoding.items()}
            item["labels"] = torch.tensor([self.labels.iloc[idx]])
            return item

    skf = StratifiedKFold(n_splits=3, shuffle=True)
    df_dct = df[["content", "label"]]
    df_dct["content"] = df_dct["content"].map(lambda x: segmenter.word_segment(x)[0])

    train_datasets, valid_datasets = [], []
    for train, test in skf.split(df_dct['content'], df_dct['label']):
        train_encoding = tokenizer(df_dct['content'][train].to_list(), padding=True)
        valid_encoding = tokenizer(df_dct['content'][test].to_list(), padding=True)

        train_dataset = DatasetSequenceVLSP(train_encoding, df_dct['label'][train])
        valid_dataset = DatasetSequenceVLSP(valid_encoding, df_dct['label'][test])

        train_datasets.append(train_dataset)
        valid_datasets.append(valid_dataset)

    f1 = evaluate.load("f1")


    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        return f1.compute(predictions=predictions, references=labels, average="macro")

    models = []
    for i, (X, y) in enumerate(zip(train_datasets, valid_datasets)):
        phobert = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=9,
                                                                     id2label=id2label, label2id=label2id).to("cuda")
        tokenizer = AutoTokenizer.from_pretrained(model_path)

        training_args = TrainingArguments(
            output_dir=f"results/{i}",
            learning_rate=3e-5,
            per_device_train_batch_size=32,
            per_device_eval_batch_size=32,
            num_train_epochs=1,
            evaluation_strategy="epoch",
            save_strategy="no",
        )

        trainer = Trainer(
            model=phobert,
            args=training_args,
            train_dataset=X,
            eval_dataset=y,
            compute_metrics=compute_metrics,
        )

        trainer.train()
        output = output_path + "/" + str(i)
        trainer.save_model(output_dir=output)

        models.append(trainer)

    for i, model in enumerate(models):
        output = output_path + "/" + str(i)
        model.save_model(output_dir=output)

