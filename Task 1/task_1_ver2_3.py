import py_vncorenlp
segmenter = py_vncorenlp.VnCoreNLP(annotators=["wseg"], save_dir=r"C:\Users\Duc Bao\PycharmProjects\VLSP All Code\vncorenlp")

# %% [code] {"execution":{"iopub.status.busy":"2023-11-03T17:29:51.442541Z","iopub.execute_input":"2023-11-03T17:29:51.443627Z","iopub.status.idle":"2023-11-03T17:30:04.409943Z","shell.execute_reply.started":"2023-11-03T17:29:51.443588Z","shell.execute_reply":"2023-11-03T17:30:04.409160Z"}}
import torch
from transformers.file_utils import is_tf_available, is_torch_available, is_torch_tpu_available
from transformers import AutoModelForSequenceClassification, AutoTokenizer, TrainingArguments, Trainer, AutoModel, \
    AutoConfig
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
import ast
import random
import pandas as pd
import torch.optim as optim
import evaluate



# %% [code] {"execution":{"iopub.status.busy":"2023-11-03T17:30:04.411029Z","iopub.execute_input":"2023-11-03T17:30:04.411605Z","iopub.status.idle":"2023-11-03T17:30:04.422000Z","shell.execute_reply.started":"2023-11-03T17:30:04.411576Z","shell.execute_reply":"2023-11-03T17:30:04.421255Z"}}
def set_seed(seed: int):
    """
    Helper function for reproducible behavior to set the seed in ``random``, ``numpy``, ``torch`` and/or ``tf`` (if
    installed).

    Args:
        seed (:obj:`int`): The seed to set.
    """
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

# %% [code] {"execution":{"iopub.status.busy":"2023-11-03T17:30:44.623818Z","iopub.execute_input":"2023-11-03T17:30:44.624251Z","iopub.status.idle":"2023-11-03T17:30:44.997030Z","shell.execute_reply.started":"2023-11-03T17:30:44.624216Z","shell.execute_reply":"2023-11-03T17:30:44.996090Z"}}

# id2label = {0:"Non Comparative",1:"Comparative"}
# label2id = {"Non Comparative":0,"Comparative":1}
id2label = {0: "None", 1: "DIF", 2: "EQL", 3: "SUP+", 4: "SUP-", 5: "SUP", 6: "COM+", 7: "COM-", 8: "COM"}
label2id = {"None": 0, "DIF": 1, "EQL": 2, "SUP+": 3, "SUP-": 4, "SUP": 5, "COM+": 6, "COM-": 7, "COM": 8}

df = pd.read_csv(r"C:\Users\Duc Bao\PycharmProjects\VLSP All Code\data\all_task_ver3_combine_as_pred.csv")
df = df[(df["content"] != " ") & (df["content"] != "")].sample(frac=1).reset_index(drop=True)
df["label"] = df["label"].apply(ast.literal_eval).apply(lambda x: x[0] if len(x) >= 1 else 0)


# %% [markdown]
# ## Combine two datasets
# 1. original (non comparative) + generated data

# %% [code]
# # --- original dataset
# ori_df = pd.read_csv('/kaggle/input/combined-bio/total.csv', index_col = 0)
# ori_df["label"] = ori_df["comparative"]
# ori_df = ori_df.loc[ori_df["comparative"] == 0, ["content", "label"]].reset_index(drop=True)
# ori_df

# %% [code]
# df["label"] = 1
# df

# %% [code]
# df['label'].value_counts()

# %% [code]
# df = pd.concat([ori_df, df.loc[:, ["content", "label"]]]).sample(frac=1).reset_index(drop=True)
# df

# %% [code] {"execution":{"iopub.status.busy":"2023-11-03T17:30:48.531241Z","iopub.execute_input":"2023-11-03T17:30:48.531958Z","iopub.status.idle":"2023-11-03T17:30:48.538587Z","shell.execute_reply.started":"2023-11-03T17:30:48.531922Z","shell.execute_reply":"2023-11-03T17:30:48.537529Z"}}
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


# %% [markdown]
# # Train - test split

# %% [code]
# train_text, valid_text, train_label, valid_label = train_test_split(df_dct['content'], df_dct['label'], test_size=0.05)

# train_encoding = tokenizer(train_text.to_list(), padding =True)
# valid_encoding = tokenizer(valid_text.to_list(), padding =True)

# train_dataset = DatasetSequenceVLSP(train_encoding,train_label)
# valid_dataset = DatasetSequenceVLSP(valid_encoding,valid_label)

# %% [code] {"execution":{"iopub.status.busy":"2023-11-03T17:30:55.503963Z","iopub.execute_input":"2023-11-03T17:30:55.504962Z","iopub.status.idle":"2023-11-03T17:31:10.684105Z","shell.execute_reply.started":"2023-11-03T17:30:55.504917Z","shell.execute_reply":"2023-11-03T17:31:10.683199Z"}}
# Dac
phobert = AutoModelForSequenceClassification.from_pretrained("vinai/phobert-base-v2", num_labels=9, id2label=id2label,
                                                             label2id=label2id)
tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base-v2")
skf = StratifiedKFold(n_splits=3, shuffle=True)

df_dct = df[["content", "label"]]
df_dct["content"] = df_dct["content"].map(lambda x: segmenter.word_segment(x)[0])

# %% [code] {"execution":{"iopub.status.busy":"2023-11-03T17:31:10.686162Z","iopub.execute_input":"2023-11-03T17:31:10.686950Z","iopub.status.idle":"2023-11-03T17:31:19.745848Z","shell.execute_reply.started":"2023-11-03T17:31:10.686913Z","shell.execute_reply":"2023-11-03T17:31:19.744991Z"}}
train_datasets, valid_datasets = [], []
for train, test in skf.split(df_dct['content'], df_dct['label']):
    train_encoding = tokenizer(df_dct['content'][train].to_list(), padding=True)
    valid_encoding = tokenizer(df_dct['content'][test].to_list(), padding=True)

    train_dataset = DatasetSequenceVLSP(train_encoding, df_dct['label'][train])
    valid_dataset = DatasetSequenceVLSP(valid_encoding, df_dct['label'][test])

    train_datasets.append(train_dataset)
    valid_datasets.append(valid_dataset)

# %% [markdown]
# # Training session

# %% [code] {"execution":{"iopub.status.busy":"2023-11-03T17:31:19.747183Z","iopub.execute_input":"2023-11-03T17:31:19.747498Z","iopub.status.idle":"2023-11-03T17:31:22.199289Z","shell.execute_reply.started":"2023-11-03T17:31:19.747471Z","shell.execute_reply":"2023-11-03T17:31:22.198437Z"}}

f1 = evaluate.load("f1")


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return f1.compute(predictions=predictions, references=labels, average="macro")


# %% [code] {"execution":{"iopub.status.busy":"2023-11-03T17:31:22.201144Z","iopub.execute_input":"2023-11-03T17:31:22.201441Z","iopub.status.idle":"2023-11-03T18:22:14.880570Z","shell.execute_reply.started":"2023-11-03T17:31:22.201415Z","shell.execute_reply":"2023-11-03T18:22:14.879337Z"}}
phobert = AutoModelForSequenceClassification.from_pretrained("vinai/phobert-base-v2", num_labels=9, id2label=id2label,
                                                             label2id=label2id).to("cuda")
tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base-v2")
train_encoding = tokenizer(df_dct['content'].to_list(), padding=True)
train_dataset = DatasetSequenceVLSP(train_encoding, df_dct['label'])

training_args = TrainingArguments(
    output_dir=f"/kaggle/working/output/have9class",
    learning_rate=3e-5,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    num_train_epochs=15,
    evaluation_strategy="epoch",
    save_strategy="no",
    #         save_total_limit = 2,
    #         load_best_model_at_end=True,
)

trainer = Trainer(
    model=phobert,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=valid_dataset,
    compute_metrics=compute_metrics,
    #         optimizers = (optim.Adam(phobert.parameters(),lr=3e-5), None)
)

trainer.train()
trainer.save_model(output_dir=f"/output/models/phobert9class")

# %% [code] {"execution":{"iopub.status.busy":"2023-11-03T18:22:23.861368Z","iopub.execute_input":"2023-11-03T18:22:23.862102Z","iopub.status.idle":"2023-11-03T20:03:13.035275Z","shell.execute_reply.started":"2023-11-03T18:22:23.862071Z","shell.execute_reply":"2023-11-03T20:03:13.034188Z"}}
models = []
for i, (X, y) in enumerate(zip(train_datasets, valid_datasets)):
    phobert = AutoModelForSequenceClassification.from_pretrained("vinai/phobert-base-v2", num_labels=9,
                                                                 id2label=id2label, label2id=label2id).to("cuda")
    tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base-v2")

    training_args = TrainingArguments(
        output_dir=f"/kaggle/working/output/{i}",
        learning_rate=3e-5,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        num_train_epochs=15,
        evaluation_strategy="epoch",
        save_strategy="no",
        #         save_total_limit = 2,
        #         load_best_model_at_end=True,
    )

    trainer = Trainer(
        model=phobert,
        args=training_args,
        train_dataset=X,
        eval_dataset=y,
        compute_metrics=compute_metrics,
        #         optimizers = (optim.Adam(phobert.parameters(),lr=3e-5), None)
    )

    trainer.train()
    trainer.save_model(output_dir=f"/model_output/working/output/{i}")

    models.append(trainer)


# %% [code]
# !zip -r /kaggle/working/models.zip /kaggle/working/output

# %% [code]
# import os
# import subprocess
# from IPython.display import FileLink, display

# def download_file(path, download_file_name):
#     os.chdir('/kaggle/working/')
#     zip_name = f"/kaggle/working/{download_file_name}.zip"
#     command = f"zip {zip_name} {path} -r"
#     result = subprocess.run(command, shell=True, capture_output=True, text=True)
#     if result.returncode != 0:
#         print("Unable to run zip command!")
#         print(result.stderr)
#         return
#     display(FileLink(f'{download_file_name}.zip'))

# download_file('/kaggle/working/models.zip', 'out')

# %% [code]

# %% [code] {"execution":{"iopub.status.busy":"2023-11-03T18:22:14.882490Z","iopub.execute_input":"2023-11-03T18:22:14.882779Z","iopub.status.idle":"2023-11-03T18:22:15.077017Z","shell.execute_reply.started":"2023-11-03T18:22:14.882752Z","shell.execute_reply":"2023-11-03T18:22:15.075608Z"}}
for i, model in enumerate(models):
    model.save_model(output_dir=f"/model_output/working/output/{i}")

# %% [code]
