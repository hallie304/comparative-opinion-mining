import py_vncorenlp
# %% [code] {"execution":{"iopub.status.busy":"2023-11-03T18:20:08.368019Z","iopub.execute_input":"2023-11-03T18:20:08.3684Z","iopub.status.idle":"2023-11-03T18:20:08.628444Z","shell.execute_reply.started":"2023-11-03T18:20:08.368359Z","shell.execute_reply":"2023-11-03T18:20:08.627363Z"}}
segmenter = py_vncorenlp.VnCoreNLP(annotators=["wseg"], save_dir=r"C:\Users\Duc Bao\PycharmProjects\VLSP All Code\vncorenlp")

# %% [code] {"execution":{"iopub.status.busy":"2023-11-03T18:20:08.631097Z","iopub.execute_input":"2023-11-03T18:20:08.63293Z","iopub.status.idle":"2023-11-03T18:20:20.95681Z","shell.execute_reply.started":"2023-11-03T18:20:08.632895Z","shell.execute_reply":"2023-11-03T18:20:20.955816Z"}}
import torch
from transformers.file_utils import is_tf_available, is_torch_available, is_torch_tpu_available
from transformers import AutoModelForSequenceClassification, AutoTokenizer, TrainingArguments, Trainer, AutoModel
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
import ast
import random
import torch.optim as optim
import pandas as pd


# %% [code] {"execution":{"iopub.status.busy":"2023-11-03T18:20:20.958022Z","iopub.execute_input":"2023-11-03T18:20:20.958558Z","iopub.status.idle":"2023-11-03T18:20:20.968569Z","shell.execute_reply.started":"2023-11-03T18:20:20.958531Z","shell.execute_reply":"2023-11-03T18:20:20.967685Z"}}
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

# %% [code] {"execution":{"iopub.status.busy":"2023-11-03T18:20:20.969854Z","iopub.execute_input":"2023-11-03T18:20:20.970433Z","iopub.status.idle":"2023-11-03T18:20:21.10331Z","shell.execute_reply.started":"2023-11-03T18:20:20.9704Z","shell.execute_reply":"2023-11-03T18:20:21.10228Z"}}

df = pd.read_csv(r'C:\Users\Duc Bao\PycharmProjects\VLSP All Code\data\all_task_ver3_combine_as_pred.csv', index_col=0)
# df = df.drop_duplicates(subset=['content'])

# %% [code] {"execution":{"iopub.status.busy":"2023-11-03T18:20:21.104724Z","iopub.execute_input":"2023-11-03T18:20:21.105078Z","iopub.status.idle":"2023-11-03T18:20:21.113949Z","shell.execute_reply.started":"2023-11-03T18:20:21.105044Z","shell.execute_reply":"2023-11-03T18:20:21.113063Z"}}
df_dct = df[["content", "label"]].sample(frac=1).reset_index(drop=True)

# %% [code] {"execution":{"iopub.status.busy":"2023-11-03T18:20:21.116976Z","iopub.execute_input":"2023-11-03T18:20:21.117321Z","iopub.status.idle":"2023-11-03T18:20:21.122506Z","shell.execute_reply.started":"2023-11-03T18:20:21.117288Z","shell.execute_reply":"2023-11-03T18:20:21.121716Z"}}
id2label = {0: "O", 1: "DIF", 2: "EQL", 3: "SUP+", 4: "SUP-", 5: "SUP", 6: "COM+", 7: "COM-", 8: "COM"}
label2id = {"O": 0, "DIF": 1, "EQL": 2, "SUP+": 3, "SUP-": 4, "SUP": 5, "COM+": 6, "COM-": 7, "COM": 8}

# %% [code] {"execution":{"iopub.status.busy":"2023-11-03T18:20:21.123481Z","iopub.execute_input":"2023-11-03T18:20:21.123773Z","iopub.status.idle":"2023-11-03T18:20:31.430331Z","shell.execute_reply.started":"2023-11-03T18:20:21.12375Z","shell.execute_reply":"2023-11-03T18:20:31.429383Z"}}
phobert = AutoModelForSequenceClassification.from_pretrained("vinai/phobert-base-v2", num_labels=9, id2label=id2label,
                                                             label2id=label2id).to("cuda")
tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base-v2")

# %% [code] {"execution":{"iopub.status.busy":"2023-11-03T18:20:31.433282Z","iopub.execute_input":"2023-11-03T18:20:31.433567Z","iopub.status.idle":"2023-11-03T18:20:48.774801Z","shell.execute_reply.started":"2023-11-03T18:20:31.433541Z","shell.execute_reply":"2023-11-03T18:20:48.773932Z"}}
df_dct["content"] = df_dct["content"].map(lambda x: segmenter.word_segment(x)[0])


# %% [code] {"execution":{"iopub.status.busy":"2023-11-03T18:20:48.775925Z","iopub.execute_input":"2023-11-03T18:20:48.776226Z","iopub.status.idle":"2023-11-03T18:20:48.780364Z","shell.execute_reply.started":"2023-11-03T18:20:48.776201Z","shell.execute_reply":"2023-11-03T18:20:48.779402Z"}}
# train_text, valid_text, train_label, valid_label = train_test_split(df_dct['content'], df_dct['label'], test_size=0.05)

# %% [code] {"execution":{"iopub.status.busy":"2023-11-03T18:20:48.781809Z","iopub.execute_input":"2023-11-03T18:20:48.78208Z","iopub.status.idle":"2023-11-03T18:20:48.793961Z","shell.execute_reply.started":"2023-11-03T18:20:48.782056Z","shell.execute_reply":"2023-11-03T18:20:48.793059Z"}}
# train_encoding = tokenizer(train_text.to_list(), padding =True)
# valid_encoding = tokenizer(valid_text.to_list(), padding =True)

# %% [code] {"execution":{"iopub.status.busy":"2023-11-03T18:20:48.795127Z","iopub.execute_input":"2023-11-03T18:20:48.79546Z","iopub.status.idle":"2023-11-03T18:20:48.806236Z","shell.execute_reply.started":"2023-11-03T18:20:48.795428Z","shell.execute_reply":"2023-11-03T18:20:48.805481Z"}}
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


# %% [code] {"execution":{"iopub.status.busy":"2023-11-03T18:20:48.807377Z","iopub.execute_input":"2023-11-03T18:20:48.807996Z","iopub.status.idle":"2023-11-03T18:20:57.230556Z","shell.execute_reply.started":"2023-11-03T18:20:48.807964Z","shell.execute_reply":"2023-11-03T18:20:57.229779Z"}}
# train_dataset = DatasetSequenceVLSP(train_encoding,train_label)
# valid_dataset = DatasetSequenceVLSP(valid_encoding,valid_label)
skf = StratifiedKFold(n_splits=3, shuffle=True)

train_datasets, valid_datasets = [], []
for train, test in skf.split(df_dct['content'], df_dct['label']):
    train_encoding = tokenizer(df_dct['content'][train].to_list(), padding=True)
    valid_encoding = tokenizer(df_dct['content'][test].to_list(), padding=True)

    train_dataset = DatasetSequenceVLSP(train_encoding, df_dct['label'][train])
    valid_dataset = DatasetSequenceVLSP(valid_encoding, df_dct['label'][test])

    train_datasets.append(train_dataset)
    valid_datasets.append(valid_dataset)

# %% [code] {"execution":{"iopub.status.busy":"2023-11-03T18:20:57.231855Z","iopub.execute_input":"2023-11-03T18:20:57.232296Z","iopub.status.idle":"2023-11-03T18:20:57.238809Z","shell.execute_reply.started":"2023-11-03T18:20:57.232261Z","shell.execute_reply":"2023-11-03T18:20:57.237885Z"}}

# %% [code] {"execution":{"iopub.status.busy":"2023-11-03T18:20:57.239932Z","iopub.execute_input":"2023-11-03T18:20:57.240273Z","iopub.status.idle":"2023-11-03T18:20:57.250546Z","shell.execute_reply.started":"2023-11-03T18:20:57.240241Z","shell.execute_reply":"2023-11-03T18:20:57.249852Z"}}
# !rm -r /kaggle/working/output/checkpoint-286

# %% [code] {"execution":{"iopub.status.busy":"2023-11-03T18:23:41.180966Z","iopub.execute_input":"2023-11-03T18:23:41.181369Z","iopub.status.idle":"2023-11-03T19:14:01.899068Z","shell.execute_reply.started":"2023-11-03T18:23:41.181337Z","shell.execute_reply":"2023-11-03T19:14:01.897883Z"}}
import evaluate

f1 = evaluate.load("f1")


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return f1.compute(predictions=predictions, references=labels, average="macro")


models = []
for i, (X, y) in enumerate(zip(train_datasets, valid_datasets)):
    phobert = AutoModelForSequenceClassification.from_pretrained("vinai/phobert-base-v2", num_labels=9,
                                                                 id2label=id2label, label2id=label2id).to("cuda")
    tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base-v2")
    #     if i == 1:
    #         continue
    training_args = TrainingArguments(
        output_dir=f"Task 3/output/{i}",
        learning_rate=3e-5,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        num_train_epochs=10,
        evaluation_strategy="epoch",
        save_strategy="no",
        save_total_limit=2,
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
    trainer.save_model(output_dir=f"Task 3/output/{i}")

    models.append(trainer)


# %% [code]
