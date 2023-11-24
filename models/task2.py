from transformers import DataCollatorWithPadding,AutoModelForTokenClassification, Trainer, TrainingArguments,AutoTokenizer,AutoModel,AutoConfig, DataCollatorForTokenClassification
import torch
import torch.nn as nn
import pandas as pd
import datasets
from datasets import load_dataset,Dataset,DatasetDict,ClassLabel,Sequence
import copy
import ast
import numpy as np
from datasets import load_metric

def task2_train(data_path, output_path, model_path):
    df = pd.read_csv(data_path)
    df = df[df["comparative"] == 1]
    df = df[(df["NER"] != "0")&(df["NER"] != "[]")]
    df["NER"] = "[" + df["NER"] + "]"
    # Expand NER
    rows = []
    for id, row in df.iterrows():
        a = copy.copy(row)
        for ner_tag_combination in ast.literal_eval(row["NER"]):
            a["NER"] = ner_tag_combination
            rows.append(a)
    df = pd.DataFrame(rows)
    df['content'] = df['content'].str.replace(r"[\[\]']", '', regex=True)
    df['content'] = df['content'].apply(lambda x: x.split(" "))

    # Explode the label lists into separate rows
    df.rename(columns={'content': 'tokens', 'NER': 'ner_tags'}, inplace=True)
    df = df.drop(['label'], axis=1).sample(frac=1).reset_index(drop=True)


    data = Dataset.from_pandas(df)
    class_label = ["O", "B-Subject", "I-Subject", "B-Object", "I-Object", "B-Aspect", "I-Aspect", "B-Predicate",
                   "I-Predicate"]
    data = data.cast_column("ner_tags", datasets.Sequence(datasets.ClassLabel(names=class_label)))

    train_testvalid = data.train_test_split(test_size=0.2, seed=15)
    dataset = DatasetDict({
        'train': data,
        'validation': train_testvalid['test']
    })

    label_names = dataset["train"].features["ner_tags"].feature.names

    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast = True)

    def tokenize_adjust_labels(all_samples_per_split):
      tokenized_samples = tokenizer.batch_encode_plus(all_samples_per_split["tokens"], is_split_into_words=True, truncation=True, max_length=256)
      total_adjusted_labels = []
      for k in range(0, len(tokenized_samples["input_ids"])):
        prev_wid = -1
        word_ids_list = tokenized_samples.word_ids(batch_index=k)
        existing_label_ids = all_samples_per_split["ner_tags"][k]
        i = -1
        adjusted_label_ids = []

        for wid in word_ids_list:
          if(wid is None):
            adjusted_label_ids.append(-100)
          elif(wid!=prev_wid):
            i = i + 1
            adjusted_label_ids.append(existing_label_ids[i])
            prev_wid = wid
          else:
            label_name = label_names[existing_label_ids[i]]
            adjusted_label_ids.append(existing_label_ids[i])

        total_adjusted_labels.append(adjusted_label_ids)
      tokenized_samples["labels"] = total_adjusted_labels
      return tokenized_samples

    tokenized_dataset = dataset.map(tokenize_adjust_labels, batched=True)

    data_collator = DataCollatorForTokenClassification(tokenizer)

    metric = load_metric("seqeval")

    def compute_metrics(p):
        predictions, labels = p
        predictions = np.argmax(predictions, axis=2)

        # Remove ignored index (special tokens)
        true_predictions = [
            [label_names[p] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        true_labels = [
            [label_names[l] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]

        results = metric.compute(predictions=true_predictions, references=true_labels)
        flattened_results = {
            "overall_precision": results["overall_precision"],
            "overall_recall": results["overall_recall"],
            "overall_f1": results["overall_f1"],
            "overall_accuracy": results["overall_accuracy"],
        }
        for k in results.keys():
          if(k not in flattened_results.keys()):
            flattened_results[k+"_f1"]=results[k]["f1"]

        return flattened_results

    id2label = {0: "O", 1: "B-Subject", 2: "I-Subject", 3: "B-Object", 4: "I-Object", 5: "B-Aspect", 6: "B-Aspect",
                7: "B-Predicate", 8: "I-Predicate"}

    label2id = {"O": 0 , "B-Subject": 1, "I-Subject": 2 , "B-Object": 3, "I-Object": 4, "B-Aspect": 5, "B-Aspect": 6,
                "B-Predicate": 7, "I-Predicate": 8}

    model = AutoModelForTokenClassification.from_pretrained(model_path, num_labels=len(label_names), id2label=id2label, label2id=label2id)

    batch_size = 32
    training_args = TrainingArguments(
    output_dir="results",
    num_train_epochs=15,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    evaluation_strategy="epoch",
    disable_tqdm=False,
    logging_steps=len(tokenized_dataset['train']) // batch_size)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )

    trainer.train()

    trainer.save_model(output_path)