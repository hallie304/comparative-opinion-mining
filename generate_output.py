from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline, DataCollatorWithPadding,AutoModelForSequenceClassification, Trainer, TrainingArguments,AutoTokenizer,AutoModel,AutoConfig, DataCollatorForTokenClassification
from transformers.modeling_outputs import TokenClassifierOutput
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import datasets
from datasets import load_dataset,Dataset,DatasetDict,ClassLabel,Sequence
import itertools
import os

"""# Load Pretrained Models and Tokenizers

## Task 1
"""

tokenizer1 = AutoTokenizer.from_pretrained("vinai/phobert-base-v2")


fold0 = AutoModelForSequenceClassification.from_pretrained(r"task1/0")

fold1 = AutoModelForSequenceClassification.from_pretrained(r"task1/1")

fold2 = AutoModelForSequenceClassification.from_pretrained(r"task1/2")

nlp0 = pipeline("sentiment-analysis", model = fold0, tokenizer = tokenizer1, device = "cuda:0")
nlp1 = pipeline("sentiment-analysis", model = fold1, tokenizer = tokenizer1, device = "cuda:0")
nlp2 = pipeline("sentiment-analysis", model = fold2, tokenizer = tokenizer1, device = "cuda:0")

"""## Task 2"""

id2label = {0: "O", 1: "B-Subject", 2: "I-Subject", 3: "B-Object", 4: "I-Object", 5: "B-Aspect", 6: "B-Aspect",
            7: "B-Predicate", 8: "I-Predicate"}

label2id = {"O": 0 , "B-Subject": 1, "I-Subject": 2 , "B-Object": 3, "I-Object": 4, "B-Aspect": 5, "B-Aspect": 6,
            "B-Predicate": 7, "I-Predicate": 8}

electra_tokenizer = AutoTokenizer.from_pretrained("NlpHUST/ner-vietnamese-electra-base")
electra_model = AutoModelForTokenClassification.from_pretrained(r"task2/electra").to("cuda")

phobert_tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base-v2")
phobert_model = AutoModelForTokenClassification.from_pretrained(r"task2/phobert").to("cuda")

multi_tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")
multi_model = AutoModelForTokenClassification.from_pretrained(r"task2/bert").to("cuda")


def align_tokens(text, tokenizer):
  word_dict = {}
  text_list = text.split(" ")
  for i in range(len(text_list)):
    tokenized_word = tokenizer.tokenize(text_list[i])
    word_dict.update({i: []})
    for x in range(len(tokenized_word)):
      if i == 0:
        word_dict[i].append(x+1)
      else:
        word_dict[i].append(x + max(word_dict[i-1]) + 1)
  return word_dict

def reduce_logits_size(text, tokenizer, model):
  tensor_list = []
  word_dict = align_tokens(text, tokenizer)
  text_input = tokenizer(text, return_tensors="pt").to("cuda")
  with torch.no_grad():
    logits = model(**text_input).logits.cpu()
  for i in word_dict:
    sum = torch.Tensor([0, 0, 0, 0, 0, 0, 0, 0, 0])
    for x in word_dict[i]:
      sum += logits[0][x]
    final_sum = sum / len(word_dict[i])
    tensor_list.append(final_sum)
  return torch.stack(tensor_list).unsqueeze(0)


def combine_model_logits(text):
  electra_output = reduce_logits_size(text, electra_tokenizer, electra_model)
  phobert_output = reduce_logits_size(text, phobert_tokenizer, phobert_model)
  multi_output = reduce_logits_size(text, multi_tokenizer, multi_model)
  final_output = 0.3 * electra_output + 0.2 * phobert_output + 0.5 * multi_output

  return final_output

def infer_logits(text):
  combined_output = combine_model_logits(text)
  predictions = torch.argmax(combined_output, dim = 2)
  predicted_token_list = [id2label[t.item()] for t in predictions[0]]
  text_list = text.split(" ")
  outputs = []
  for i in range(len(text_list)):
    outputs.append({"text": text_list[i], "value": predicted_token_list[i]})
  return outputs

"""## Task 3"""

from transformers import AutoTokenizer
tokenizer1 = AutoTokenizer.from_pretrained("vinai/phobert-base-v2")
fold3 = AutoModelForSequenceClassification.from_pretrained(r"task3/0")
fold4 = AutoModelForSequenceClassification.from_pretrained(r"task3/1")
fold5 = AutoModelForSequenceClassification.from_pretrained(r"task3/2")

nlp3 = pipeline("sentiment-analysis", model = fold3, tokenizer = tokenizer1, device = "cuda:0")
nlp4 = pipeline("sentiment-analysis", model = fold4, tokenizer = tokenizer1, device = "cuda:0")
nlp5 = pipeline("sentiment-analysis", model = fold5, tokenizer = tokenizer1, device = "cuda:0")

"""# Post Processing"""

def post_process(tokens, entities):
    post_process_text = ""
    entity = []
    idx = 0
    for token in tokens:
        if not token.startswith("##"):
            post_process_text += " "
            entity.append(entities[idx])
        post_process_text += token.lstrip("##")
        idx += 1
    return post_process_text.lstrip(), entity

def subjectObjectEqual(input):
    phrases_to_check = ["Cả", "cả", "Cả hai", "Cả 2", "cả hai", "cả 2", "2 chiếc",
                        "2 máy", "2 thiết", "2 điện thoại", "Hai chiếc", "Hai máy",
                        "Hai thiết", "Hai điện thoại", "hai chiếc", "hai máy", "hai thiết",
                        "hai điện thoại", "hai sản phẩm", "bộ đôi", "mỗi điện thoại", "hai mẫu",
                       "2 mẫu", "Hai mẫu"]

    if not input['subject'] and input['object']:
        objects = input['object']
        object_phrase = ' '.join(obj.split("&&")[1] for obj in objects)
        if any(phrase in object_phrase for phrase in phrases_to_check):
            input['subject'] = objects
    elif not input['object'] and input['subject']:
        subjects = input['subject']
        subject_phrase = ' '.join(obj.split("&&")[1] for obj in subjects)
        if any(phrase in subject_phrase for phrase in phrases_to_check):
            input['object'] = subjects
    return input

def combinedBootstrap(sentence, model1, model2, model3):
    outputs = []
    outputs.append(model1(sentence))
    outputs.append(model2(sentence))
    outputs.append(model3(sentence))

    combined_output = {}

    for output in outputs:
        label = output[0]['label']
        score = output[0]['score']

        if label in combined_output:
            combined_output[label].append(score)
        else:
            combined_output[label] = [score]

    for label, scores in combined_output.items():
        average_score = sum(scores) / len(scores)
        combined_output[label] = average_score

    if len(combined_output) > 1:
        max_label = max(combined_output, key=combined_output.get)
        combined_output = [{'label': max_label, 'score': combined_output[max_label]}]
    else:
        max_label = label
        combined_output = [{'label': max_label, 'score': combined_output[max_label]}]
    return combined_output


def generate_all_possible_quadruple(sentence, input_quintuple):
    lists_to_split = ["subject", "object", "aspect", "predicate"]

    split_quintuples = {list_name: [] for list_name in lists_to_split}
    split_quintuples = {"subject": [], "object": [], "aspect": [], "predicate": [], "label": ""}
    results = []

    for list_name in lists_to_split:
        current_list = input_quintuple[list_name]
        current_group = []
        current_word = ""

        for item in current_list:
            index = int(item.split("&&")[0])

            if not current_group or index == current_group[-1] + 1:
                current_group.append(index)
            else:
                split_quintuples[list_name].append([s for s in input_quintuple[list_name] if int(s.split("&&")[0]) in current_group])
                current_group = [index]

        split_quintuples[list_name].append([s for s in input_quintuple[list_name] if int(s.split("&&")[0]) in current_group])

    subjects = split_quintuples['subject']
    objects = split_quintuples['object']
    aspects = split_quintuples['aspect']
    predicates = split_quintuples['predicate']

    combinations = itertools.product(*[subjects, objects, aspects, predicates])

    for combo in combinations:
        quintuple_combination = {"subject": combo[0], "object": combo[1], "aspect": combo[2], "predicate": combo[3]}
        result = " ".join([
                sentence.replace("\n", ""),
                "|",
                ",".join([" ".join(i.split("&&")[1] for i in quintuple_combination["subject"]),
                          " ".join(i.split("&&")[1] for i in quintuple_combination["object"]),
                          " ".join(i.split("&&")[1] for i in quintuple_combination["aspect"]),
                          " ".join(i.split("&&")[1] for i in quintuple_combination["predicate"])
                          ])])
        quintuple_combination["input_task3"] = result
        results.append(quintuple_combination)
    return results

"""# Output"""

os.mkdir("raw_output")
input_directory = 'data/private_test'
output_directory = 'raw_output'
for file_number in range(1, 37):
    input_file_name = f'test_{str(file_number).zfill(4)}.txt'
    input_file_path = os.path.join(input_directory, input_file_name)
    output_file_path = os.path.join(output_directory, input_file_name)

    with open(output_file_path, 'w') as output_file:
        with open(input_file_path, 'r') as input_file:
            lines = input_file.readlines()
        input_sentence = []
        for line in lines:
            sentences = line.split('\t')
            if len(sentences) >= 2:
                output_file.write(sentences[1])


import json
import os
for file_number in range(1, 37):
    input_file_name = f'test_{str(file_number).zfill(4)}.txt'
    input_file_path = os.path.join(input_directory, input_file_name)
    output_file_path = os.path.join(output_directory, input_file_name)
    with open(output_file_path, 'r') as input_file:
        lines = input_file.readlines()
    print(input_file_name)

    with open(output_file_path, 'w') as output_file:
        comparative = []
        task2 = []
        preference = []
        # Task 1: Identify comparative sentence, two classes: Comparative and Non Comparative
        for line in lines:
            comparative.append(combinedBootstrap(line, nlp0, nlp1, nlp2))
        for i, line in enumerate(lines):
            label = comparative[i][0]['label']
            if label != 'None':
                task2.append(line)
        ner = []
        # Task 2: NER
        for sentence in task2:
            sentence = sentence.replace(' \u200b\u200b ', ' ')
            sentence = " ".join(sentence.split())
            ner.append(infer_logits(sentence))
        # Posprocessing NER result
        ner_post_processed = []
        for token_list in ner:
            tokens = [token['text'] for token in token_list]
            entities = [token['value'] for token in token_list]

            post_processed_text,entity = post_process(tokens,entities)
            post_processed_words = post_processed_text.split()

            result_list = [{'entity': entity[i], 'index': i + 1, 'word': post_processed_words[i]} for i in range(len(post_processed_words))]
            ner_post_processed.append(result_list)

        idx = 0
        # Append quintuple
        for i, line in enumerate(lines):
            label = comparative[i][0]['label']
            if label != 'None':
                quintuple = {"subject": [], "object": [], "aspect": [], "predicate": [], "label": ""}
                for tag in ner_post_processed[idx]:
                    entity = tag['entity']
                    index = tag['index']
                    word = tag['word']
                    if entity in ['B-Subject', 'I-Subject']:
                        quintuple['subject'].append(f"{index}&&{word}")
                    elif entity in ['B-Object', 'I-Object']:
                        quintuple['object'].append(f"{index}&&{word}")
                    elif entity in ['B-Aspect', 'I-Aspect']:
                        quintuple['aspect'].append(f"{index}&&{word}")
                    elif entity in ['B-Predicate', 'I-Predicate']:
                        quintuple['predicate'].append(f"{index}&&{word}")
                # Generate_process all possible quadruple for input of task 3
                results = generate_all_possible_quadruple(line, quintuple)
                temp = []
                for i in range(len(results)):
                  # Task 3: comparative type identification
                    preference = combinedBootstrap(results[i]['input_task3'], nlp3, nlp4, nlp5)
                    if preference[0]['label'] != "O":
                        results[i]['label'] = preference[0]['label']
                        results[i].pop('input_task3')
                for item in results:
                    if ('input_task3' in item) or (not item['predicate']):
                        # print(item)
                        results.remove(item)
                final_quintuples = []
                # Handle the case where subject is equal
                for input in results:
                    if input:
                        final_quintuples.append(subjectObjectEqual(input))

                modified_line = f"{line}"

                for quintuple in final_quintuples:
                    if quintuple:
                        modified_line += f"{json.dumps(quintuple, ensure_ascii=False)}\n"
                modified_line += "\n"
                idx += 1
            else:
                modified_line = f"{line}\n"

            output_file.write(modified_line)