import pandas as pd
import os
import data_utils
import random
import json
random.seed(25)

''' This block will extract resource for data generaion
-----------------------------------------------------------'''

# Extract data from outer source
json_brands = r'C:\Users\Public\VLSP23-Comparative-Opinion-Mining\data\device and brand\brands.json'
brands = []
with open(json_brands, 'r') as j:
     brands_json = json.loads(j.read())['RECORDS']
for i in range(len(brands_json)):
    brands.append(brands_json[i]['name'])
json_devices = r'C:\Users\Public\VLSP23-Comparative-Opinion-Mining\data\device and brand\devices.json'
devices = []
with open(json_devices, 'r') as j:
     devices_json = json.loads(j.read())['RECORDS']
for i in range(len(devices_json)):
    devices.append(devices_json[i]['name'])
# Data extract from orginal dataset
files_btc = [f for f in os.listdir(r"C:\Users\Public\VLSP23-Comparative-Opinion-Mining\data\public and train data")
             if os.path.isfile(os.path.join(r"C:\Users\Public\VLSP23-Comparative-Opinion-Mining\data\public and train data", f))]
resource = dict()
df_read_txt = pd.DataFrame(columns=["content", "comparative", "subject", "object", "aspect", "predicate", "label", "NER"])
vocab = {'subject':set(),'object':set(),'aspect':dict(),'predicate':dict()}

for i in files_btc:
  df_read_txt = df_read_txt._append(
      data_utils.preprocessing_with_BIO(r"C:\Users\Public\VLSP23-Comparative-Opinion-Mining\data\public and train data" + "/" + i))
  new_source = data_utils.preprocessing(r"C:\Users\Public\VLSP23-Comparative-Opinion-Mining\data\public and train data" + "/" + i, vocab)
  for j in new_source.keys():
      resource[j] = new_source[j]
# Summing up all sources
for i in brands:
    vocab['subject'].add(i)
    vocab['object'].add(i)
for i in devices:
    vocab['subject'].add(i)
    vocab['object'].add(i)

''' Data generation '''
gen_data = dict()
# Generating new sentences
for i in range(1,len(data_utils.type_of_compare)):
    dict_of_sentence = dict()
    for j in resource.keys():
        for k in resource[j]:
            if k['label'] == data_utils.type_of_compare[i]:
                if j not in dict_of_sentence.keys():
                    dict_of_sentence[j] = [k]
                else:
                    dict_of_sentence[j].append(k)
    list_of_pred = []
    for j in vocab['predicate'].keys():
        if data_utils.type_of_compare[i] in vocab['predicate'][j]:
            list_of_pred.append(j)
    list_of_asp = []
    for j in vocab['aspect'].keys():
        if data_utils.type_of_compare[i] in vocab['aspect'][j]:
            list_of_asp.append(j)
    while data_utils.stat[i] <= 5000:
        sentence = random.choice(list(dict_of_sentence.keys()))
        sentence_split = sentence.split()
        list_of_json = dict_of_sentence[sentence]
        new_word = random.choice(list_of_pred)
        new_word = new_word.lower()
        new_sentence = data_utils.update_thing(list_of_json, sentence_split, new_word.split(), gen_data, 'predicate', list_of_json[0]['predicate'])
        if new_sentence is not None:
            sentence = new_sentence
        k = 0
        try:
            while k < len(gen_data[sentence]):
                if len(gen_data[sentence][k]['aspect']) != 0:
                    new_word = random.choice(list_of_asp)
                    new_word = new_word.lower()
                    list_of_json = gen_data[sentence]
                    new_sentence = data_utils.update_thing(list_of_json, sentence.split(), new_word.split(), gen_data, 'aspect', gen_data[sentence][k]['aspect'])
                    if new_sentence is not None:
                        sentence = new_sentence
                k += 1
            k = 0
            while k < len(gen_data[sentence]):
                if len(gen_data[sentence][k]['subject']) != 0:
                    new_word = random.choice(list(vocab['subject']))
                    new_word = new_word.lower()
                    list_of_json = gen_data[sentence]
                    sentence = data_utils.update_thing(list_of_json, sentence.split(), new_word.split(), gen_data, 'subject', gen_data[sentence][k]['subject'])
                    if new_sentence is not None:
                        sentence = new_sentence
                k += 1
            k = 0
            while k < len(gen_data[sentence]):
                if len(gen_data[sentence][k]['object']) != 0:
                    new_word = random.choice(list(vocab['object']))
                    new_word = new_word.lower()
                    list_of_json = gen_data[sentence]
                    sentence = data_utils.update_thing(list_of_json, sentence.split(), new_word.split(), gen_data, 'object', gen_data[sentence][k]['object'])
                    if new_sentence is not None:
                        sentence = new_sentence
                k += 1
        except:
            continue

def write_total_sampling(filename):
    with open(filename, 'w', encoding='utf-8') as f:
        for i in gen_data.keys():
            f.write(i+'\n')
            for j in gen_data[i]:
                f.write(str(j)+'\n')
            f.write('\n')
