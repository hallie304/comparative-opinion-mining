import pandas as pd
import os
from data_utils import sentence_with_corresponding_quintuples
import random
import json
random.seed(25)

''' This block will extract resource for data generaion
-----------------------------------------------------------'''

# Extract data from outer source
json_brands = '/content/drive/MyDrive/Gen data/brands.json'
brands = []
with open(json_brands, 'r') as j:
     brands_json = json.loads(j.read())['RECORDS']
for i in range(len(brands_json)):
    brands.append(brands_json[i]['name'])
json_devices = '/content/drive/MyDrive/Gen data/devices.json'
devices = []
with open(json_devices, 'r') as j:
     devices_json = json.loads(j.read())['RECORDS']
for i in range(len(devices_json)):
    devices.append(devices_json[i]['name'])

# Data extract from orginal dataset
files_btc = [f for f in os.listdir("/content/drive/MyDrive/Gen data/Data_btc") if os.path.isfile(os.path.join("/content/drive/MyDrive/Gen data/Data_btc", f))]
type_of_compare = ["DIF", "EQL", "SUP+", "SUP-", "SUP", "COM+", "COM-", "COM"]
resource = dict()
vocab = {'subject':set(),'object':set(),'aspect':dict(),'predicate':dict()}
stat = [0,0,0,0,0,0,0,0]
# Preprocessing for extraction only
def preprocessing(filename):
    data_row = []
    labeled_row = dict()
    data = []
    with (open('/content/drive/MyDrive/Gen data/Data_btc/' + filename, 'r', encoding="utf-8") as f):
        is_drop = False
        row = f.readline()
        while row != "":
            signal = ""
            split_row = row.split("\t")
            if len(split_row) > 1:
                r = split_row[1]
                if len(split_row) > 2:
                    signal = split_row[2]
            else:
                r = row
            if 'origin' in signal:
                is_drop = True
                row = f.readline()
                signal = ""
                continue
            elif 'quintuple' in signal:
                is_drop = False
                signal = ""
            if is_drop:
                row = f.readline()
                continue
            if '[origin' in r or '[quintuple]' in r or '[spellquintuple]' in r:
                r = r.replace('[origin]','')
                r = r.replace('[quintuple]','')
                r = r.replace('[spellquintuple]','')
                r = r.replace('[original]','')
            r = r.replace("\n", "")
            data_row.append(r)
            row = f.readline()
    f.close()
    del f
    j = 0
    for i in range(len(data_row) - 1, -1, -1):
        if data_row[i] == "":
            data_row.remove(data_row[i])
    labeled_row = sentence_with_corresponding_quintuples(data_row)
    for i in labeled_row.keys():
        for j in labeled_row[i]:
            subject = j['subject']
            object = j['object']
            aspect = j['aspect']
            predicate = j['predicate']
            s = ["" for k in range(len(subject))]
            o = ["" for k in range(len(object))]
            a = ["" for k in range(len(aspect))]
            p = ["" for k in range(len(predicate))]
            if len(subject) != 0:
                for k in range(len(subject)):
                    s[k] = subject[k][subject[k].rfind('&')+1:]
            if len(object) != 0:
                for k in range(len(object)):
                    o[k] = object[k][object[k].rfind('&')+1:]
            if len(aspect) != 0:
                for k in range(len(aspect)):
                    a[k] = aspect[k][aspect[k].rfind('&')+1:]
            if len(predicate) != 0:
                for k in range(len(predicate)):
                    p[k] = predicate[k][predicate[k].rfind('&')+1:]
            vocab['subject'].add(" ".join(s))
            vocab['object'].add(" ".join(o))
            joined_aspect = " ".join(a)
            joined_predicate = " ".join(p)
            if joined_aspect not in vocab['aspect'].keys():
                vocab['aspect'][joined_aspect] = set()
                vocab['aspect'][joined_aspect].add(j['label'])
            else:
                vocab['aspect'][joined_aspect].add(j['label'])
            if joined_predicate not in vocab['predicate'].keys():
                vocab['predicate'][joined_predicate] = set()
                vocab['predicate'][joined_predicate].add(j['label'])
            else:
                vocab['predicate'][joined_predicate].add(j['label'])
            stat[type_of_compare.index(j['label'])] += 1
    return labeled_row

df_read_txt = pd.DataFrame(columns=["content", "comparative", "subject", "object", "aspect", "predicate", "label", "NER"])

for i in files_btc:
  df_read_txt = df_read_txt._append(preprocessing('/content/drive/MyDrive/Gen data/Data_ver1/'+i))
  new_source = preprocessing(i)
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

# Preparation
gen_data = dict()
generror = []
stat=[0,0,0,0,0,0,0,0]
# Extract position
def extract_pos(lst):
    return int(lst[0][:lst[0].find('&')]),int(lst[-1][:lst[-1].find('&')])

# Re-allocate position
def reposition(element,diff):
    return str(int(element[:element.find('&')])+diff)+element[element.find('&'):]

# Rewrite the list in each components
def rewrite(thing,other,new_word_split,k,origin):
    new_obj_list = []
    for o in k[other]:
        new_obj_list.append(reposition(o,len(new_word_split)-len(origin)))
    return new_obj_list

# Join the words from list in the components to a full word
def join_word(lst):
    n_lst = []
    for i in lst:
        n_lst.append(i[i.rfind('&')+1:])
    return ' '.join(n_lst)

# Return the new upsampling sentence with new index in quintuples
def update_thing(list_of_json,sentence_split,new_word_split,gen_data,thing, origin):
    list_of_thing = ['subject','aspect','object','predicate']
    for k in list_of_json:
        if k[thing] == origin:
            position = extract_pos(origin)
            new_sentence_split = sentence_split[:position[0]-1] + new_word_split + sentence_split[position[1]:]
            for l in range(len(new_word_split)):
                new_word_split[l] = str(position[0]+l)+'&&'+new_word_split[l]
            break
    new_sentence = " ".join(new_sentence_split)
    s_split = new_sentence.split()
    if len(s_split) < 4:
        return
    new_list_of_json = []
    for k in list_of_json:
        new_dict = {"subject": k["subject"], "object": k["object"], "aspect": k["aspect"], "predicate": k['predicate'], 'label':k['label']}
        if k[thing] == origin:
            new_dict[thing] = new_word_split
        n_sub = s_sub = ''
        n_ob = s_ob = ''
        n_as = s_asp = ''
        n_pre = s_pred = ''
        if len(k['subject']) != 0:
            if 'subject' != thing:
                thing_pos = extract_pos(k['subject'])
                if (thing_pos[0] >= position[0] and thing_pos[1] <= position[1]) or \
                   (thing_pos[0] <= position[1] and thing_pos[0] >= position[0] and thing_pos[1] > position[1]) or \
                   (thing_pos[1] <= position[1] and thing_pos[1] >= position[0] and thing_pos[0] < position[0]):
                    new_dict['subject'] = []
                elif thing_pos[0] > position[1]:
                    new_dict['subject'] = rewrite(thing,'subject',new_word_split,k,origin)
                    n_sub = join_word(new_dict['subject'])
                    pos_sub = extract_pos(new_dict['subject'])
                    s_sub = ' '.join(s_split[pos_sub[0]-1:pos_sub[1]])
                elif thing_pos[0] <= position[0] and thing_pos[1] >= position[1]:
                    new_subject_list = []
                    for l in k['subject']:
                        new_subject_list.append(l)
                        if int(l[:l.find('&')]) == position[0]:
                            break
                    for l in range(1,len(new_word_split)):
                        new_subject_list.append(new_word_split[l])
                    original_subject_list = k['subject'][position[0]-thing_pos[0]:position[1] - thing_pos[1]]
                    for l in range(len(k['subject'])+position[1] - thing_pos[1],len(k['subject'])):
                        new_subject_list.append(reposition(k['subject'][l],len(new_word_split)-len(original_subject_list)))
                    new_dict['subject'] = new_subject_list
        if len(k['aspect']) != 0:
            if 'aspect' != thing:
                thing_pos = extract_pos(k['aspect'])
                if (thing_pos[0] >= position[0] and thing_pos[1] <= position[1]) or \
                   (thing_pos[0] <= position[1] and thing_pos[0] >= position[0] and thing_pos[1] > position[1]) or \
                   (thing_pos[1] <= position[1] and thing_pos[1] >= position[0] and thing_pos[0] < position[0]):
                    new_dict['aspect'] = []
                elif thing_pos[0] > position[1]:
                    new_dict['aspect'] = rewrite(thing,'aspect',new_word_split,k,origin)
                    n_as = join_word(new_dict['aspect'])
                    pos_asp = extract_pos(new_dict['aspect'])
                    s_asp = ' '.join(s_split[pos_asp[0]-1:pos_asp[1]])
                elif thing_pos[0] <= position[0] and thing_pos[1] >= position[1]:
                    new_subject_list = []
                    for l in k['aspect']:
                        new_subject_list.append(l)
                        if int(l[:l.find('&')]) == position[0]:
                            break
                    for l in range(1,len(new_word_split)):
                        new_subject_list.append(new_word_split[l])
                    original_subject_list = k['aspect'][position[0]-thing_pos[0]:position[1] - thing_pos[1]]
                    for l in range(len(k['aspect'])+position[1] - thing_pos[1],len(k['aspect'])):
                        new_subject_list.append(reposition(k['aspect'][l],len(new_word_split)-len(original_subject_list)))
                    new_dict['aspect'] = new_subject_list
        if len(k['object']) != 0:
            if 'object' != thing:
                thing_pos = extract_pos(k['object'])
                if (thing_pos[0] >= position[0] and thing_pos[1] <= position[1]) or \
                   (thing_pos[0] <= position[1] and thing_pos[0] >= position[0] and thing_pos[1] > position[1]) or \
                   (thing_pos[1] <= position[1] and thing_pos[1] >= position[0] and thing_pos[0] < position[0]):
                    new_dict['object'] = []
                elif thing_pos[0] > position[1]:
                    new_dict['object'] = rewrite(thing,'object',new_word_split,k,origin)
                    n_ob = join_word(new_dict['object'])
                    pos_obj = extract_pos(new_dict['object'])
                    s_ob = ' '.join(s_split[pos_obj[0]-1:pos_obj[1]])
                elif thing_pos[0] <= position[0] and thing_pos[1] >= position[1]:
                    new_subject_list = []
                    for l in k['object']:
                        new_subject_list.append(l)
                        if int(l[:l.find('&')]) == position[0]:
                            break
                    for l in range(1,len(new_word_split)):
                        new_subject_list.append(new_word_split[l])
                    original_subject_list = k['object'][position[0]-thing_pos[0]:position[1] - thing_pos[1]]
                    for l in range(len(k['object'])+position[1] - thing_pos[1],len(k['object'])):
                        new_subject_list.append(reposition(k['object'][l],len(new_word_split)-len(original_subject_list)))
                    new_dict['object'] = new_subject_list
        if len(k['predicate']) != 0:
            if 'predicate' != thing:
                thing_pos = extract_pos(k['predicate'])
                if (thing_pos[0] >= position[0] and thing_pos[1] <= position[1]) or \
                   (thing_pos[0] <= position[1] and thing_pos[0] >= position[0] and thing_pos[1] > position[1]) or \
                   (thing_pos[1] <= position[1] and thing_pos[1] >= position[0] and thing_pos[0] < position[0]):
                    new_sentence = ' '.join(sentence_split)
                    return
                elif thing_pos[0] > position[1]:
                    new_dict['predicate'] = rewrite(thing,'predicate',new_word_split,k,origin)
                    n_pre = join_word(new_dict['predicate'])
                    pos_pred = extract_pos(new_dict['predicate'])
                    s_pred = ' '.join(s_split[pos_pred[0]-1:pos_pred[1]])
                elif thing_pos[0] <= position[0] and thing_pos[1] >= position[1]:
                    new_subject_list = []
                    for l in k['predicate']:
                        new_subject_list.append(l)
                        if int(l[:l.find('&')]) == position[0]:
                            break
                    for l in range(1,len(new_word_split)):
                        new_subject_list.append(new_word_split[l])
                    original_subject_list = k['predicate'][position[0]-thing_pos[0]:position[1] - thing_pos[1]]
                    for l in range(len(k['predicate'])+position[1] - thing_pos[1],len(k['predicate'])):
                        new_subject_list.append(reposition(k['predicate'][l],len(new_word_split)-len(original_subject_list)))
                    new_dict['predicate'] = new_subject_list
        if n_sub == s_sub and n_ob == s_ob and n_as == s_asp and n_pre == s_pred:
            new_list_of_json.append(new_dict)
        else:
            generror.append(new_sentence+'\n'+new_dict.__str__())
            return
    if new_sentence not in gen_data.keys():
        gen_data[new_sentence] = []
        for i in range(len(new_list_of_json)):
            if new_list_of_json[i] not in gen_data[new_sentence]:
                gen_data[new_sentence].append(new_list_of_json[i])
        stat[type_of_compare.index(new_list_of_json[i]['label'])] += 1
        return new_sentence

# Generating new sentences
for i in range(len(type_of_compare)):
    dict_of_sentence = dict()
    for j in resource.keys():
        for k in resource[j]:
            if k['label'] == type_of_compare[i]:
                if j not in dict_of_sentence.keys():
                    dict_of_sentence[j] = [k]
                else:
                    dict_of_sentence[j].append(k)
    list_of_pred = []
    for j in vocab['predicate'].keys():
        if type_of_compare[i] in vocab['predicate'][j]:
            list_of_pred.append(j)
    list_of_asp = []
    for j in vocab['aspect'].keys():
        if type_of_compare[i] in vocab['aspect'][j]:
            list_of_asp.append(j)
    while stat[i] <= 5000:
        sentence = random.choice(list(dict_of_sentence.keys()))
        sentence_split = sentence.split()
        list_of_json = dict_of_sentence[sentence]
        new_word = random.choice(list_of_pred)
        new_word = new_word.lower()
        new_sentence = update_thing(list_of_json, sentence_split, new_word.split(), gen_data, 'predicate',list_of_json[0]['predicate'])
        if new_sentence is not None:
            sentence = new_sentence
        k = 0
        try:
            while k < len(gen_data[sentence]):
                if len(gen_data[sentence][k]['aspect']) != 0:
                    new_word = random.choice(list_of_asp)
                    new_word = new_word.lower()
                    list_of_json = gen_data[sentence]
                    new_sentence = update_thing(list_of_json, sentence.split(), new_word.split(),gen_data,'aspect', gen_data[sentence][k]['aspect'])
                    if new_sentence is not None:
                        sentence = new_sentence
                k += 1
            k = 0
            while k < len(gen_data[sentence]):
                if len(gen_data[sentence][k]['subject']) != 0:
                    new_word = random.choice(list(vocab['subject']))
                    new_word = new_word.lower()
                    list_of_json = gen_data[sentence]
                    sentence = update_thing(list_of_json, sentence.split(), new_word.split(),gen_data,'subject',gen_data[sentence][k]['subject'])
                    if new_sentence is not None:
                        sentence = new_sentence
                k += 1
            k = 0
            while k < len(gen_data[sentence]):
                if len(gen_data[sentence][k]['object']) != 0:
                    new_word = random.choice(list(vocab['object']))
                    new_word = new_word.lower()
                    list_of_json = gen_data[sentence]
                    sentence = update_thing(list_of_json, sentence.split(), new_word.split(),gen_data,'object',gen_data[sentence][k]['object'])
                    if new_sentence is not None:
                        sentence = new_sentence
                k += 1
        except:
            continue