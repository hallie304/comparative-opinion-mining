import pandas as pd
import re
import json
import random


# Preparation
stat = [0,0,0,0,0,0,0,0,0]
type_of_compare = ["Non","DIF", "EQL", "SUP+", "SUP-", "SUP", "COM+", "COM-", "COM"]
generror = []
# Split sentence with quintuples and entitle the quintuples to corresponding sentence
def sentence_with_corresponding_quintuples(data_row):
    j = 0
    labeled_row = dict()
    for i in range(len(data_row) - 1):
        if (re.match("^\{\"subject\": \[", data_row[i + 1]) and re.match("^\{\"subject\": \[", data_row[i])) or (re.match("^\{\'subject\': \[", data_row[i + 1]) and re.match("^\{\'subject\': \[", data_row[i])):
            row = eval(data_row[i + 1])
            try:
                labeled_row[data_row[j]].append(row)
            except Exception as e:
                    print(e,data_row[i+1])
        elif re.match("^\{\"subject\": \[", data_row[i + 1]) or re.match("^\{\'subject\': \[", data_row[i + 1]):
            if data_row[i] not in labeled_row.keys():
                j = i
                row = eval(data_row[i + 1])
                labeled_row[data_row[j]] = [row]
            else:
                try:
                    row = eval(data_row[i + 1])
                    labeled_row[data_row[i]].append(row)
                except Exception as e:
                    print(e,data_row[i+1])
    return labeled_row

def preprocessing(filename, vocab):
    data_row = []
    labeled_row = dict()
    data = []
    with (open(filename, 'r', encoding="utf-8") as f):
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

# Main preprocess method
def preprocessing_with_BIO(filename):
    data_frame = {"content": [], "comparative": [], "subject": [], "object": [], "aspect": [], "predicate": [], "label": [], "NER": []}
    data_row = []
    data = []
    error = False
    # Read file
    with (open(filename, 'r', encoding="utf-8") as f):
        is_drop = False
        row = f.readline()
        while row != "":
            signal = ""
            split_row = row.split("\t")
            if len(split_row) > 1:
                r = split_row[1]
            else:
                r = row
            r = r.replace("\n", "")
            data_row.append(r)
            row = f.readline()
    f.close()
    # Remove empty row
    for i in range(len(data_row) - 1, -1, -1):
        if data_row[i] == "":
            data_row.remove(data_row[i])
    # Extract quintuples
    labeled_row = sentence_with_corresponding_quintuples(data_row)
    # Extract sentence
    for i in data_row:
        if re.match("^\{\"subject\": \[", i) is None and re.match("^\{\'subject\': \[", i) is None:
            data.append(i)
    # Process the data into dataframe
    for i in data:
        sentence_label = []
        key = 0
        subj = []
        obj = []
        asp = []
        pred = []
        lab = []
        w = i.split(" ")
        label = [0 for k in range(len(w))]
        if i in labeled_row.keys():
            key = 1
            for k in range(len(labeled_row.get(i))):
                label = [0 for k in range(len(w))]
                list_dct = labeled_row.get(i)
                if isinstance(list_dct[k]['subject'],str):
                        continue
                for j in list_dct[k].keys():
                    arr = list_dct[k].get(j)
                    consecutive = []
                    if j == "label":
                        continue
                    for e in range(len(arr)):
                        consecutive.append(int(arr[e][:arr[e].find("&")]))
                        arr[e] = re.sub("^[0-9]+&&", "", arr[e])
                    s = " ".join(arr)
                    list_dct[k][j] = s
                    for q in range(len(consecutive)):
                        if j == "subject":
                            try:
                                if label[consecutive[q]-1] == 0:
                                    if q == 0:
                                        label[consecutive[q] - 1] = 1
                                    else:
                                        label[consecutive[q] - 1] = 2
                            except:
                                error = True
                                break
                        elif j == "object":
                            try:
                                if label[consecutive[q]-1] == 0:
                                    if q == 0:
                                        label[consecutive[q] - 1] = 3
                                    else:
                                        label[consecutive[q] - 1] = 4
                            except:
                                error = True
                                break
                        elif j == "aspect":
                            try:
                                if label[consecutive[q]-1] == 0:
                                    if q == 0:
                                        label[consecutive[q] - 1] = 5
                                    else:
                                        label[consecutive[q] - 1] = 6
                            except:
                                error = True
                                break
                        elif j == "predicate":
                            try:
                                if label[consecutive[q]-1] == 0:
                                    if q == 0:
                                        label[consecutive[q] - 1] = 7
                                    else:
                                        label[consecutive[q] - 1] = 8
                            except:
                                error = True
                                break
                    try:
                        word_list = []
                        for m in consecutive:
                            word_list.append(w[m-1])
                        if ' '.join(word_list) == s:
                            pass
                        else:
                            error = True
                            break
                    except:
                        error = True
                        break
                if error:
                    break
                sentence_label.append(label)
                subj.append(list_dct[k]["subject"])
                obj.append(list_dct[k]["object"])
                pred.append(list_dct[k]["predicate"])
                asp.append(list_dct[k]["aspect"])
                lab.append(type_of_compare.index(list_dct[k]["label"]))
                stat[type_of_compare.index(list_dct[k]["label"])] += 1
        else:
            sentence_label = label
        if error == True:
            error = False
            continue
        data_frame["content"].append(i)
        data_frame["comparative"].append(key)
        data_frame["subject"].append(subj)
        data_frame["object"].append(obj)
        data_frame["predicate"].append(pred)
        data_frame["aspect"].append(asp)
        data_frame["label"].append(lab)
        data_frame["NER"].append(sentence_label)
        error = False
    return pd.DataFrame(data=data_frame)

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

''' Bao '''

def remove_text_before_tab(input_string):
    parts = input_string.split('\t', 1)
    if len(parts) > 1:
        return parts[1]
    else:
        return input_string

def get_block_list(filepath):
  block_list = []
  # Open the .txt file for reading
  with open(filepath, 'r', encoding="utf8") as file:
      content = file.read()

  # Split the content into blocks separated by empty lines
  blocks = content.split('\n\n')

  # Define a regular expression pattern to match lines containing JSON (square brackets)
  pattern = r'\[.*\]'

  # Iterate through the blocks and print or store the ones containing JSON lines
  for block in blocks:
      if re.search(pattern, block, re.DOTALL) and "[original]" not in block:
          if "[quintuple]" in block:
            block = block.replace("[quintuple]", "")
            block = remove_text_before_tab(block)
            block_list.append(block)
          elif "[spellquintuple]" in block:
            block = block.replace("[spellquintuple]", "")
            block = remove_text_before_tab(block)
            block_list.append(block)
          else:
            block = remove_text_before_tab(block)
            block_list.append(block)
  return block_list

def extract_dictionary_and_sentence(block):
  lines = block.split('\n')

  # Initialize a list to store the extracted dictionaries
  extracted_dicts = []

  # Iterate through the lines, skip the first line, and parse the rest as dictionaries
  for line in lines[1:]:
      if line.strip():  # Skip empty lines
          extracted_dict = json.loads(line)  # Parse the line as a dictionary
          extracted_dicts.append(extracted_dict)
  return extracted_dicts, lines[0]

def extract_dictionary_and_sentence(block):
  lines = block.split('\n')

  # Initialize a list to store the extracted dictionaries
  extracted_dicts = []

  # Iterate through the lines, skip the first line, and parse the rest as dictionaries
  for line in lines[1:]:
      if line.strip():  # Skip empty lines
          extracted_dict = json.loads(line)  # Parse the line as a dictionary
          extracted_dicts.append(extracted_dict)
  return extracted_dicts, lines[0]

def get_sentence_components(components):
  subject = [comp.split("&&")[1] for comp in components[0]["subject"]]
  object = [comp.split("&&")[1] for comp in components[0]["object"]]
  aspect = [comp.split("&&")[1] for comp in components[0]["aspect"]]
  predicate = [comp.split("&&")[1] for comp in components[0]["predicate"]]
  label = components[0]["label"]
  component_list = []
  component_list.append(subject)
  component_list.append(object)
  component_list.append(aspect)
  component_list.append(predicate)
  component_list.append(label)
  return component_list

def replace_nested_list(nested_list, old_list, new_list):
  # loop through the nested list
  for i in range(len(nested_list)):
    # check if the current element is equal to the old list
    if nested_list[i] == old_list:
      # replace it with the new list
      nested_list[i] = new_list
  # return the modified nested list
  return nested_list

def add_indices_to_dict(input_sentence, dictionary):
    # Split the input sentence into words
    words = input_sentence.split()

    # Initialize an output dictionary
    output_dict = {}

    for key, values in dictionary.items():
        output_values = []
        for value in values:
            # Find the index of the value in a case-insensitive manner
            index = next((i + 1 for i, word in enumerate(words) if word.lower() == value.lower()), None)
            if index is not None:
                # Add the index to the value with "&&" sign
                indexed_value = f"{index}&&{value}"
                output_values.append(indexed_value)

        output_dict[key] = output_values
    output_dict["label"] = dictionary["label"]
    return output_dict

def make_unique(main_list):
  unique_list = []
  unique_set = set()

  for sublist in main_list:
      # Convert the sublist to a tuple for hashing
      sub_tuple = tuple(sublist)

      # Check if the tuple is not in the set (i.e., it's unique)
      if sub_tuple not in unique_set:
          unique_set.add(sub_tuple)
          unique_list.append(list(sub_tuple))  # Convert the tuple back to a list
  return unique_list

def make_unique_for_pred(main_list):
  unique_list = []
  unique_set = set()

  for sublist in main_list:
      sub_tuple = (tuple(sublist[0]), sublist[1])  # Convert the inner list to a tuple
      if sub_tuple not in unique_set:
          unique_set.add(sub_tuple)
          unique_list.append([list(sub_tuple[0]), sub_tuple[1]])
  return unique_list

def swap_sentences(first_sentence, pred_list, subject_list, object_list, aspect_list):
  new_component_dict = {}
  first_content = extract_dictionary_and_sentence(first_sentence)[1]
  first_component = get_sentence_components(extract_dictionary_and_sentence(first_sentence)[0])
  first_subject = " ".join(first_component[0])
  first_object = " ".join(first_component[1])
  first_aspect = " ".join(first_component[2])
  first_pred = " ".join(first_component[3])
  first_label = first_component[4]

  random_subject_int = random.randint(0, len(subject_list) - 1)
  random_object_int = random.randint(0, len(object_list) - 1)
  random_aspect_int = random.randint(0, len(aspect_list) - 1)
  random_pred_label_int = random.randint(0, len(pred_list) - 1)

  random_subject = subject_list[random_subject_int]
  random_object = object_list[random_object_int]
  random_aspect = aspect_list[random_aspect_int]
  random_pred = pred_list[random_pred_label_int][0]
  random_label = pred_list[random_pred_label_int][1]


  #for first new sentence
  if len(first_subject) > 0 and len(first_object) > 0 and first_subject == first_object:
    first_new_content = first_content.replace(first_subject, " ".join(random_subject))
    new_component_dict.update({"subject": random_subject})
    new_component_dict.update({"object": random_subject})
  else:
    if len(first_subject) > 0:
      first_new_content = first_content.replace(first_subject, " ".join(random_subject), 1)
      new_component_dict.update({"subject": random_subject})
    else:
      first_new_content = first_content
      new_component_dict.update({"subject": first_component[0]})

    if len(first_object) > 0:
      first_new_content = first_new_content.replace(first_object, " ".join(random_object), 1)
      new_component_dict.update({"object": random_object})
    else:
        new_component_dict.update({"object": first_component[1]})

  if len(first_aspect) > 0:
    first_new_content = first_new_content.replace(first_aspect, " ".join(random_aspect), 1)
    new_component_dict.update({"aspect": random_aspect})
  else:
    new_component_dict.update({"aspect": first_component[2]})

  # new_component_dict.update({"aspect": first_component[2]})

  first_new_content = first_new_content.replace(first_pred, " ".join(random_pred), 1)
  new_component_dict.update({"predicate": random_pred})
  new_component_dict.update({"label": random_label})

  new_component_dict = add_indices_to_dict(first_new_content, new_component_dict)
  output_block = first_new_content + "\n" + json.dumps(new_component_dict, ensure_ascii=False)
  return output_block

def swap_multi_label_sentence_majority(first_sentence, subject_list, object_list, aspect_list):
  first_content = extract_dictionary_and_sentence(first_sentence)[1]
  dict_list = extract_dictionary_and_sentence(first_sentence)[0]
  component_list = []
  for i in dict_list:
    component_list.append(get_sentence_components([i]))

  new_component_list = []
  for i in dict_list:
    new_component_list.append(get_sentence_components([i]))

  unique_subject_list = []
  unique_object_list = []
  unique_aspect_list = []
  unique_pred_label_list = []
  #iterate through the component list
  new_first_content = first_content
  for i in new_component_list:
    if len(i[0]) > 0:
      unique_subject_list.append(i[0])
    if len(i[1]) > 0:
      unique_object_list.append(i[1])
    if len(i[2]) > 0:
      unique_aspect_list.append(i[2])
    unique_pred_label_list.append([i[3], i[4]])

  unique_subject_list = make_unique(unique_subject_list)
  unique_object_list = make_unique(unique_object_list)
  unique_aspect_list = make_unique(unique_aspect_list)
  unique_pred_label_list = make_unique_for_pred(unique_pred_label_list)
  # print(unique_subject_list)
  # print(unique_object_list)
  # print(unique_aspect_list)
  # print(unique_pred_label_list)


  # for i in range(len(component_list)):
  #   first_subject = " ".join(component_list[i][0])
  #   first_object = " ".join(component_list[i][1])
  #   new_first_content = new_first_content.replace(first_subject, second_subject, 1)
  #   new_first_content = new_first_content.replace(first_object, second_object, 1)
  # output_block = new_first_content + "\n"
  if len(unique_subject_list) > 0:
    for i in unique_subject_list:
      random_subject_int = random.randint(0, len(subject_list) - 1)
      random_subject = subject_list[random_subject_int]
      new_first_content = new_first_content.replace(" ".join(i), " ".join(random_subject))
      for x in range(len(component_list)):
        new_component_list[x] = replace_nested_list(new_component_list[x], i, random_subject)
  if len(unique_object_list) > 0:
    for i in unique_object_list:
      random_object_int = random.randint(0, len(object_list) - 1)
      random_object = object_list[random_object_int]
      new_first_content = new_first_content.replace(" ".join(i), " ".join(random_object))
      for x in range(len(component_list)):
        new_component_list[x] = replace_nested_list(new_component_list[x], i, random_object)

  if len(unique_aspect_list) > 0:
    for i in unique_aspect_list:
      random_aspect_int = random.randint(0, len(aspect_list) - 1)
      random_aspect = aspect_list[random_aspect_int]
      new_first_content = new_first_content.replace(" ".join(i), " ".join(random_aspect))
      for x in range(len(component_list)):
        new_component_list[x] = replace_nested_list(new_component_list[x], i, random_aspect)

  # if len(unique_pred_label_list) > 0:
  #   for i in unique_pred_label_list:
  #         random_pred_label_int = random.randint(0, len(predicate_label_list) - 1)
  #         random_pred = predicate_label_list[random_pred_label_int][0]
  #         random_label = predicate_label_list[random_pred_label_int][1]
  #         true_pred = i[0]
  #         true_label = i[1]
  #         new_first_content = new_first_content.replace(" ".join(true_pred), " ".join(random_pred))
  #         for x in range(len(component_list)):
  #           if new_component_list[x][3] == true_pred:
  #             new_component_list[x][4] = random_label
  #           new_component_list[x] = replace_nested_list(new_component_list[x], true_pred, random_pred)


  output_block = new_first_content + "\n"
  for i in new_component_list:
    final_dict = {}
    final_dict.update({"subject": i[0]})
    final_dict.update({"object": i[1]})
    final_dict.update({"aspect": i[2]})
    final_dict.update({"predicate": i[3]})
    final_dict.update({"label": i[4]})
    final_dict = add_indices_to_dict(new_first_content, final_dict)
    output_block = output_block + json.dumps(final_dict, ensure_ascii=False) + "\n"
  return output_block

def swap_multi_label_sentence_minority(first_sentence, pred_list, subject_list, object_list, aspect_list):
  first_content = extract_dictionary_and_sentence(first_sentence)[1]
  dict_list = extract_dictionary_and_sentence(first_sentence)[0]
  component_list = []
  for i in dict_list:
    component_list.append(get_sentence_components([i]))

  new_component_list = []
  for i in dict_list:
    new_component_list.append(get_sentence_components([i]))

  unique_subject_list = []
  unique_object_list = []
  unique_aspect_list = []
  unique_pred_label_list = []
  #iterate through the component list
  new_first_content = first_content
  for i in new_component_list:
    if len(i[0]) > 0:
      unique_subject_list.append(i[0])
    if len(i[1]) > 0:
      unique_object_list.append(i[1])
    if len(i[2]) > 0:
      unique_aspect_list.append(i[2])
    unique_pred_label_list.append([i[3], i[4]])

  unique_subject_list = make_unique(unique_subject_list)
  unique_object_list = make_unique(unique_object_list)
  unique_aspect_list = make_unique(unique_aspect_list)
  unique_pred_label_list = make_unique_for_pred(unique_pred_label_list)
  # print(unique_subject_list)
  # print(unique_object_list)
  # print(unique_aspect_list)
  # print(unique_pred_label_list)


  # for i in range(len(component_list)):
  #   first_subject = " ".join(component_list[i][0])
  #   first_object = " ".join(component_list[i][1])
  #   new_first_content = new_first_content.replace(first_subject, second_subject, 1)
  #   new_first_content = new_first_content.replace(first_object, second_object, 1)
  # output_block = new_first_content + "\n"
  if len(unique_subject_list) > 0:
    for i in unique_subject_list:
      random_subject_int = random.randint(0, len(subject_list) - 1)
      random_subject = subject_list[random_subject_int]
      new_first_content = new_first_content.replace(" ".join(i), " ".join(random_subject))
      for x in range(len(component_list)):
        new_component_list[x] = replace_nested_list(new_component_list[x], i, random_subject)
  if len(unique_object_list) > 0:
    for i in unique_object_list:
      random_object_int = random.randint(0, len(object_list) - 1)
      random_object = object_list[random_object_int]
      new_first_content = new_first_content.replace(" ".join(i), " ".join(random_object))
      for x in range(len(component_list)):
        new_component_list[x] = replace_nested_list(new_component_list[x], i, random_object)

  if len(unique_aspect_list) > 0:
    for i in unique_aspect_list:
      random_aspect_int = random.randint(0, len(aspect_list) - 1)
      random_aspect = aspect_list[random_aspect_int]
      new_first_content = new_first_content.replace(" ".join(i), " ".join(random_aspect))
      for x in range(len(component_list)):
        new_component_list[x] = replace_nested_list(new_component_list[x], i, random_aspect)

  if len(unique_pred_label_list) > 0:
    for i in unique_pred_label_list:
          random_pred_label_int = random.randint(0, len(pred_list) - 1)
          random_pred = pred_list[random_pred_label_int][0]
          random_label = pred_list[random_pred_label_int][1]
          true_pred = i[0]
          true_label = i[1]
          new_first_content = new_first_content.replace(" ".join(true_pred), " ".join(random_pred))
          for x in range(len(component_list)):
            if new_component_list[x][3] == true_pred:
              new_component_list[x][4] = random_label
            new_component_list[x] = replace_nested_list(new_component_list[x], true_pred, random_pred)


  output_block = new_first_content + "\n"
  for i in new_component_list:
    final_dict = {}
    final_dict.update({"subject": i[0]})
    final_dict.update({"object": i[1]})
    final_dict.update({"aspect": i[2]})
    final_dict.update({"predicate": i[3]})
    final_dict.update({"label": i[4]})
    final_dict = add_indices_to_dict(new_first_content, final_dict)
    output_block = output_block + json.dumps(final_dict, ensure_ascii=False) + "\n"
  return output_block

