import re
import random
import json
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

blocklist = []
for i in range(1, 61):
  if i < 10:
    text_path = r"C:\Users\Duc Bao\PycharmProjects\VLSP All Code\data\public and train data\train_000" + str(i) + ".txt"
  if i >= 10:
    text_path = r"C:\Users\Duc Bao\PycharmProjects\VLSP All Code\data\public and train data\train_00" + str(i) + ".txt"
  blocklist.append(get_block_list(text_path))

for i in range(1, 25):
  if i < 10:
    text_path = r"C:\Users\Duc Bao\PycharmProjects\VLSP All Code\data\public and train data\dev_000" + str(i) + ".txt"
  if i >= 10:
    text_path = r"C:\Users\Duc Bao\PycharmProjects\VLSP All Code\data\public and train data\dev_00" + str(i) + ".txt"
  blocklist.append(get_block_list(text_path))

block_list = [item for sublist in blocklist for item in sublist]

block_list[670] = block_list[670].replace("\n \nTại sao lại có luồng ý kiến này?\tTại sao lại có luồng ý kiến này ?", "")

# count = 0
# for i in block_list:
#   if "DIF" in i:
#     count += 1
# count
#
# count = 0
# for i in block_list:
#   if "EQL" in i:
#     count += 1
# count
#
# count = 0
# for i in block_list:
#   if "SUP" in i and ("SUP+" not in i and "SUP-" not in i) and "SUPER" not in i:
#     count += 1
# count
#
# count = 0
# for i in block_list:
#   if "SUP+" in i:
#     count += 1
# count
#
# count = 0
# for i in block_list:
#   if "SUP-" in i:
#     count += 1
# count
#
# count = 0
# for i in block_list:
#   if "COM" in i and ("COM+" not in i and "COM-" not in i):
#     count += 1
# count
#
# count = 0
# for i in block_list:
#   if "COM+" in i:
#     count += 1
# count
#
# count = 0
# for i in block_list:
#   if "COM-" in i:
#     count += 1
# count


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

#extract components
subject_list = []
object_list = []
aspect_list = []
predicate_label_list = []
for i in block_list:
  dict_list = extract_dictionary_and_sentence(i)[0]
  component_list = []
  try:
    for x in dict_list:
      component_list.append(get_sentence_components([x]))
    for z in component_list:
      component = z
      try:
        component = get_sentence_components(extract_dictionary_and_sentence(i)[0])
        subject = component[0]
        object = component[1]
        aspect = component[2]
        predicate_label = [component[3], component[4]]
        if len(subject) > 0:
          subject_list.append(subject)
        if len(object) > 0:
          object_list.append(object)
        if len(aspect) > 0:
          aspect_list.append(aspect)
        predicate_label_list.append(predicate_label)
      except:
        print(i)
  except:
    print(i)

filepath_list = [r"C:\Users\Duc Bao\PycharmProjects\VLSP All Code\data\predicate list\dif_predicate.txt", r"C:\Users\Duc Bao\PycharmProjects\VLSP All Code\data\predicate list\eql_predicate_list.txt", r"C:\Users\Duc Bao\PycharmProjects\VLSP All Code\data\predicate list\sup+_predicate.txt", r"C:\Users\Duc Bao\PycharmProjects\VLSP All Code\data\predicate list\sup-_predicate.txt", r"C:\Users\Duc Bao\PycharmProjects\VLSP All Code\data\predicate list\com_predicate.txt", r"C:\Users\Duc Bao\PycharmProjects\VLSP All Code\data\predicate list\com+_predicate.txt", r"C:\Users\Duc Bao\PycharmProjects\VLSP All Code\data\predicate list\com-_predicate.txt", r"C:\Users\Duc Bao\PycharmProjects\VLSP All Code\data\predicate list\sup_predicate.txt"]
error_count = 0
for i in filepath_list:
  with open(i, "r", encoding="utf8") as file:
      # loop through each line in the file
      for line in file:
          try:
          # remove the newline character at the end of the line
            line = line.strip()
            # evaluate the line as a Python expression and append it to the list
            predicate_label_list.append(eval(line))
          except:
            error_count += 1
            print(line)
print(error_count)

import json

name_list = []

# Load the JSON data from the file
with open(r'C:\Users\Duc Bao\PycharmProjects\VLSP All Code\data\device and brand\devices.json', 'r') as file:
    data = json.load(file)

# Extract "name" field from each record, split it, and append to a list
for record in data["RECORDS"]:
    name = record["name"]
    words = name.split()
    name_list.append(words)

# Load the JSON data from the file
with open(r'C:\Users\Duc Bao\PycharmProjects\VLSP All Code\data\device and brand\brands.json', 'r') as brand_file:
    brand_data = json.load(brand_file)

# Extract "name" field from each record, split it, and append to a list
for record in brand_data["RECORDS"]:
    name = record["name"]
    words = name.split()
    name_list.append(words)

for i in name_list:
  subject_list.append(i)
  object_list.append(i)

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
subject_list = make_unique(subject_list)
object_list = make_unique(object_list)
aspect_list = make_unique(aspect_list)
predicate_label_list = make_unique_for_pred(predicate_label_list)

random.shuffle(subject_list)
random.shuffle(object_list)
random.shuffle(aspect_list)
random.shuffle(predicate_label_list)

# with open("pred_label_list.txt", "w") as file:
#     # Iterate through the list and write each element to the file
#     for item in predicate_label_list:
#         file.write(" ".join(item[0]) + ": " + item[1] + "\n")

for i in predicate_label_list:
  if "SUP-" == i[1]:
    print(i)

def swap_sentences(first_sentence, pred_list):
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

multi_label_block = []
for i in block_list:
  try:
    if len(extract_dictionary_and_sentence(i)[0]) > 1:
      multi_label_block.append(i)
  except:
    print(i)
    print(block_list.index(i))


def swap_multi_label_sentence_majority(first_sentence):
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

def swap_multi_label_sentence_minority(first_sentence, pred_list):
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

current_num = 200
num_sentence_generate = round((1500 - current_num) / 2)
num_multi = num_sentence_generate // 4
num_single = num_sentence_generate - num_multi

label = '"EQL"}'
sup_predicate = []
for i in predicate_label_list:
  if "EQL" == i[1]:
    sup_predicate.append(i)

multi_sup_list = []
for i in multi_label_block:
  if label in i and ("cả hai" not in i or "Cả hai" not in i):
    multi_sup_list.append(i)


#majority
with open(r"C:\Users\Duc Bao\PycharmProjects\VLSP All Code\sampling output\eql_v5.txt", "w", encoding="utf8") as file:
  for i in range(num_sentence_generate):
    if i <= num_multi:
      try:
        rand_int_1 = random.randint(0, len(multi_sup_list) - 1)
        block = swap_multi_label_sentence_majority(multi_sup_list[rand_int_1])
        file.write(block + "\n\n")
      except:
        print(i)
    if i > num_multi:
      try:
        rand_int_1 = random.randint(0, len(block_list) - 1)
        block = swap_sentences(block_list[rand_int_1], sup_predicate)
        file.write(block + "\n\n")
      except:
        print(i)

#minority
with open(r"C:\Users\Duc Bao\PycharmProjects\VLSP All Code\sampling output\eql_v5.txt", "w", encoding="utf8") as file:
  for i in range(num_sentence_generate):
    if i <= num_multi:
      try:
        rand_int_1 = random.randint(0, len(multi_label_block) - 1)
        block = swap_multi_label_sentence_minority(multi_label_block[rand_int_1], sup_predicate)
        file.write(block + "\n\n")
      except:
        print(i)
    if i > num_multi:
      try:
        rand_int_1 = random.randint(0, len(block_list) - 1)
        block = swap_sentences(block_list[rand_int_1], sup_predicate)
        file.write(block + "\n\n")
      except:
        print(i)