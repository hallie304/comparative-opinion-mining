import data_utils
import random
import json
import new_data_sampling_A


blocklist = []
for i in range(1, 61):
    if i < 10:
        text_path = r"../../data/public and train data\train_000" + str(i) + ".txt"
    if i >= 10:
        text_path = r"../../data/public and train data\train_00" + str(i) + ".txt"
    blocklist.append(data_utils.get_block_list(text_path))

for i in range(1, 25):
  if i < 10:
    text_path = r"../../data/public and train data\dev_000" + str(i) + ".txt"
  if i >= 10:
    text_path = r"../../data/public and train data\dev_00" + str(i) + ".txt"
  blocklist.append(data_utils.get_block_list(text_path))

block_list = [item for sublist in blocklist for item in sublist]

block_list[670] = block_list[670].replace("\n \nTại sao lại có luồng ý kiến này?\tTại sao lại có luồng ý kiến này ?", "")

#extract components
subject_list = []
object_list = []
aspect_list = []
predicate_label_list = []
for i in block_list:
  dict_list = data_utils.extract_dictionary_and_sentence(i)[0]
  component_list = []
  try:
    for x in dict_list:
      component_list.append(data_utils.get_sentence_components([x]))
    for z in component_list:
      component = z
      try:
        component = data_utils.get_sentence_components(data_utils.extract_dictionary_and_sentence(i)[0])
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

filepath_list = [r"../../data/predicate list/dif_predicate.txt",
                 r"../../data/predicate list/eql_predicate_list.txt",
                 r"../../data/predicate list/sup+_predicate.txt",
                 r"../../data/predicate list/sup-_predicate.txt",
                 r"../../data/predicate list/com_predicate.txt",
                 r"../../data/predicate list/com+_predicate.txt",
                 r"../../data/predicate list/com-_predicate.txt",
                 r"../../data/predicate list/sup_predicate.txt"]
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

name_list = []

# Load the JSON data from the file
with open(r'../../data/device and brand/devices.json', 'r') as file:
    data = json.load(file)

# Extract "name" field from each record, split it, and append to a list
for record in data["RECORDS"]:
    name = record["name"]
    words = name.split()
    name_list.append(words)

# Load the JSON data from the file
with open(r'../../data/device and brand/brands.json', 'r') as brand_file:
    brand_data = json.load(brand_file)

# Extract "name" field from each record, split it, and append to a list
for record in brand_data["RECORDS"]:
    name = record["name"]
    words = name.split()
    name_list.append(words)

for i in name_list:
    subject_list.append(i)
    object_list.append(i)

subject_list = data_utils.make_unique(subject_list)
object_list = data_utils.make_unique(object_list)
aspect_list = data_utils.make_unique(aspect_list)
predicate_label_list = data_utils.make_unique_for_pred(predicate_label_list)

random.shuffle(subject_list)
random.shuffle(object_list)
random.shuffle(aspect_list)
random.shuffle(predicate_label_list)


for i in predicate_label_list:
  if "SUP-" == i[1]:
    print(i)

def generate_data(lab,num,aim):
    multi_label_block = []
    for i in block_list:
        try:
            if len(data_utils.extract_dictionary_and_sentence(i)[0]) > 1:
                multi_label_block.append(i)
        except:
            print(i)
            print(block_list.index(i))
    current_num = num
    num_sentence_generate = round((aim - current_num) / 2)
    num_multi = num_sentence_generate // 4
    num_single = num_sentence_generate - num_multi
    label = '"'+lab+'"}'
    sup_predicate = []
    for i in predicate_label_list:
      if lab == i[1]:
        sup_predicate.append(i)

    multi_sup_list = []
    for i in multi_label_block:
      if label in i and ("cả hai" not in i or "Cả hai" not in i):
        multi_sup_list.append(i)

    # majority
    if lab != 'SUP' and lab != 'SUP-' and lab != 'COM':
        with open("Generate/"+lab+".txt", "w", encoding="utf8") as file:
          for i in range(num_sentence_generate):
            if i <= num_multi:
              try:
                rand_int_1 = random.randint(0, len(multi_sup_list) - 1)
                block = data_utils.swap_multi_label_sentence_majority(multi_sup_list[rand_int_1], subject_list, object_list, aspect_list)
                file.write(block + "\n\n")
              except:
                print(i)
            if i > num_multi:
              try:
                rand_int_1 = random.randint(0, len(block_list) - 1)
                block = data_utils.swap_sentences(block_list[rand_int_1], sup_predicate, subject_list, object_list, aspect_list)
                file.write(block + "\n\n")
              except:
                print(i)

    #minority
    with open("Generate/"+lab+".txt", "w", encoding="utf8") as file:
        for i in range(num_sentence_generate):
            if i <= num_multi:
                try:
                    rand_int_1 = random.randint(0, len(multi_label_block) - 1)
                    block = data_utils.swap_multi_label_sentence_minority(multi_label_block[rand_int_1], sup_predicate, subject_list, object_list, aspect_list)
                    file.write(block + "\n\n")
                except:
                    print(i)
            if i > num_multi:
                try:
                    rand_int_1 = random.randint(0, len(block_list) - 1)
                    block = data_utils.swap_sentences(block_list[rand_int_1], sup_predicate, subject_list, object_list, aspect_list)
                    file.write(block + "\n\n")
                except:
                    print(i)
            for i in range(num_sentence_generate):
                pass

