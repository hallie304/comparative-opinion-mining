import pandas as pd
import re

stat = [0,0,0,0,0,0,0,0,0]
type_of_compare = ["Non","DIF", "EQL", "SUP+", "SUP-", "SUP", "COM+", "COM-", "COM"]

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