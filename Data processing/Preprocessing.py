import pandas as pd
import os
import re
only_files = [f for f in os.listdir("VLSP23-Comparative-Opinion-Mining/data/relabledData") if
              os.path.isfile(os.path.join("VLSP23-Comparative-Opinion-Mining/data/relabledData", f))]
type_of_compare = ["DIF", "EQL", "SUP+", "SUP-", "SUP", "COM+", "COM-", "COM"]
data_frame = {"content": [], "comparative": [], "subject": [], "object": [], "aspect": [], "predicate": [], "label": [], "NER": []}


def preprocessing_with_BIO(filename):
    data_row = []
    labeled_row = dict()
    data = []
    with (open('VLSP23-Comparative-Opinion-Mining/data/relabledData/' + filename, 'r', encoding="utf-8") as f):
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
            r = r.replace("\n", "")
            if 'quintuple' in signal:
                row = f.readline()
                signal = ""
                continue
            data_row.append(r)
            row = f.readline()
    f.close()
    del f
    j = 0
    for i in range(len(data_row) - 1, -1, -1):
        if data_row[i] == "":
            data_row.remove(data_row[i])
    for i in range(len(data_row) - 1):
        if re.match("^\{\"subject\": \[", data_row[i + 1]) and re.match("^\{\"subject\": \[", data_row[i]):
            try:
                row = eval(data_row[i + 1])
            except Exception as e:
                print(filename,e,data_row[i+1])
            labeled_row[data_row[j]].append(row)
        elif re.match("^\{\"subject\": \[", data_row[i + 1]):
            if data_row[i] not in labeled_row.keys():
                j = i
                row = eval(data_row[i + 1])
                labeled_row[data_row[j]] = [row]
            else:
                try:
                    row = eval(data_row[i + 1])
                    labeled_row[data_row[i]].append(row)
                except Exception as e:
                    print(filename,e,data_row[i+1])
    for i in data_row:
        if re.match("^\{\"subject\": \[", i) is None:
            data.append(i)
    del data_row
    del j
    del row
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
                            if label[consecutive[q]-1] == 0:
                                if q == 0:
                                    label[consecutive[q] - 1] = 1
                                else:
                                    label[consecutive[q] - 1] = 2
                        elif j == "object":
                            if label[consecutive[q]-1] == 0:
                                if q == 0:
                                    label[consecutive[q] - 1] = 3
                                else:
                                    label[consecutive[q] - 1] = 4
                        elif j == "aspect":
                            if label[consecutive[q]-1] == 0:
                                if q == 0:
                                    label[consecutive[q] - 1] = 5
                                else:
                                    label[consecutive[q] - 1] = 6
                        elif j == "predicate":
                            if label[consecutive[q]-1] == 0:
                                if q == 0:
                                    label[consecutive[q] - 1] = 7
                                else:
                                    label[consecutive[q] - 1] = 8
                sentence_label.append(label)
                subj.append(list_dct[k]["subject"])
                obj.append(list_dct[k]["object"])
                pred.append(list_dct[k]["predicate"])
                asp.append(list_dct[k]["aspect"])
                lab.append(type_of_compare.index(list_dct[k]["label"]) + 1)
        else:
            sentence_label = label
        data_frame["content"].append(i)
        data_frame["comparative"].append(key)
        data_frame["subject"].append(subj)
        data_frame["object"].append(obj)
        data_frame["predicate"].append(pred)
        data_frame["aspect"].append(asp)
        data_frame["label"].append(lab)
        data_frame["NER"].append(sentence_label)
        for j in range(len(pred)):
            if pred[j] not in pred_label.keys():
                pred_label[pred[j]] = [lab[j]]
            elif pred[j] in pred_label.keys() and lab[j] not in pred_label[pred[j]]:
                pred_label[pred[j]].append(lab[j])
    del sentence_label
    del key
    del subj
    del obj
    del asp
    del pred
    del lab
    del w
    del label
    del consecutive
    del i
    del j
    del q
    del s
    return data_frame
for i in only_files:
    if i == 'total_multilabel.csv':
        continue
    preprocessing_with_BIO(i)

