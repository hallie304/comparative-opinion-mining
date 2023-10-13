import numpy as np
import pandas as pd

from modelUtils import *

def getTextFileGeneralInfo(path, curDir = curDir, idx = range(1, 61, 1)):
    trainWithLabel = []
    sentenceList = []
    # prevAltDes = False

    for i in idx:
        fileName = "//train_%04d.txt" % (i)
        fileName = path + curDir + fileName
        lineNo = 0

        with open(fileName, 'r', encoding='utf8') as txtFile:
            for line in txtFile:
                lineNo += 1
                if (line[:-1] != ''):
                    try:
                        data = ast.literal_eval(line)
                        if isinstance(data, dict):
                            trainWithLabel.append(data)

                            # if (prevAltDes):
                            #     print("%d %d" % (i, lineNo))
                    except:
                        # prevAltDes = False
                        # if isinstance(line[:-1], str):
                        sentenceList.append(line[:-1])

                        # if (line[:-1][:4] == "alt:"):
                        # if ((line[:-1][:4] == "alt:") | (line[:-1][:4] == "des:")):
                        # prevAltDes = True

    temp = trainWithLabel[0].keys()

    print("trainWithLabel size: %d" % (len(trainWithLabel)))
    print("Total number of sentences: %d" % (len(sentenceList)))
    print("Dictionary keys: %s" % (str(list(temp))))

    for i in range(len(trainWithLabel)):
        if (len(set(trainWithLabel[i].keys()).intersection(temp)) != 5):
            print("NOT QUINTUPLE!!")

    for i in range(len(trainWithLabel)):
        if (len(trainWithLabel[i]['predicate']) == 0):
            print("PREDICATE EMPTY!")
        if (len(trainWithLabel[i]['label']) == 0):
            print("LABEL EMPTY!")

    countDict = {'DIF': 0, 'EQL': 0, 'SUP': 0, 'SUP+': 0, 'SUP-': 0, 'COM': 0, 'COM+': 0, 'COM-': 0}
    for i in range(len(trainWithLabel)):
        countDict[trainWithLabel[i]['label']] += 1
    print("Label count: %s" % (countDict.__str__()))


def txt2csv(path, curDir = curDir, idx = range(1, 61, 1)):
    trainWithLabel = []
    sentenceList = []
    isComparative = []
    prevSentence = False
    # prevAltDes = False

    for i in idx:
        fileName = "//train_%04d.txt" % (i)
        fileName = curDir + path + fileName
        lineNo = 0

        with open(fileName, 'r', encoding='utf8') as txtFile:
            for line in txtFile:
                lineNo += 1
                if (line[:-1] != ''):
                    try:
                        data = ast.literal_eval(line)
                        if isinstance(data, dict):
                            trainWithLabel.append(data)
                            isComparative.append(1)
                            if (prevSentence):
                                prevSentence = False
                            else:
                                sentenceList.append(sentenceList[-1])

                    except:
                        # prevAltDes = False
                        if (prevSentence):
                            trainWithLabel.append(
                                {'subject': [], 'object': [], 'aspect': [], 'predicate': [], 'label': []})
                            isComparative.append(0)

                        sentenceList.append(line[:-1])
                        prevSentence = True

                        # if (line[:-1][:4] == "alt:"):
                        # if ((line[:-1][:4] == "alt:") | (line[:-1][:4] == "des:")):
                        # prevAltDes = True

    trainWithLabel.append({'subject': [], 'object': [], 'aspect': [], 'predicate': [], 'label': []})
    isComparative.append(0)

    # Creating csv
    subj = []
    obj = []
    asp = []
    predicate = []
    label = []

    for i in range(len(trainWithLabel)):
        subj.append(trainWithLabel[i]['subject'])
        obj.append(trainWithLabel[i]['object'])
        asp.append(trainWithLabel[i]['aspect'])
        predicate.append(trainWithLabel[i]['predicate'])
        label.append(trainWithLabel[i]['label'])

    myData = pd.DataFrame(list(zip(sentenceList, isComparative, subj, obj, asp, predicate, label)),
                          columns = ['Input sentence', 'isComparative', 'subject',
                                     'object', 'aspect', 'predicate', 'label'])
    return myData


def createDataNERCSV(dataCSV):
    # data = pd.read_csv(curDir + "//datasets//modified//preprocessed//VLSP23.csv", index_col="Unnamed: 0")
    data = dataCSV

    for i in range(len(data)):
        # data.at[i, 'Input sentence'] = data.at[i, 'Input sentence'].replace('.', ' .')
        data.at[i, 'Input sentence'] = data.at[i, 'Input sentence'].replace('+', ' +')
        data.at[i, 'Input sentence'] = data.at[i, 'Input sentence'].replace('/', ' / ')
        data.at[i, 'Input sentence'] = data.at[i, 'Input sentence'].replace('(', ' ( ')
        data.at[i, 'Input sentence'] = data.at[i, 'Input sentence'].replace(')', ' ) ')
        # data.at[i, 'Input sentence'] = data.at[i, 'Input sentence'].replace('  ', ' ')
        data.at[i, 'Input sentence'] = data.at[i, 'Input sentence'].replace(' ', '   ')
        data.at[i, 'Input sentence'] = data.at[i, 'Input sentence'].replace('​', ' ​ ')
        data.at[i, 'Input sentence'] = data.at[i, 'Input sentence'].replace('@', ' @ ')
        data.at[i, 'Input sentence'] = data.at[i, 'Input sentence'].replace('%', ' % ')

        if isinstance(data.at[i, 'subject'], str):
            data.at[i, 'subject'] = segmentWordInList(ast.literal_eval(data.at[i, 'subject']))
            data.at[i, 'object'] = segmentWordInList(ast.literal_eval(data.at[i, 'object']))
            data.at[i, 'aspect'] = segmentWordInList(ast.literal_eval(data.at[i, 'aspect']))
            data.at[i, 'predicate'] = segmentWordInList(ast.literal_eval(data.at[i, 'predicate']))

    # Select with label isComparative = True
    dataSelected = data[data['isComparative'] == 1]

    sentenceSegmented = list(map(rdrSegmenter.word_segment, dataSelected['Input sentence']))
    sentenceSegmentedConcat = np.concatenate(sentenceSegmented)

    labelNER = []
    for i in range(len(sentenceSegmented)):
        # for i in range(1):
        _, _, subj, obj, aspect, predicate, _ = dataSelected.iloc[i]
        # subj = ast.literal_eval(subj)
        # obj = ast.literal_eval(obj)
        # aspect = ast.literal_eval(aspect)
        # predicate = ast.literal_eval(predicate)
        count = 0
        for sentence in sentenceSegmented[i]:
            tempLabelNER = []
            for subSentence in sentence.split(" "):
                labelSet = set()
                for subSubSentence in subSentence.split("_"):
                    count += 1
                    if (subSubSentence.count('.') > 0):
                        count += subSubSentence.count('.') - 1
                    if (subSubSentence.count(',') > 0):
                        count += subSubSentence.count(',') - 1
                    labelSet.add(
                        checkInList(subj, obj, aspect, predicate, rdrSegmenter.word_segment(subSubSentence)[0], count,
                                    labelDict))
                if (len(labelSet) == 1):
                    tempLabelNER.append(labelSet.pop())
                elif ((len(labelSet) == 2) & (labelDict['<noClass>'] in labelSet)):
                    labelSet.remove(labelDict['<noClass>'])
                    tempLabelNER.append(labelSet.pop())
                else:
                    print('WARNING: PHRASE DIFFERENT LABEL!!')
                    tempLabelNER.append(labelDict['<noClass>'])
            labelNER.append(deepcopy(tempLabelNER))
            tempLabelNER.clear()
    labelNER = np.array(labelNER, dtype = list)
    labelIsComparative = np.ones_like(labelNER)
    labelComparisonType = []

    for i in range(len(sentenceSegmented)):
        for j in range(len(sentenceSegmented[i])):
            labelComparisonType.append(comparisonLabel2id[dataSelected['label'].iloc[i]])
    labelComparisonType = np.array(labelComparisonType)

    checkForSegmentedSentenceLabelAlign(sentenceSegmentedConcat, labelNER)

    myData = pd.DataFrame(list(zip(sentenceSegmentedConcat, labelIsComparative, labelNER, labelComparisonType)),
                          columns = ['Input sentence segmented', 'labelIsComparative', 'labelNER', 'labelComparisonType'])

    # Select with label isComparative = False
    dataSelected = data[data['isComparative'] == 0]

    sentenceSegmented = list(map(rdrSegmenter.word_segment, dataSelected['Input sentence']))
    sentenceSegmentedConcat = np.concatenate(sentenceSegmented)

    labelNER = []
    for i in range(len(sentenceSegmentedConcat)):
        labelNER.append([labelDict['<noClass>']] * len(sentenceSegmentedConcat[i].split(" ")))
    labelNER = np.array(labelNER, dtype = list)
    labelIsComparative = np.zeros_like(labelNER)
    labelComparisonType = []

    for i in range(len(sentenceSegmented)):
        for j in range(len(sentenceSegmented[i])):
            labelComparisonType.append(comparisonLabel2id['<noClass>'])
    labelComparisonType = np.array(labelComparisonType)

    checkForSegmentedSentenceLabelAlign(sentenceSegmentedConcat, labelNER)

    myData1 = pd.DataFrame(list(zip(sentenceSegmentedConcat, labelIsComparative, labelNER, labelComparisonType)),
                          columns = ['Input sentence segmented', 'labelIsComparative', 'labelNER', 'labelComparisonType'])

    return myData, myData1


def tokenizeAndProcess(dataCSVIsComparative, dataCSVNotIsComparative):
    # dataNER = pd.read_csv(curDir + "//datasets//modified//preprocessed//VLSP23_NER.csv", index_col="Unnamed: 0")
    dataNER = dataCSVIsComparative

    dropInd = []
    for i in range(len(dataNER)):
        if isinstance(dataNER.at[i, 'labelNER'], str):
            dataNER.at[i, 'labelNER'] = ast.literal_eval(dataNER.at[i, 'labelNER'])
        if (len(set(dataNER.iloc[i]['labelNER'])) == 1):
            dropInd.append(i)
    dataNER = dataNER.drop(dropInd)

    dataCSVNotIsComparative = dataCSVNotIsComparative.drop_duplicates(subset = ['Input sentence segmented'])

    dataNER = pd.concat([dataNER, dataCSVNotIsComparative], ignore_index = True)

    sentenceTokenized = tokenize_and_align_labels(dataNER)

    return sentenceTokenized


def segmentWordInList(lst, segmenter = rdrSegmenter.word_segment):
    res = []
    for i in range(len(lst)):
        pos, word = lst[i].split('&&')
        word = segmenter(word)[0]
        res.append(pos + '&&' + word)

    return res


def checkInList(subjLst, objLst, aspectLst, predicateLst, word, id, labelDict):
    for element in subjLst:
        elementID, elementWord = element.split("&&")
        if ((int(elementID) == id) & (elementWord == word)):
            return labelDict['subject']

    for element in objLst:
        elementID, elementWord = element.split("&&")
        if ((int(elementID) == id) & (elementWord == word)):
            return labelDict['object']

    for element in aspectLst:
        elementID, elementWord = element.split("&&")
        if ((int(elementID) == id) & (elementWord == word)):
            return labelDict['aspect']

    for element in predicateLst:
        elementID, elementWord = element.split("&&")
        if ((int(elementID) == id) & (elementWord == word)):
            return labelDict['predicate']

    return labelDict['<noClass>']


def checkForSegmentedSentenceLabelAlign(sentenceSegmentedConcat, labelNER):
    if (len(sentenceSegmentedConcat) != len(labelNER)):
        print('WARNING: DIFF TOTAL NO.LABELS AND TOTAL NO.SENTENCES')
    else:
        for i in range(len(sentenceSegmentedConcat)):
            # for i in range(1):
            if (len(sentenceSegmentedConcat[i].split(' ')) != len(labelNER[i])):
                print('WARNING: DIFF NO.LABELS AND NO.SUBWORDS')


def tokenize_and_align_labels(datasetCSV):
    tokenized_inputs = phobertTokenizer(list(datasetCSV["Input sentence segmented"]), truncation = True,
                                        is_split_into_words = False, padding = True)
    labels = []

    for i, label in enumerate(datasetCSV["labelNER"]):
        word_ids = tokenized_inputs.word_ids(batch_index = i)  # Map tokens to their respective word.
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:  # Set the special tokens to -100.
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:  # Only label the first token of a given word.
                label_ids.append(label[word_idx])
            else:
                label_ids.append(-100)
            previous_word_idx = word_idx
        labels.append(label_ids)

    tokenized_inputs["labelNER"] = labels
    tokenized_inputs["labelIsComparative"] = list(datasetCSV["labelIsComparative"])
    tokenized_inputs["labelComparisonType"] = list(datasetCSV["labelComparisonType"])

    sentenceTokenizedDict = dict(tokenized_inputs)
    dataNERTokenized = pd.DataFrame(sentenceTokenizedDict)
    # dataNERTokenized.to_csv(r"C:\Users\nguye\Downloads\VLSP23_dataNERPhoBERTTokenized.csv")

    return dataNERTokenized


class DataNERPhoBERTTorch(torch.utils.data.Dataset):

    def __init__(self, datasetNERTokenizedCSV, transform = None, target_transform = None):
        # self.datasetNER = pd.read_csv(datasetNERTokenizedPath, index_col='Unnamed: 0').iloc[range(90)]
        # self.datasetNER = pd.read_csv(datasetNERTokenizedPath, index_col = 'Unnamed: 0')
        self.datasetNER = datasetNERTokenizedCSV
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.datasetNER)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        if isinstance(idx, int):
            idx = [idx]

        dataDict = (self.datasetNER.iloc[idx]).to_dict(orient = 'list')
        sentenceTokenized = dataDict['input_ids']
        attentionMask = dataDict['attention_mask']
        labelNER = dataDict['labelNER']
        labelIsComparative = dataDict['labelIsComparative']
        labelComparisonType = dataDict['labelComparisonType']

        try:
            sentenceTokenized = ast.literal_eval(sentenceTokenized)
            attentionMask = ast.literal_eval(attentionMask)
            labelNER = ast.literal_eval(labelNER)
            labelIsComparative = ast.literal_eval(labelIsComparative)
            labelComparisonType = ast.literal_eval(labelComparisonType)
        except:
            pass

        sentenceTokenized = torch.tensor(sentenceTokenized)
        attentionMask = torch.tensor(attentionMask)
        labelNER = torch.tensor(labelNER)
        labelIsComparative = torch.tensor(labelIsComparative)
        labelComparisonType = torch.tensor(labelComparisonType)

        if self.transform:
            sentenceTokenized = self.transform(sentenceTokenized)

        if self.target_transform:
            labelNER = self.target_transform(labelNER)

        return {'input_ids': sentenceTokenized,
                'attention_mask': attentionMask,
                'labelNER': labelNER,
                'labelIsComparative': labelIsComparative,
                'labelComparisonType': labelComparisonType}


def splitDataset(dataset, splitSize1, splitSize2, splitSize3, batchSize, shuffleDataset, randSeed):
    datasetSize = len(dataset)
    ind = list(range(datasetSize))
    if (np.around(splitSize1 + splitSize2 + splitSize3) != 1):
        print('WARNING: PROPORTION NOT SUM UP TO 1')
    else:
        splitInd = int(np.floor(splitSize1 * datasetSize))
        splitInd2 = int(np.ceil(splitSize2 * datasetSize))
    if (shuffleDataset):
        np.random.seed(randSeed)
        np.random.shuffle(ind)
    splitSet1Ind, splitSet2Ind, splitSet3Ind = ind[: splitInd], ind[splitInd: splitInd + splitInd2], ind[splitInd + splitInd2:]
    set1Sampler, set2Sampler, set3Sampler = (torch.utils.data.SubsetRandomSampler(splitSet1Ind),
                                             torch.utils.data.SubsetRandomSampler(splitSet2Ind),
                                             torch.utils.data.SubsetRandomSampler(splitSet3Ind))
    return (torch.utils.data.DataLoader(dataset, batch_size = batchSize, sampler = set1Sampler),
            torch.utils.data.DataLoader(dataset, batch_size = batchSize, sampler = set2Sampler),
            torch.utils.data.DataLoader(dataset, batch_size = batchSize, sampler = set3Sampler))


