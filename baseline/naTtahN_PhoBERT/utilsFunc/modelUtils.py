import numpy as np
import torch.nn

from utilsFunc.libsAndPackages import *

num_isComparative_labels = 2
num_NER_labels = len(label2id)
num_comparisonType_labels = len(comparisonLabel2id)
num_labels = num_isComparative_labels + num_NER_labels + num_comparisonType_labels

rdrSegmenter = py_vncorenlp.VnCoreNLP(annotators = ["wseg"], save_dir = curDir + "/vncorenlp")

# phobertFeatureExtractor = AutoModel.from_pretrained("vinai/phobert-base")
phobertTokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base", model_max_length = maxSeqLen)
phobertTokenClassification = AutoModelForTokenClassification.from_pretrained(
    pretrained_model_name_or_path = "vinai/phobert-base",
    num_labels = num_labels,
    # id2label = id2label,
    # label2id = label2id
)

# undertheseaNER = AutoModel.from_pretrained("undertheseanlp/vietnamese-ner-v1.4.0a2")
# undertheseaTokenize = AutoTokenizer.from_pretrained("undertheseanlp/vietnamese-ner-v1.4.0a2")

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print("Using device: %s" % (device))


def createEmptyTuple(num_tasks = num_tasks, task_id_to_num_classes = task_id_to_num_classes):
    totalLst = []
    for taskID in range(num_tasks):
        classDict = dict()
        for c in range(task_id_to_num_classes[taskID]):
            classDict[c] = (0, 0, 0)
        totalLst.append(classDict)
    return tuple(totalLst)


def lossCombine(logitsIsComparative, logitsNER, logitsComparisonType, trueIsComparative, trueNER, trueComparisonType):
    # CAN BE FURTHER IMPROVED! (TRY OTHER WAYS TO COMBINE THE LAST RETURN LOSS)
    lossIsComparative = torch.nn.CrossEntropyLoss()
    lossNER = torch.nn.CrossEntropyLoss()
    lossComparisonType = torch.nn.CrossEntropyLoss()

    logitsIsComparative = logitsIsComparative.mean(dim = 1, keepdim = False)
    logitsNER = logitsNER.reshape((-1, num_NER_labels))
    trueNER = trueNER.reshape((-1, ))
    logitsComparisonType = logitsComparisonType.mean(dim = 1, keepdim = False)

    return (lossIsComparative(logitsIsComparative, trueIsComparative) +
            lossNER(logitsNER, trueNER) +
            lossComparisonType(logitsComparisonType, trueComparisonType))


def computePredictions(logitsIsComparative, logitsNER, logitsComparisonType):
    logitsIsComparative = logitsIsComparative.mean(dim = 1, keepdim = False)
    logitsNER = logitsNER.reshape((-1, num_NER_labels))
    logitsComparisonType = logitsComparisonType.mean(dim = 1, keepdim = False)

    predIsComparative = torch.argmax(logitsIsComparative, dim = -1)
    predNER = torch.argmax(logitsNER, dim = -1)
    predComparisonType = torch.argmax(logitsComparisonType, dim = -1)

    return (predIsComparative, predNER, predComparisonType)


def createMaskIsComparative(trueIsComparative, maxSeqLen = maxSeqLen, forComparisonType = False):
    if (forComparisonType):
        return (trueIsComparative == 1)
    else:
        return (trueIsComparative.unsqueeze(dim = 1).expand((-1, maxSeqLen)).reshape((-1, )) == 1)


def createMaskNER(trueNER):
    return (trueNER != -100)


def computePredictionsAndMasks(logitsIsComparative, logitsNER, logitsComparisonType, trueIsComparative, trueNER):
    predIsComparative, predNER, predComparisonType = computePredictions(
                    logitsIsComparative, logitsNER, logitsComparisonType
                    )

    maskIsComparativeForNER = createMaskIsComparative(trueIsComparative, forComparisonType = False)
    maskIsComparativeForComparisonType = createMaskIsComparative(trueIsComparative, forComparisonType = True)
    maskNER = createMaskNER(trueNER)

    return (predIsComparative, predNER, predComparisonType,
            maskNER, maskIsComparativeForNER, maskIsComparativeForComparisonType)


def computeAccuracy(predIsComparative, predNER, predComparisonType,
                    trueIsComparative, trueNER, trueComparisonType,
                    maskIsComparativeForNER, maskIsComparativeForComparisonType, maskNER):
    countCorrectIsComparative = (predIsComparative == trueIsComparative).sum().item()
    countCorrectNER = ((predNER == trueNER) & maskIsComparativeForNER & maskNER).sum().item()
    countCorrectComparisonType = ((predComparisonType == trueComparisonType) & maskIsComparativeForComparisonType).sum().item()

    totalIsComparative = trueIsComparative.shape[0]
    totalNER = (maskIsComparativeForNER & maskNER).sum().item()
    totalComparisonType = maskIsComparativeForComparisonType.sum().item()

    accuracyIsComparative = 100 * (countCorrectIsComparative / totalIsComparative)
    accuracyNER = 100 * (countCorrectNER / totalNER)
    accuracyComparisonType = 100 * (countCorrectComparisonType / totalComparisonType)

    return accuracyIsComparative, accuracyNER, accuracyComparisonType


def computePrecision(TP, FP):
    if (np.isclose(TP + FP, 0)):
        print("WARNING: PRECISION - DIVISION BY 0!")
        return 0.0
    return TP / (TP + FP)


def computeRecall(TP, FN):
    if (np.isclose(TP + FN, 0)):
        print("WARNING: RECALL - DIVISION BY 0!")
        return 0.0
    return TP / (TP + FN)


def computeF1(TP, FP, FN):
    if (np.isclose((2 * TP) + FP + FN, 0)):
        print("WARNING: F1 - DIVISION BY 0!")
        return 0.0
    return (2 * TP) / ((2 * TP) + FP + FN)


# isComparative: 1: positive, 0: negative
def getIsComparativeTP_FP_FN(predIsComparative, trueIsComparative, PClass = 1):
    TPisComparative = ((predIsComparative == PClass) & (trueIsComparative == PClass)).sum().item()
    FPisComparative = ((predIsComparative == PClass) & (trueIsComparative != PClass)).sum().item()
    FNisComparative = ((predIsComparative != PClass) & (trueIsComparative == PClass)).sum().item()
    return TPisComparative, FPisComparative, FNisComparative


def getNERTP_FP_FN(predNER, trueNER, maskIsComparativeForNER, maskNER, PClass):
    TPNER = ((predNER == PClass) & (trueNER == PClass) & maskIsComparativeForNER & maskNER).sum().item()
    FPNER = ((predNER == PClass) & (trueNER != PClass) & maskIsComparativeForNER & maskNER).sum().item()
    FNNER = ((predNER != PClass) & (trueNER == PClass) & maskIsComparativeForNER & maskNER).sum().item()
    return TPNER, FPNER, FNNER


def getComparisonTypeTP_FP_FN(predComparisonType, trueComparisonType, maskIsComparativeForComparisonType, PClass):
    TPComparisonType = ((predComparisonType == PClass) &
                        (trueComparisonType == PClass) &
                        maskIsComparativeForComparisonType).sum().item()
    FPComparisonType = ((predComparisonType == PClass) &
                        (trueComparisonType != PClass) &
                        maskIsComparativeForComparisonType).sum().item()
    FNComparisonType = ((predComparisonType != PClass) &
                        (trueComparisonType == PClass) &
                        maskIsComparativeForComparisonType).sum().item()
    return TPComparisonType, FPComparisonType, FNComparisonType


def getIsComparativeAllClassTP_FP_FN(predIsComparative, trueIsComparative):
    isComparativeAllClass = dict()
    for c in range(num_isComparative_labels):
        isComparativeAllClass[c] = getIsComparativeTP_FP_FN(predIsComparative, trueIsComparative, PClass = c)
    return isComparativeAllClass


def getNERAllClassTP_FP_FN(predNER, trueNER, maskIsComparativeForNER, maskNER):
    NERAllClass = dict()
    for c in range(num_NER_labels - 1): # Exclude <noClass>
        NERAllClass[c] = getNERTP_FP_FN(predNER, trueNER, maskIsComparativeForNER, maskNER, PClass = c)
    return NERAllClass


def getComparisonTypeAllClassTP_FP_FN(predComparisonType, trueComparisonType, maskIsComparativeForComparisonType):
    comparisonTypeAllClass = dict()
    for c in range(num_comparisonType_labels - 1): # Exclude <noClass>
        comparisonTypeAllClass[c] = getComparisonTypeTP_FP_FN(predComparisonType, trueComparisonType,
                                                              maskIsComparativeForComparisonType, PClass = c)
    return comparisonTypeAllClass


def getAllTP_FP_FN(predIsComparative, predNER, predComparisonType,
                    trueIsComparative, trueNER, trueComparisonType,
                    maskIsComparativeForNER, maskIsComparativeForComparisonType, maskNER):
    return (
        getIsComparativeAllClassTP_FP_FN(predIsComparative, trueIsComparative),
        getNERAllClassTP_FP_FN(predNER, trueNER, maskIsComparativeForNER, maskNER),
        getComparisonTypeAllClassTP_FP_FN(predComparisonType, trueComparisonType, maskIsComparativeForComparisonType)
    )


def computeTotalTP_FP_FN(TP_FP_FN_Lst, num_tasks = num_tasks, task_id_to_num_classes = task_id_to_num_classes):
    """
    Input:

    - A list size (noIter, )

    - Each element is a tuple size (3, ) containing TP_FP_FN dictionary for tasks: isComparative, NER, comparisonType.

    - Each dictionary contains ('number of classes') key-value pairs.

    - Key is the class id. Value is a tuple size (3, ): TP, FP, FN corresponding to class.
    """
    totalTuple = createEmptyTuple(num_tasks, task_id_to_num_classes)
    for iter in range(len(TP_FP_FN_Lst)):
        for taskID in range(num_tasks):
            for c in range(task_id_to_num_classes[taskID]):
                # print(taskID, c)
                # print(totalTuple)
                # print(TP_FP_FN_Lst)
                # print(totalTuple[taskID][c])
                # print(TP_FP_FN_Lst[iter][taskID][c])
                totalTuple[taskID][c] = tuple(
                    [sum(x) for x in zip(totalTuple[taskID][c], TP_FP_FN_Lst[iter][taskID][c])]
                )
    return totalTuple


def computePrecisionRecallF1_all(totalTuple, num_tasks = num_tasks, task_id_to_num_classes = task_id_to_num_classes):
    outputMetrics = createEmptyTuple(num_tasks, task_id_to_num_classes)
    for taskID in range(num_tasks):
        for c in range(task_id_to_num_classes[taskID]):
            TP, FP, FN = totalTuple[taskID][c]
            precision, recall, f1 = computePrecision(TP, FP), computeRecall(TP, FN), computeF1(TP, FP, FN)
            outputMetrics[taskID][c] = (precision, recall, f1)

    return outputMetrics


def computeMicroAvgF1(totalTuple, num_tasks = num_tasks, task_id_to_num_classes = task_id_to_num_classes):
    outpMicroAvgF1 = []
    for taskID in range(num_tasks):
        totalTP_allClass, totalFP_allClass, totalFN_allClass = 0, 0, 0
        for c in range(task_id_to_num_classes[taskID]):
            TP, FP, FN = totalTuple[taskID][c]
            totalTP_allClass += TP
            totalFP_allClass += FP
            totalFN_allClass += FN
        microAvgF1 = computeF1(totalTP_allClass, totalFP_allClass, totalFN_allClass)
        outpMicroAvgF1.append(microAvgF1)
    return tuple(outpMicroAvgF1)


def computeAllFinalMetrics(TP_FP_FN_Lst, num_tasks = num_tasks, task_id_to_num_classes = task_id_to_num_classes):
    totalTuple = computeTotalTP_FP_FN(TP_FP_FN_Lst, num_tasks, task_id_to_num_classes)
    outpPRF1_all = computePrecisionRecallF1_all(totalTuple, num_tasks, task_id_to_num_classes)
    outpMicroAvgF1_all = computeMicroAvgF1(totalTuple, num_tasks, task_id_to_num_classes)
    return (outpPRF1_all, outpMicroAvgF1_all)

# def computeIsComparativePrecisionRecallF1(predIsComparative, trueIsComparative, PClass = 1):
#     TP, FP, FN = getIsComparativeTP_FP_FN(predIsComparative, trueIsComparative, PClass)
#     precision = computePrecision(TP, FP)
#     recall = computeRecall(TP, FN)
#     f1 = computeF1(TP, FP, FN)
#     return (precision, recall, f1)


# def computeNERPrecisionRecallF1(predNER, trueNER, maskIsComparativeForNER, maskNER, PClass):
#     TP, FP, FN = getNERTP_FP_FN(predNER, trueNER, maskIsComparativeForNER, maskNER, PClass)
#     precision = computePrecision(TP, FP)
#     recall = computeRecall(TP, FN)
#     f1 = computeF1(TP, FP, FN)
#     return (precision, recall, f1)
#
#
# def computeComparisonTypePrecisionRecallF1(predComparisonType, trueComparisonType,
#                                            maskIsComparativeForComparisonType, PClass):
#     TP, FP, FN = getComparisonTypeTP_FP_FN(predComparisonType, trueComparisonType,
#                                            maskIsComparativeForComparisonType, PClass)
#     precision = computePrecision(TP, FP)
#     recall = computeRecall(TP, FN)
#     f1 = computeF1(TP, FP, FN)
#     return (precision, recall, f1)


# def computeNERMetricsAll(predNER, trueNER, maskIsComparativeForNER, maskNER):
#     NERMetrics = dict()
#     for c in range(num_NER_labels - 1): # Exclude <noClass>
#         NERMetrics[c] = computeNERPrecisionRecallF1(predNER, trueNER, maskIsComparativeForNER, maskNER, PClass = c)
#     return NERMetrics
#
#
# def computeComparisonTypeMetricsAll(predComparisonType, trueComparisonType, maskIsComparativeForComparisonType):
#     comparisonTypeMetrics = dict()
#     for c in range(num_comparisonType_labels - 1): # Exclude <noClass>
#         comparisonTypeMetrics[c] = computeComparisonTypePrecisionRecallF1(predComparisonType, trueComparisonType,
#                                                                           maskIsComparativeForComparisonType, PClass = c)
#     return comparisonTypeMetrics


def forward(batch, model, device, mode = 'training'):
    batchDict = {k: v.to(device) for k, v in batch.items()}
    batchDict['input_ids'] = torch.squeeze(batchDict['input_ids'], dim = 1)
    batchDict['attention_mask'] = torch.squeeze(batchDict['attention_mask'], dim = 1)
    batchDict['labelNER'] = torch.squeeze(batchDict['labelNER'], dim = 1)
    batchDict['labelIsComparative'] = torch.squeeze(batchDict['labelIsComparative'], dim = 1)
    batchDict['labelComparisonType'] = torch.squeeze(batchDict['labelComparisonType'], dim = 1)

    outputs = model(input_ids = batchDict['input_ids'], attention_mask = batchDict['attention_mask'])
    logitsIsComparative = outputs.logits[:, :, :2]
    logitsNER = outputs.logits[:, :, 2:7]
    logitsComparisonType = outputs.logits[:, :, 7:]

    if (mode == 'training'):
        trueIsComparative = batchDict['labelIsComparative']
        trueNER = batchDict['labelNER']
        trueComparisonType = batchDict['labelComparisonType']

        return (logitsIsComparative, logitsNER, logitsComparisonType,
                trueIsComparative, trueNER, trueComparisonType)

    elif (mode == 'inference'):
        return (logitsIsComparative, logitsNER, logitsComparisonType)


def evaluate(model, valLoader, lossCombine, device):

    model.eval()
    totalLoss = 0
    totalAccuracyIsComparative = 0
    totalAccuracyNER = 0
    totalAccuracyComparisonType = 0
    TP_FP_FN_Lst = []

    for i, batch in enumerate(valLoader):
        with (torch.no_grad()):
            (logitsIsComparative, logitsNER, logitsComparisonType,
             trueIsComparative, trueNER, trueComparisonType) = forward(batch, model, device)
            loss = lossCombine(logitsIsComparative, logitsNER, logitsComparisonType,
                               trueIsComparative, trueNER, trueComparisonType)

            trueNER = trueNER.reshape((-1, ))

            (predIsComparative, predNER, predComparisonType,
             maskNER, maskIsComparativeForNER, maskIsComparativeForComparisonType) = computePredictionsAndMasks(
                logitsIsComparative, logitsNER, logitsComparisonType, trueIsComparative, trueNER
            )

            accuracyIsComparative, accuracyNER, accuracyComparisonType = computeAccuracy(
                predIsComparative, predNER, predComparisonType,
                trueIsComparative, trueNER, trueComparisonType,
                maskIsComparativeForNER, maskIsComparativeForComparisonType, maskNER
            )

            TP_FP_FN_Lst.append(getAllTP_FP_FN(
                predIsComparative, predNER, predComparisonType,
                trueIsComparative, trueNER, trueComparisonType,
                maskIsComparativeForNER, maskIsComparativeForComparisonType, maskNER
            ))

            totalAccuracyIsComparative += accuracyIsComparative
            totalAccuracyNER += accuracyNER
            totalAccuracyComparisonType += accuracyComparisonType
            totalLoss += loss
    noBatch = len(valLoader)
    if (noBatch != len(TP_FP_FN_Lst)):
        print("WARNING EVALUATION PROCESS: METRICS CALCULATIONS DO NOT MATCH!")
    else:
        PRF1_all, microAvgF1_all = computeAllFinalMetrics(TP_FP_FN_Lst)

    return (totalLoss / noBatch,
            totalAccuracyIsComparative / noBatch, totalAccuracyNER / noBatch, totalAccuracyComparisonType / noBatch,
            PRF1_all, microAvgF1_all)


def train(model, num_epochs, trainLoader, valLoader, optimizer, device, **kwargs):

    lr_scheduler = False
    for key in kwargs.keys():
        if (key == 'lr_scheduler'):
            lr_scheduler = True

    model.to(device)
    model.train()

    lossAccLst = []

    for epoch in range(num_epochs):

        totalLoss = 0
        totalAccuracyIsComparative = 0
        totalAccuracyNER = 0
        totalAccuracyComparisonType = 0
        TP_FP_FN_Lst = []

        with (tqdm(trainLoader, unit = 'batch') as tqdmTrainLoader):
            for i, batch in enumerate(tqdmTrainLoader):
                tqdmTrainLoader.set_description("Epoch %04d, iteration %d" % (epoch + 1, i + 1))

                (logitsIsComparative, logitsNER, logitsComparisonType,
                 trueIsComparative, trueNER, trueComparisonType) = forward(batch, model, device)
                loss = lossCombine(logitsIsComparative, logitsNER, logitsComparisonType,
                                   trueIsComparative, trueNER, trueComparisonType)

                loss.backward()
                optimizer.step()
                if (lr_scheduler):
                    kwargs['lr_scheduler'].step()

                trueNER = trueNER.reshape((-1, ))

                (predIsComparative, predNER, predComparisonType,
                maskNER, maskIsComparativeForNER, maskIsComparativeForComparisonType) = computePredictionsAndMasks(
                    logitsIsComparative, logitsNER, logitsComparisonType, trueIsComparative, trueNER
                )

                accuracyIsComparative, accuracyNER, accuracyComparisonType = computeAccuracy(
                    predIsComparative, predNER, predComparisonType,
                    trueIsComparative, trueNER, trueComparisonType,
                    maskIsComparativeForNER, maskIsComparativeForComparisonType, maskNER
                )

                TP_FP_FN_Lst.append(getAllTP_FP_FN(
                    predIsComparative, predNER, predComparisonType,
                    trueIsComparative, trueNER, trueComparisonType,
                    maskIsComparativeForNER, maskIsComparativeForComparisonType, maskNER
                ))

                # (valLoss, valAccuracyIsComparative, valAccuracyNER, valAccuracyComparisonType,
                #  valPRF1_all, valMicroAvgF1_all) = evaluate(model, valLoader, lossCombine, device)

                tqdmTrainLoader.set_postfix(
                    training_loss = loss.item(),
                    training_accuracy_isComparative = accuracyIsComparative,
                    training_accuracy_NER = accuracyNER,
                    training_accuracy_comparisonType = accuracyComparisonType,
                    # validation_loss = valLoss.item(),
                    # validation_accuracy_isComparative = valAccuracyIsComparative,
                    # validation_accuracy_NER = valAccuracyNER,
                    # validation_accuracy_comparisonType = valAccuracyComparisonType,
                )

                totalAccuracyIsComparative += accuracyIsComparative
                totalAccuracyNER += accuracyNER
                totalAccuracyComparisonType += accuracyComparisonType
                totalLoss += loss.item()
                optimizer.zero_grad()

        noBatch = len(trainLoader)
        if (noBatch != len(TP_FP_FN_Lst)):
            print("WARNING TRAINING PROCESS: METRICS CALCULATIONS DO NOT MATCH!")
        else:
            PRF1_all, microAvgF1_all = computeAllFinalMetrics(TP_FP_FN_Lst)

        print(
            'Accuracy isComparative = %.4f, accuracy NER = %.4f, accuracy comparisonType = %.4f, loss = %.4f'
            % (totalAccuracyIsComparative / noBatch, totalAccuracyNER / noBatch, totalAccuracyComparisonType / noBatch,
               totalLoss / noBatch)
        )

        lossAccLst.append((
            (totalLoss / noBatch,
             totalAccuracyIsComparative / noBatch, totalAccuracyNER / noBatch, totalAccuracyComparisonType / noBatch),
            # (valLoss.item(),
            #  valAccuracyIsComparative, valAccuracyNER, valAccuracyComparisonType)
        ))
    return lossAccLst


def getInferencePredictions(model, dataLoader, device, maxSeqLen = maxSeqLen):

    model.to(device)
    model.eval()

    allPredConcat = []

    for i, batch in enumerate(dataLoader):
        with (torch.no_grad()):
            (logitsIsComparative, logitsNER, logitsComparisonType) = forward(batch, model, device, mode = 'inference')
            (predIsComparative, predNER, predComparisonType) = computePredictions(
                logitsIsComparative, logitsNER, logitsComparisonType
            )
            predIsComparative = predIsComparative.reshape((-1, 1))
            predNER = predNER.reshape((-1, maxSeqLen))
            predComparisonType = predComparisonType.reshape((-1, 1))

            predConcat = torch.concat([predIsComparative, predNER, predComparisonType], dim=1)
            allPredConcat.append(predConcat)

    allPredConcat = torch.concat(allPredConcat)
    return (allPredConcat[:, 0].reshape((-1, 1)), allPredConcat[:, 1 : 193], allPredConcat[:, 193].reshape((-1, 1)))


def loadModel(model, path):
    model.load_state_dict(torch.load(path))
    model.eval()


def filterPredictions(nerPhoBERTTorchDataset, allPredIsComparative, allPredNER, allPredComparisonType):

    allAttentionMask = nerPhoBERTTorchDataset.__getitem__(range(allPredIsComparative.shape[0]))['attention_mask']

    # allInputIDs = nerPhoBERTTorchDataset.__getitem__(range(allPredIsComparative.shape[0]))['input_ids']
    # allInputIDs = allInputIDs * allAttentionMask

    allAttentionMask = allAttentionMask.cuda()
    # allInputIDs = allInputIDs.cuda()

    allPredNER = torch.where(allPredIsComparative.expand(allPredNER.shape) == 1, allPredNER, 4)
    allPredNER = torch.where(allAttentionMask == 1, allPredNER, 4)
    allPredNER = torch.where(allPredComparisonType.expand(allPredNER.shape) != 8, allPredNER, 4)

    allPredComparisonType = torch.where(allPredIsComparative == 1, allPredComparisonType, 8)

    return (allPredIsComparative, allPredNER, allPredComparisonType)


def postProcessNERPredictions(allPredNER, allPredIsComparative, wordIndTAP, maxSeqLen = maxSeqLen):
    postProcessNER = []
    # postProcessWordIndTAP = []
    for sentenceInd in range(allPredIsComparative.shape[0]):
        prevWordInd = -1
        tempPostNER = []
        tempPostWordIndTAP = []
        for tokInd in range(maxSeqLen):
            if (wordIndTAP[sentenceInd][tokInd] is None):
                pass
            elif (isinstance(wordIndTAP[sentenceInd][tokInd], int)):
                tempPostWordIndTAP.append(wordIndTAP[sentenceInd][tokInd])
                if (prevWordInd == wordIndTAP[sentenceInd][tokInd]):
                    pass
                else:
                    tempPostNER.append(allPredNER[sentenceInd, tokInd].item())
                    prevWordInd = wordIndTAP[sentenceInd][tokInd]
            else:
                print("WARNING: WORD IDS NOT WORKING PROPERLY")
        postProcessNER.append(deepcopy(tempPostNER))
        # postProcessWordIndTAP.append(deepcopy(tempPostWordIndTAP))

    return postProcessNER


def getOutputNER(postProcessNER, segmentedOriginalSentence):
    outputNER = []
    for sentenceInd in range(len(segmentedOriginalSentence)):
        tempOutputNER = []
        segmentedWordList = segmentedOriginalSentence.loc[sentenceInd].split(" ")
        countSpecial = 0

        for segmentedWordInd in range(len(segmentedWordList)):
            segmentedSubWordList = segmentedWordList[segmentedWordInd].split("_")

            for count in range(len(segmentedSubWordList)):
                if (segmentedSubWordList[count] == '\xa0'):
                    tempOutputNER.append(4)
                    countSpecial += 1
                else:
                    tempOutputNER.append(postProcessNER[sentenceInd][segmentedWordInd - countSpecial])

        outputNER.append(tempOutputNER)

    return outputNER


def getSentenceListForPostProcess(segmentedOriginalSentence):
    sentenceListForPostProcess = []
    for sentenceInd in range(len(segmentedOriginalSentence)):
        sentenceListForPostProcess.append(segmentedOriginalSentence[sentenceInd].replace('_', ' ').split(' '))

    return sentenceListForPostProcess


def getOutputForTxt(outputNER, allPredComparisonType, sentenceListForPostProcess):
    outputForTxt = []

    # outputNER, allPredComparisonType, sentenceListForPostProcess,
    for sentenceInd in range(len(sentenceListForPostProcess)):
        outpTxtTemp = {0: [], 1: [], 2: [], 3: [], 100: None}
        sentence = sentenceListForPostProcess[sentenceInd]
        outputNERForSentence = outputNER[sentenceInd]

        if (len(outputNERForSentence) != len(sentence)):
            print("WARNING: POST PROCESS DOES NOT MATCH")

        outpTxtTemp[100] = allPredComparisonType[sentenceInd].item()
        if (outpTxtTemp[100] != 8):
            for wordInd in range(len(sentence)):
                if (outputNERForSentence[wordInd] != 4):
                    outpTxtTemp[outputNERForSentence[wordInd]].append((wordInd + 1, sentence[wordInd]))
        outputForTxt.append(outpTxtTemp)

    return outputForTxt


def getsentenceIndCDNCCount(sentenceIndCDNC):
    sentenceIndCDNCCount = []
    i = 0
    while (sentenceIndCDNC.count(i) != 0):
        sentenceIndCDNCCount.append(sentenceIndCDNC.count(i))
        i += 1
    return sentenceIndCDNCCount


def writeOutputTxt(outputForTxt, sentenceIndCDNCCount, pathOriginal, splitName, pathOut="/outputTxt", curDir=curDir,
                   idx=range(1, 25, 1)):
    os.system("rm -r .%s/*" % (pathOut))
    countLine = 0
    countAll = 0

    for fileInd in idx:
        fileName = "/" + splitName + "_%04d.txt" % (fileInd)
        fileNameOriginal = curDir + pathOriginal + fileName
        fileNameOut = curDir + pathOut + fileName

        with open(fileNameOut, 'x', encoding='utf8') as outpTxtFile, \
                open(fileNameOriginal, 'r', encoding='utf8') as originalTxtFile:

            linesInOriginal = originalTxtFile.readlines()

            for lineOriginal in linesInOriginal:
                lineOriginalStripped = lineOriginal.strip("\n")

                if (lineOriginalStripped == ''):
                    pass
                else:
                    outpTxtFile.write("%s\n" % (lineOriginalStripped))

                    # print(repr(lineOriginal))
                    for sentenceIndCount in range(sentenceIndCDNCCount[countLine]):
                        outpDict = outputForTxt[countAll]

                        if ((((len(outpDict[0]) == 0) &
                              (len(outpDict[1]) == 0) &
                              (len(outpDict[2]) == 0) &
                              (len(outpDict[3]) == 0))) | (outpDict[100] == 8)):
                            pass

                        else:
                            tempDict = dict()
                            for nerLab in range(4):

                                tempDictVal = []
                                for elem in outpDict[nerLab]:
                                    outpStr = str(elem[0]) + "&&" + elem[1]
                                    tempDictVal.append(outpStr)

                                tempDict[id2label[nerLab]] = deepcopy(tempDictVal)
                            tempDict["label"] = id2comparisonLabel[outpDict[100]]

                            outpTxtFile.write("%s\n" % (json.dumps(tempDict, ensure_ascii=False)))

                        countAll += 1
                    outpTxtFile.write("\n")

                    countLine += 1

    print('Write output finished!')
    pass




