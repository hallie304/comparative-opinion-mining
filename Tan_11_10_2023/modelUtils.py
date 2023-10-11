import torch.nn

from libsAndPackages import *

num_isComparative_labels = 2
num_NER_labels = 5
num_comparisonType_labels = 9
num_labels = num_isComparative_labels + num_NER_labels + num_comparisonType_labels

rdrSegmenter = py_vncorenlp.VnCoreNLP(annotators = ["wseg"], save_dir = curDir + "/vncorenlp")

phobertFeatureExtractor = AutoModel.from_pretrained("vinai/phobert-base")
phobertTokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base")
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

    return predIsComparative, predNER, predComparisonType


def computeAccuracy(predIsComparative, predNER, predComparisonType,
                    trueIsComparative, trueNER, trueComparisonType, maskNER):

    countCorrectIsComparative = (predIsComparative == trueIsComparative).sum()
    countCorrectNER = ((predNER == trueNER) & maskNER).sum()
    countCorrectComparisonType = (predComparisonType == trueComparisonType).sum()

    totalIsComparative = trueIsComparative.shape[0]
    totalNER = maskNER.sum()
    totalComparisonType = trueComparisonType.shape[0]

    accuracyIsComparative = 100 * (countCorrectIsComparative / totalIsComparative).item()
    accuracyNER = 100 * (countCorrectNER / totalNER).item()
    accuracyComparisonType = 100 * (countCorrectComparisonType / totalComparisonType).item()

    return accuracyIsComparative, accuracyNER, accuracyComparisonType


def evalOnValSet(model, valLoader, lossCombine, device):
    model.eval()
    totalLoss = 0
    totalAccuracyIsComparative = 0
    totalAccuracyNER = 0
    totalAccuracyComparisonType = 0
    for i, batch in enumerate(valLoader):
        with torch.no_grad():
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
            trueIsComparative = batchDict['labelIsComparative']
            trueNER = batchDict['labelNER']
            trueComparisonType = batchDict['labelComparisonType']
            loss = lossCombine(logitsIsComparative, logitsNER, logitsComparisonType,
                               trueIsComparative, trueNER, trueComparisonType)

            predIsComparative, predNER, predComparisonType = computePredictions(
                logitsIsComparative, logitsNER, logitsComparisonType
            )

            trueNER = trueNER.reshape((-1, ))
            maskNER = (trueNER != -100)

            accuracyIsComparative, accuracyNER, accuracyComparisonType = computeAccuracy(
                predIsComparative, predNER, predComparisonType,
                trueIsComparative, trueNER, trueComparisonType, maskNER
            )

            totalAccuracyIsComparative += accuracyIsComparative
            totalAccuracyNER += accuracyNER
            totalAccuracyComparisonType += accuracyComparisonType
            totalLoss += loss
    return (totalLoss / (i + 1),
            totalAccuracyIsComparative / (i + 1), totalAccuracyNER / (i + 1), totalAccuracyComparisonType / (i + 1))


def train(model, num_epochs, trainLoader, valLoader, optimizer, device, **kwargs):
    lr_scheduler = False
    for key in kwargs.keys():
        if (key == 'lr_scheduler'):
            lr_scheduler = True

    model.to(device)

    model.train()
    for epoch in range(num_epochs):
        totalAccuracyIsComparative = 0
        totalAccuracyNER = 0
        totalAccuracyComparisonType = 0
        totalLoss = 0
        with tqdm(trainLoader, unit = 'batch') as tqdmTrainLoader:
            for i, batch in enumerate(tqdmTrainLoader):
                tqdmTrainLoader.set_description("Epoch %04d, iteration %d" % (epoch + 1, i + 1))

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
                trueIsComparative = batchDict['labelIsComparative']
                trueNER = batchDict['labelNER']
                trueComparisonType = batchDict['labelComparisonType']
                loss = lossCombine(logitsIsComparative, logitsNER, logitsComparisonType,
                                   trueIsComparative, trueNER, trueComparisonType)

                loss.backward()

                optimizer.step()
                if (lr_scheduler):
                    kwargs['lr_scheduler'].step()

                predIsComparative, predNER, predComparisonType = computePredictions(logitsIsComparative, logitsNER, logitsComparisonType)
                trueNER = trueNER.reshape((-1, ))
                maskNER = (trueNER != -100)

                accuracyIsComparative, accuracyNER, accuracyComparisonType = computeAccuracy(
                    predIsComparative, predNER, predComparisonType,
                    trueIsComparative, trueNER, trueComparisonType, maskNER
                )
                valLoss, valAccuracyIsComparative, valAccuracyNER, valAccuracyComparisonType = evalOnValSet(
                    model, valLoader, lossCombine, device
                )

                tqdmTrainLoader.set_postfix(
                    training_loss = loss.item(),
                    training_accuracy_isComparative = accuracyIsComparative,
                    training_accuracy_NER = accuracyNER,
                    training_accuracy_comparisonType = accuracyComparisonType,
                    validation_loss = valLoss.item(),
                    validation_accuracy_isComparative = valAccuracyIsComparative,
                    validation_accuracy_NER = valAccuracyNER,
                    validation_accuracy_comparisonType = valAccuracyComparisonType
                )

                totalAccuracyIsComparative += accuracyIsComparative
                totalAccuracyNER += accuracyNER
                totalAccuracyComparisonType += accuracyComparisonType
                totalLoss += loss.item()
                optimizer.zero_grad()
        print(
            'Accuracy isComparative = %.4f, accuracy NER = %.4f, accuracy comparisonType = %.4f, loss = %.4f'
            % (totalAccuracyIsComparative / (i + 1), totalAccuracyNER / (i + 1), totalAccuracyComparisonType / (i + 1),
               totalLoss / (i + 1))
        )



