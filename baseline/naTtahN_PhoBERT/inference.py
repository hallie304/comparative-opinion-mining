# %cd
# %cd naTtahN_T1
# %cd VLSP23-Comparative-Opinion-Mining
# %cd
# 
# !git clone --single-branch --branch fast_tokenizers_BARTpho_PhoBERT_BERTweet https://github.com/datquocnguyen/transformers.git
# %cd transformers
# %pip install -e .
# 
# %pip install pandas
# %pip install evaluate
# %pip install py_vncorenlp
# # %pip install underthesea[deep]
# # %pip install underthesea
# %pip install seqeval
# %pip install accelerate
# %pip install matplotlib

import ast

import numpy as np
import pandas as pd
import torch.utils.data
import os

# print(torch.cuda.is_available())
# print(torch.cuda.device_count())
# print(torch.cuda.get_device_name(0))
# os.chdir('/home/group2/naTtahN_T1/VLSP23_Task1')

# from libsAndPackages import *
# from modelUtils import *
from utilsFunc.dataUtils import *
os.chdir(curDir)

dataCSV = txt2csv(path = "/datasets/original/VLSP2023_ComOM_public_test_nolabel/VLSP2023_ComOM_public_test_nolabel",
                  curDir = curDir, splitName = 'dev', version = 'new', idx = range(1, 25, 1))
dataCSVIsComparative, dataCSVNotIsComparative, sentenceIndCDNC = createDataNERCSV(dataCSV, mode = 'inference')
datasetNERTokenizedCSV, wordIndTAP = tokenizeAndProcess(dataCSVIsComparative, dataCSVNotIsComparative, mode = 'inference')
nerPhoBERTTorchDataset = DataNERPhoBERTTorch(datasetNERTokenizedCSV)

batchSize = 64
inferenceLoader, _, _ = splitDataset(nerPhoBERTTorchDataset, 1.0, 0.0, 0.0,
                                     batchSize, False, 14)
loadModel(phobertTokenClassification, 'phobertFinetuned.pt')

(allPredIsComparative, allPredNER, allPredComparisonType) = getInferencePredictions(phobertTokenClassification,
                                                                                    inferenceLoader, device)
(allPredIsComparative, allPredNER, allPredComparisonType) = filterPredictions(
    nerPhoBERTTorchDataset, allPredIsComparative, allPredNER, allPredComparisonType
)

postProcessNER = postProcessNERPredictions(allPredNER, allPredIsComparative, wordIndTAP)

segmentedOriginalSentence = deepcopy(dataCSVNotIsComparative['Input sentence segmented'])

outputNER = getOutputNER(postProcessNER, segmentedOriginalSentence)
sentenceListForPostProcess = getSentenceListForPostProcess(segmentedOriginalSentence)
outputForTxt = getOutputForTxt(outputNER, allPredComparisonType, sentenceListForPostProcess)

sentenceIndCDNCCount = getsentenceIndCDNCCount(sentenceIndCDNC)

writeOutputTxt(outputForTxt, sentenceIndCDNCCount,
               pathOriginal = "/datasets/original/VLSP2023_ComOM_public_test_nolabel/VLSP2023_ComOM_public_test_nolabel",
               splitName = "dev", pathOut = "/outputTxt", curDir = curDir, idx = range(1, 25, 1))



