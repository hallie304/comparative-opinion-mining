# %cd
# %cd naTtahN_T1
# %cd VLSP23_Task1
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
from utils.dataUtils import *

dataCSV = txt2csv(path = "/datasets/original/VLSP2023_ComOM_public_test_nolabel/VLSP2023_ComOM_public_test_nolabel",
                  curDir = curDir, splitName = 'train', version = 'old')
dataCSVIsComparative, dataCSVNotIsComparative = createDataNERCSV(dataCSV, mode = 'train')
datasetNERTokenizedCSV = tokenizeAndProcess(dataCSVIsComparative, dataCSVNotIsComparative, mode = 'train')
nerPhoBERTTorchDataset = DataNERPhoBERTTorch(datasetNERTokenizedCSV)


num_epochs = 10
batchSize = 64
trainLoader, valLoader, testLoader = splitDataset(nerPhoBERTTorchDataset, 0.95, 0.05, 0.0, batchSize, True, 14)

optimizer = AdamW(phobertTokenClassification.parameters(), lr = 5e-5)

num_training_steps = num_epochs * len(trainLoader)
lr_scheduler = get_scheduler(
    name = "linear", optimizer = optimizer, num_warmup_steps = 0, num_training_steps = num_training_steps
)

exit()
trainLog = train(phobertTokenClassification, num_epochs, trainLoader, valLoader, optimizer, device, lr_scheduler = lr_scheduler)
torch.save(phobertTokenClassification.state_dict(), "phobertFinetuned.pt")
evalOnValSet(phobertTokenClassification, testLoader, lossCombine, device)


# 0 - train, 1 - val 
# 0 - loss, 1 - isComparative, 2 = NER, 3 = comparisonType
plotData = []
plotData1 = []
plotInfo = 0
for i in range(num_epochs):
    plotData.append(trainLog[i][0][plotInfo])
    plotData1.append(trainLog[i][1][plotInfo])
plt.plot(plotData)
plt.plot(plotData1)
plt.show()
plt.savefig("lossGraph.png")


