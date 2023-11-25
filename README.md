# VLSP 2023: Comparative Opinion Mining from Vietnamese Product Reviews
Finetune multiple pre-trained Transformer-based models to solve the challenge of Comparative Opinion Mining from Vietnamese Product Reviews in the VLSP2023 shared task.
## Recreating the result
### Cloining the repository, install the requirements, and set up the files
```
git clone https://github.com/hallie304/VLSP23-Comparative-Opinion-Mining.git
cd VLSP23-Comparative-Opinion-Mining
pip install -r requirements.txt
```

For confidentiality reasons, we cannot provide the data used in this project as it is the property of the VLSP2023 competition. Please check VLSP's website to see if the data is freely available to be downloaded, if it is, download it and put it in a folder called "data".

Within the models/task1_and_task3.py and train.py files, there a line that denotes the current working directory
```
working_directory = "/home/group2/group1/github_test"
```

Please change this line to the directory where you cloned the repository.
### Preprocessing data
Run the total_generate.py file to upsample the data, and convert them into one .csv file to be used to train
```
python preprocess/Generate_process/total_generate.py
```
This will create a file called "Upsample_data.csv" in preprocess/Generate_process. This file will be used to train the models.
### Training the models
To train the models, run the train.py file. This will create three folders, one for each model, and save the model weights in them. The folders are named "task1", "task2", and "task3"
```
python train.py
```

### Generate the raw predictions
To generate the raw predictions, run the generate_output.py file. This will create a folder call raw_output that contains the raw output
```
python generate_output.py
```

### Generate the final predictions
To generate the final predictions, run the convert_output.py file. This will create a folder called processed_output. In this folder are the final output files
```
python convert_output.py
```

## Technical Report
Link will be added
