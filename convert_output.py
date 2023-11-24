import glob
import torch
import json
import os

files = glob.glob("raw_output/*")
os.mkdir("processed_output")
output_path = "processed_output/"

# Read all outputs
results = {}
rows = {}
index = 0
new_lines = []
for file in files:
    lines = open(file, "r").readlines()
    for line in lines:
        if "predicate" in line:
            json.loads(line)
            new_lines.append(line)
        elif "\n" == line:
            rows[index] = "".join(new_lines)
            index += 2
            new_lines = []

    results[file.split("/")[-1]] = rows
    new_lines = []
    index = 0
    rows = {}

for file in glob.glob("data/private_test/*.txt"):
    original_texts = open(file).readlines()
    file_name = file.split("/")[-1]
    submit_file = output_path + file_name

    output_lines = results[file_name]
    output_texts = []
    # Loop through each row of original texts
    # 1. add output below it
    for index, row in enumerate(original_texts):
        output_texts.append(row)
        if " " in row:
            output_texts.append(output_lines[index])

    with open(submit_file, 'w') as f:
        f.write("".join(output_texts))