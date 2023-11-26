import new_data_sampling_A
import new_data_sampling_B
import data_utils
import pandas as pd
import os

os.mkdir('Generate')
new_data_sampling_A.write_total_sampling('Generate/generate.txt')
''' New data records according to number of json '''
usable_gen_data = data_utils.preprocessing_with_BIO('Generate/generate.txt')
df_gen1_mono = pd.DataFrame(columns=["content", "comparative", "subject", "object", "aspect", "predicate", "label", "NER"])
df_gen1_multi = pd.DataFrame(columns=["content", "comparative", "subject", "object", "aspect", "predicate", "label", "NER"])
df_gen2_mono = pd.DataFrame(columns=["content", "comparative", "subject", "object", "aspect", "predicate", "label", "NER"])
df_gen2_multi = pd.DataFrame(columns=["content", "comparative", "subject", "object", "aspect", "predicate", "label", "NER"])
df_gen3_mono = pd.DataFrame(columns=["content", "comparative", "subject", "object", "aspect", "predicate", "label", "NER"])
df_gen3_multi = pd.DataFrame(columns=["content", "comparative", "subject", "object", "aspect", "predicate", "label", "NER"])
df_gen4_mono = pd.DataFrame(columns=["content", "comparative", "subject", "object", "aspect", "predicate", "label", "NER"])
df_gen4_multi = pd.DataFrame(columns=["content", "comparative", "subject", "object", "aspect", "predicate", "label", "NER"])
df_gen5_mono = pd.DataFrame(columns=["content", "comparative", "subject", "object", "aspect", "predicate", "label", "NER"])
df_gen5_multi = pd.DataFrame(columns=["content", "comparative", "subject", "object", "aspect", "predicate", "label", "NER"])
df_gen6_mono = pd.DataFrame(columns=["content", "comparative", "subject", "object", "aspect", "predicate", "label", "NER"])
df_gen6_multi = pd.DataFrame(columns=["content", "comparative", "subject", "object", "aspect", "predicate", "label", "NER"])
df_gen7_mono = pd.DataFrame(columns=["content", "comparative", "subject", "object", "aspect", "predicate", "label", "NER"])
df_gen7_multi = pd.DataFrame(columns=["content", "comparative", "subject", "object", "aspect", "predicate", "label", "NER"])
df_gen8_mono = pd.DataFrame(columns=["content", "comparative", "subject", "object", "aspect", "predicate", "label", "NER"])
df_gen8_multi = pd.DataFrame(columns=["content", "comparative", "subject", "object", "aspect", "predicate", "label", "NER"])

for i in range(len(usable_gen_data)):
    if len(usable_gen_data.iloc[i,6]) == 1 and 1 in usable_gen_data.iloc[i,6]:
        df_gen1_mono = df_gen1_mono._append(usable_gen_data.iloc[i])
    if len(usable_gen_data.iloc[i,6]) > 1 and 1 in usable_gen_data.iloc[i,6]:
        df_gen1_multi = df_gen1_multi._append(usable_gen_data.iloc[i])
    if len(usable_gen_data.iloc[i,6]) == 1 and 2 in usable_gen_data.iloc[i,6]:
        df_gen2_mono = df_gen2_mono._append(usable_gen_data.iloc[i])
    if len(usable_gen_data.iloc[i,6]) > 1 and 2 in usable_gen_data.iloc[i,6]:
        df_gen2_multi = df_gen2_multi._append(usable_gen_data.iloc[i])
    if len(usable_gen_data.iloc[i,6]) == 1 and 3 in usable_gen_data.iloc[i,6]:
        df_gen3_mono = df_gen3_mono._append(usable_gen_data.iloc[i])
    if len(usable_gen_data.iloc[i,6]) > 1 and 3 in usable_gen_data.iloc[i,6]:
        df_gen3_multi = df_gen3_multi._append(usable_gen_data.iloc[i])
    if len(usable_gen_data.iloc[i,6]) == 1 and 4 in usable_gen_data.iloc[i,6]:
        df_gen4_mono = df_gen4_mono._append(usable_gen_data.iloc[i])
    if len(usable_gen_data.iloc[i,6]) > 1 and 4 in usable_gen_data.iloc[i,6]:
        df_gen4_multi = df_gen4_multi._append(usable_gen_data.iloc[i])
    if len(usable_gen_data.iloc[i,6]) == 1 and 5 in usable_gen_data.iloc[i,6]:
        df_gen5_mono = df_gen5_mono._append(usable_gen_data.iloc[i])
    if len(usable_gen_data.iloc[i,6]) > 1 and 5 in usable_gen_data.iloc[i,6]:
        df_gen5_multi = df_gen5_multi._append(usable_gen_data.iloc[i])
    if len(usable_gen_data.iloc[i,6]) == 1 and 6 in usable_gen_data.iloc[i,6]:
        df_gen6_mono = df_gen6_mono._append(usable_gen_data.iloc[i])
    if len(usable_gen_data.iloc[i,6]) > 1 and 6 in usable_gen_data.iloc[i,6]:
        df_gen6_multi = df_gen6_multi._append(usable_gen_data.iloc[i])
    if len(usable_gen_data.iloc[i,6]) == 1 and 7 in usable_gen_data.iloc[i,6]:
        df_gen7_mono = df_gen7_mono._append(usable_gen_data.iloc[i])
    if len(usable_gen_data.iloc[i,6]) > 1 and 7 in usable_gen_data.iloc[i,6]:
        df_gen7_multi = df_gen7_multi._append(usable_gen_data.iloc[i])
    if len(usable_gen_data.iloc[i,6]) == 1 and 8 in usable_gen_data.iloc[i,6]:
        df_gen8_mono = df_gen8_mono._append(usable_gen_data.iloc[i])
    if len(usable_gen_data.iloc[i,6]) > 1 and 8 in usable_gen_data.iloc[i,6]:
        df_gen8_multi = df_gen8_multi._append(usable_gen_data.iloc[i])

df_gen1 = df_gen1_mono.sample(505,random_state=50)._append(df_gen1_multi.sample(201,random_state=50))
df_gen2 = df_gen2_mono.sample(385,random_state=50)._append(df_gen2_multi.sample(153,random_state=50))
df_gen3 = df_gen3_mono.sample(488,random_state=50)._append(df_gen3_multi.sample(195,random_state=50))
df_gen4 = df_gen4_mono.sample(534,random_state=50)._append(df_gen4_multi.sample(213,random_state=50))
df_gen5 = df_gen5_mono.sample(665,random_state=50)
df_gen6 = df_gen6_mono.sample(283,random_state=50)._append(df_gen6_multi.sample(113,random_state=50))
df_gen7 = df_gen7_mono.sample(480,random_state=50)._append(df_gen7_multi.sample(192,random_state=50))
df_gen8 = df_gen8_mono.sample(521,random_state=50)._append(df_gen8_multi.sample(208,random_state=50))

df_gen_total = df_gen1._append(df_gen2)._append(df_gen3)._append(df_gen4)._append(df_gen5)._append(df_gen6)._append(df_gen7)._append(df_gen8)


''' Non-comparative adding '''
non_comparative = []
with open('../../data/predicate list/Non_comparative.txt','r',encoding='utf-8') as file:
    r = file.readline()
    while r != '':
        non_comparative.append(r)
        r = file.readline()
data_frame_non_comparative = {"content": [], "comparative": [], "subject": [], "object": [], "aspect": [], "predicate": [], "label": [], "NER": []}
for i in non_comparative:
    spl = i.split()
    data_frame_non_comparative['content'].append(i)
    data_frame_non_comparative['comparative'].append(0)
    data_frame_non_comparative['subject'].append([])
    data_frame_non_comparative['object'].append([])
    data_frame_non_comparative['aspect'].append([])
    data_frame_non_comparative['predicate'].append([])
    data_frame_non_comparative['label'].append([])
    data_frame_non_comparative['NER'].append([[0 for j in range(len(spl))]])
df_noncomparative = pd.DataFrame(data=data_frame_non_comparative)

''' Get the original dataset '''
df_btc_data = pd.DataFrame(columns=["content", "comparative", "subject", "object", "aspect", "predicate", "label", "NER"])
only_files_btc = [f for f in os.listdir("../../data/public and train data") if
              os.path.isfile(os.path.join("../../data/public and train data", f))]
for i in only_files_btc:
  df_btc_data=df_btc_data._append(data_utils.preprocessing_with_BIO('../../data/public and train data/'+i))

toc = ["DIF", "EQL", "SUP+", "SUP-", "SUP", "COM+", "COM-", "COM"]
num = [0, 93, 443, 145, 6, 10, 733, 168, 42]
for i in range(len(toc)):
    new_data_sampling_B.generate_data(toc[i],num[i+1],1500)

''' Double check '''
only_files = [f for f in os.listdir("Generate") if
              os.path.isfile(os.path.join("Generate", f))]
df_read_txt = pd.DataFrame(columns=["content", "comparative", "subject", "object", "aspect", "predicate", "label", "NER"])
for i in only_files:
    if i == 'generate.txt':
        continue
    df_read_txt = df_read_txt._append(data_utils.preprocessing_with_BIO('Generate/' + i))

''' Sum up data'''
df_final = df_read_txt._append(df_gen_total)._append(df_btc_data)._append(df_noncomparative)
df_final.to_csv('Upsample_data.csv')