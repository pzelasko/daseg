import os, sys
import pandas as pd
import glob


data_tsv_path_list = sys.argv[1]


if ',' in data_tsv_path_list:
    data_tsv_path_list = data_tsv_path_list.split(',')
    data_tsv_path_list = '*'.join(data_tsv_path_list)
    data_tsv_path_list = glob.glob(data_tsv_path_list)
else:
    data_tsv_path_list = [data_tsv_path_list]

class2count_punct = {}
class2count_casing = {}

for data_tsv_path in data_tsv_path_list:
    print(data_tsv_path)
    data_tsv = pd.read_csv(data_tsv_path, sep=',', header=None)


    for doc_label_path in data_tsv[1].values:
        doc_labels = pd.read_csv(doc_label_path, sep=',')
        doc_labels_labelgroups = doc_labels.groupby('label')
        #import pdb; pdb.set_trace()

        for label_group in doc_labels_labelgroups.groups.keys():
            if not label_group in class2count_casing:
                class2count_casing[label_group] = 0
            class2count_casing[label_group] += len(doc_labels_labelgroups.groups[label_group])

        doc_labels_labelgroups = doc_labels.groupby('label2')
        for label_group in doc_labels_labelgroups.groups.keys():
            if not label_group in class2count_punct:
                class2count_punct[label_group] = 0
            class2count_punct[label_group] += len(doc_labels_labelgroups.groups[label_group])

print('#####################################################################')
class2count_casing = dict(sorted(class2count_casing.items()))
print(class2count_casing)
print(f'total num of tokens are {sum(class2count_casing.values())}')


for i,j in class2count_casing.items():
    print(i,j)


print('#####################################################################')
class2count_punct = dict(sorted(class2count_punct.items()))
print(class2count_punct)
print(f'total num of tokens are {sum(class2count_punct.values())}')

num_files = len(data_tsv_path_list)
print(f'num_files are {num_files}')


print('#####################################################################')
class2count_casing = {i:int(j/num_files) for i,j in class2count_casing.items()}
print(class2count_casing)
print(f'total num of tokens per file are {sum(class2count_casing.values())}')
print(f'numbe of classes are {len(class2count_casing)}')

class2count_punct = {i:int(j/num_files) for i,j in class2count_punct.items()}
print(f'total num of tokens per file are {sum(class2count_punct.values())}')
print(f'numbe of classes are {len(class2count_casing)}')

