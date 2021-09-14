import json
import os, sys
import pandas as pd
from sklearn.model_selection import KFold
import numpy as np
import random


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)


def get_dataset(dataset_type):
    # /dih4/dih4_2/data-share/jhu/topic_change/
    if dataset_type == 'energa':
        data_path = '/dih4/dih4_2/data-share/jhu/topic_change/energa-punctuation-deepsense-masked-slavic-bert-punctuation-deepsense-functional_acts_163_dialogs_utterances_fixed.json'
    elif dataset_type == 'logisfera':
        data_path = '/dih4/dih4_2/data-share/jhu/topic_change/logisfera-masked-slavic-bert-punctuation-deepsense-functional_acts_167_dialogs_utterances.json'
    elif dataset_type == 'telco':
        data_path = '/dih4/dih4_2/data-share/jhu/topic_change/telco_play_conllu-punctuation-deepsense-functional_acts_fixed_missing_letters_utterances.json'

    return json.load(open(data_path, 'r'))


def obtain_utt_topic_tokenwise(label_segmentation_scheme, utt_text, utt_topic, only_boundary_label):

    if label_segmentation_scheme == 'Exact':
        utt_topics_tokenwise = [utt_topic for _ in range(len(utt_text))]

    elif label_segmentation_scheme == 'I-':
        utt_topics_tokenwise = ['I-' for _ in range(len(utt_text))]
        if only_boundary_label:
            utt_topics_tokenwise[-1] = 'Boundary'
        else:
            utt_topics_tokenwise[-1] = utt_topic

    elif label_segmentation_scheme == 'IE-':
        raise ValueError(f'not implemented for label_segmentation_scheme {label_segmentation_scheme} yet')
    else:
        raise ValueError(f'invalid label_segmentation_scheme {label_segmentation_scheme} ')

    return utt_topics_tokenwise


 


out_data_dir = sys.argv[1]
#split = sys.argv[2]
n_splits = int(sys.argv[2])
dataset_type = sys.argv[3] # energa logisfera telco 
label_segmentation_scheme = sys.argv[4] # I-, Exact, IE-
add_speaker_segmentation = sys.argv[5]
only_boundary_label = sys.argv[6]
smooth_boundaries = sys.argv[7] # 


add_speaker_segmentation = (lambda x:x.lower()=='true')(add_speaker_segmentation)
print(add_speaker_segmentation)
only_boundary_label = (lambda x:x.lower()=='true')(only_boundary_label)
print(only_boundary_label)
smooth_boundaries = (lambda x:x.lower()=='true')(smooth_boundaries)


if only_boundary_label:
    if label_segmentation_scheme != 'I-':
        raise ValueError(f'when doing only segmentation, you can not use original class labels so only "I-" segmenattion scheme is valid')

out_data_dir = os.path.abspath(out_data_dir)
out_transcripts_dir = out_data_dir + '/transcripts_withlabels/'
os.makedirs(out_transcripts_dir, exist_ok=True)
 
random_seed = 11
seed_everything(random_seed)


if ',' in dataset_type:
    dataset_type = dataset_type.split(',')
    data = []
    for temp in dataset_type:
        data += get_dataset(temp)
else:
    data = get_dataset(dataset_type)



random.shuffle(data) # just to avoid any bias in the data order
print(f'number of calls in the data are {len(data)}')
folds = KFold(n_splits=n_splits)

folds_ind_list = []
for train_ind, test_ind in folds.split(range(len(data))):
    print(test_ind)
    folds_ind_list.append(test_ind)


doc_id_list = []
tsv_data = []
utt2csvpath_data = []

for call in data:
    ## obtain word to topic id
    doc_text = []
    doc_labels = []
    previous_topic = None    
    temp_doc_text = []
    temp_doc_labels = []

    for turn in call['turns']:
        for utt in turn['utterances']:
            utt_text = utt['text'].split()
            current_topic = utt['topic']
            utt_topics_tokenwise = obtain_utt_topic_tokenwise(label_segmentation_scheme, utt_text, current_topic)
            
            temp_doc_text += utt_text
            temp_doc_labels += utt_topics_tokenwise

            if (previous_topic is not None) and (current_topic != previous_topic):
                doc_text += temp_doc_text
                doc_labels += temp_doc_text
                temp_doc_text = []
                temp_doc_labels = []
                previous_topic = current_topic

        if add_speaker_segmentation:
            temp_doc_text += ['[SEP]']
            temp_doc_labels += ['DoNotExist']


    assert len(doc_text) == len(doc_labels)

    df = pd.DataFrame(columns=['original_word', 'word', 'label'])
    df = pd.DataFrame(columns=['original_word', 'word', 'label', 'label2'])
    df['original_word'] = doc_text
    df['word'] = doc_text
    df['label'] = doc_labels
    df['label2'] = doc_labels

    doc_labels_path = out_transcripts_dir + '/' + call['dialogue_id'] + '.txt'
    doc_id_list.append(call['dialogue_id'])
    tsv_data.append(call['dialogue_id'] + ',' + doc_labels_path + '\n')
    utt2csvpath_data.append(call['dialogue_id'] + ',' + doc_labels_path + '\n')

    df.to_csv(doc_labels_path, index=False)

for split_ind in range(n_splits):
    split_ind_dict = {} 
    test_split_ind = split_ind
    if test_split_ind == n_splits - 1:
        dev_split_ind = 0
    else:
        dev_split_ind = test_split_ind + 1

    split_ind_dict['test'] = list(folds_ind_list[test_split_ind])
    split_ind_dict['dev'] = list(folds_ind_list[dev_split_ind])


    split_ind_dict['train'] = []

    train_ind = []
    for i in range(n_splits):
        if (i != test_split_ind) and (i != dev_split_ind):
            train_ind.append(i)
            split_ind_dict['train'] += list(folds_ind_list[i])
    print(dev_split_ind, test_split_ind)
    print(train_ind)

    for split in ['train', 'dev', 'test']:
        out_data_dir_cv = out_data_dir + '/cv_' + str(split_ind+1)
        os.makedirs(out_data_dir_cv, exist_ok=True)
        f_split = open(out_data_dir_cv + '/' + split + '.tsv', 'w')
        f_utt2csvpath_split = open(out_data_dir_cv + '/' + '/utt2csvpath_' +  split, 'w')
        
        for ind in split_ind_dict[split]:
            f_split.write(tsv_data[ind])
            f_utt2csvpath_split.write(utt2csvpath_data[ind])
    
    
           


