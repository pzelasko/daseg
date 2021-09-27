import json
import os, sys
import pandas as pd
from sklearn.model_selection import KFold


out_data_dir = sys.argv[1]
#split = sys.argv[2]
n_splits = int(sys.argv[2])


out_data_dir = os.path.abspath(out_data_dir)
out_transcripts_dir = out_data_dir + '/transcripts_withlabels/'
os.makedirs(out_transcripts_dir, exist_ok=True)
 

data_path = '/dih4/dih4_2/data-share/jhu/topic_change/energa-punctuation-deepsense-masked-slavic-bert-punctuation-deepsense-functional_acts_163_dialogs_utterances_fixed.json'
data = json.load(open(data_path, 'r'))

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
    for turn in call['turns']:
        for utt in turn['utterances']:
            utt_text = utt['text'].split()
            utt_topic = [utt['topic'] for _ in range(len(utt_text))]
            doc_text += utt_text
            doc_labels += utt_topic

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
    
    
           

