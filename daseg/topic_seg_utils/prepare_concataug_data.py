import json
import os, sys
import pandas as pd
from sklearn.model_selection import KFold
import numpy as np
import random
import string


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


def obtain_utt_topic_tokenwise(label_segmentation_scheme, utt_text, utt_topic):

    if label_segmentation_scheme == 'Exact':
        utt_topics_tokenwise = [utt_topic for _ in range(len(utt_text))]

    elif label_segmentation_scheme == 'I-':
        utt_topics_tokenwise = ['I-' for _ in range(len(utt_text))]
        utt_topics_tokenwise[-1] = utt_topic

    elif label_segmentation_scheme == 'IE-':
        raise ValueError(f'not implemented for label_segmentation_scheme {label_segmentation_scheme} yet')
    else:
        raise ValueError(f'invalid label_segmentation_scheme {label_segmentation_scheme} ')

    return utt_topics_tokenwise


def obtain_smooth_boundaries(doc_labels):    
    previous_topic_ind = None
    previous_topic = None

    for ind,topic in enumerate(doc_labels):
        if (topic != 'I-') and (topic != 'DoNotExist'):
            if topic == previous_topic:
                doc_labels[previous_topic_ind] = 'I-'

            previous_topic_ind = ind
            previous_topic = topic
    return doc_labels    


def remove_class_labels(doc_labels):
    new_doc_labels = []
    for topic in doc_labels:
        if (topic != 'I-') and (topic != 'DoNotExist'):
            new_doc_labels.append('Boundary')
        else:
            new_doc_labels.append(topic)
    return new_doc_labels


def extract_single_topic_segments(doc_path):
    ignore_labels = ['DoNotExist']
    doc = pd.read_csv(doc_path, sep=',')
    doc_as_list = doc.values.tolist()
    change_points = [0]
    prev_topic = doc_as_list[0][2]
    prev_topic_ind = 0
    print(prev_topic)
    for ind in range(1, len(doc_as_list)):
        if (not doc_as_list[ind][2] in ignore_labels) and (doc_as_list[ind][2] != prev_topic):
            change_points.append(prev_topic_ind)
            prev_topic = doc_as_list[ind][2]
        elif (not doc_as_list[ind][2] in ignore_labels) and (doc_as_list[ind][2] == prev_topic):
            prev_topic_ind = ind
    print(change_points)
    doc_segments = []
    for ind in range(1, len(change_points)):
        begin_ind = change_points[ind-1] + 1
        end_ind = change_points[ind]
        segment = doc_as_list[begin_ind:end_ind]
        ## remove begining and ending seperators, they are added when concatenating segments
        if len(segment) > 10: # remove shorter segments
            if segment[0][0] == '[SEP]':
                segment = segment[1:]
            if segment[-1][0] == '[SEP]':
                segment = segment[:-1]
                
            doc_segments.append(segment)

    #import pdb; pdb.set_trace()
    lengths = [len(i) for i in doc_segments]
    print(lengths)
    return doc_segments


def mix_doc_segments(data_doc_segments):
    original_ind = list(range(len(data_doc_segments)))
    shuffle_ind = list(range(len(data_doc_segments)))
    random.shuffle(shuffle_ind)    
    new_doc_segments = []
    for ind in zip(original_ind, shuffle_ind):
        new_doc = data_doc_segments[ind[0]] + [['[SEP]', '[SEP]', 'DoNotExist', 'DoNotExist']] + data_doc_segments[ind[1]]
        new_doc_segments.append(new_doc)
    return new_doc_segments


out_data_dir = sys.argv[1]
#split = sys.argv[2]
fold_ind = int(sys.argv[2])
data_tsv_path = sys.argv[3] # train.tsv path
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
out_transcripts_dir = out_data_dir + '/cv_' + str(fold_ind) + '/transcripts_withlabels/'
os.makedirs(out_transcripts_dir, exist_ok=True)
 
random_seed = 11
seed_everything(random_seed)


doc_id_list = []
tsv_data = []
utt2csvpath_data = []

data_doc_segments = []
with open(data_tsv_path, 'r') as f:
    for i in f.readlines():
        doc_path = i.split(',')[1].strip('\n')
        data_doc_segments += extract_single_topic_segments(doc_path)

mix_count = 5 #2
data_doc_segments_mixed = []
for _ in range(mix_count):
    data_doc_segments_mixed += mix_doc_segments(data_doc_segments)
    print(len(data_doc_segments_mixed))

## obtain word to topic id
for doc in data_doc_segments_mixed:
    df = pd.DataFrame(doc, columns=['original_word', 'word', 'label', 'label2'])

    if smooth_boundaries:
        df['label'] = obtain_smooth_boundaries(df['label'])
        df['label2'] = obtain_smooth_boundaries(df['label2'])
    
    if only_boundary_label:
        df['label'] = remove_class_labels(df['label'])
        df['label2'] = remove_class_labels(df['label2'])

        if len(set(df['label'])) > 3:
            print(set(df['label']))
            import pdb; pdb.set_trace()
        if len(set(df['label2'])) > 3:
            print(set(df['label2']))
            import pdb; pdb.set_trace()


    letters = string.ascii_lowercase
    dialogue_id =  ''.join(random.choice(letters) for i in range(10)) 
    doc_labels_path = out_transcripts_dir + '/' + dialogue_id + '.txt'
    doc_id_list.append(dialogue_id)

    tsv_data.append(dialogue_id + ',' + doc_labels_path + '\n')
    utt2csvpath_data.append(dialogue_id + ',' + doc_labels_path + '\n')
    df.to_csv(doc_labels_path, index=False)


out_data_dir_cv = out_data_dir + '/cv_' + str(fold_ind)
os.makedirs(out_data_dir_cv, exist_ok=True)
split = 'train'
f_split = open(out_data_dir_cv + '/' + split + '.tsv', 'w')
f_utt2csvpath_split = open(out_data_dir_cv + '/' + '/utt2csvpath_' +  split, 'w')
for ind in range(len(tsv_data)):
    f_split.write(tsv_data[ind])
    f_utt2csvpath_split.write(utt2csvpath_data[ind])
 

           





