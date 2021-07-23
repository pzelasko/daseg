import h5py
import pickle as pkl
import os, sys
from daseg.conversion import write_logits_h5


pkl_path = sys.argv[1] # ADReSSo_IS2021_expts/ADReSSo_CV_1_ExactLabelScheme_smoothSegmentation_text_model_TrainDevTest_BERT_Epochs_30_AlignmentsSpkrOnlyINVPAR_CorrectedBugs/bert-base-uncased_ADReSSo_42/results.pkl


out_h5_path = pkl_path.split('.pkl')[0] + '.h5'
results = pkl.load(open(pkl_path, 'rb'))

if 'utt_id' in results.keys():
    write_logits_h5(results, out_h5_path)
else:
    print(f'utt_id is not written to pkl files, please run the inference again for {pkl_path}')

