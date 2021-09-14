import os, sys
import numpy as np
import pickle as pkl
import sklearn.metrics.pairwise as sklearn_pairwise


results_pkl = sys.argv[1]


results = pkl.load(open(results_pkl, 'rb'))


last_hidden_state = results['last_layer_outputs_op0']

for doc_ind in range(len(last_hidden_state)):
    import pdb; pdb.set_trace()
    #dot_products = np.dot(last_hidden_state[doc_ind][0], last_hidden_state[doc_ind][0])
    dot_products = sklearn_pairwise.cosine_similarity(last_hidden_state[doc_ind][0], last_hidden_state[doc_ind][0])
    upper_diag_elements = [dot_products[i][i+1] for i in range(dot_products.shape[0]-1)]



