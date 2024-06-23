# -*- coding: utf-8 -*-
"""
Created on Tue Aug 22 17:32:38 2023

@author: plai3
"""

import streamlit as st
import numpy as np
import pandas as pd

from keras.models import model_from_json

from Bio import SeqIO
from io import StringIO
from anarci import anarci

def one_hot_encoder(s):
    d = {'A': 0, 'C': 1, 'D': 2, 'E': 3, 'F': 4, 'G': 5, 'H': 6, 'I': 7, 'K': 8, 'L': 9, 'M': 10, 'N': 11, 'P': 12, 'Q': 13, 'R': 14, 'S': 15, 'T': 16, 'V': 17, 'W': 18, 'Y': 19, '-': 20}
    x = np.zeros((len(d), len(s)))
    x[[d[c] for c in s], range(len(s))] = 1
    return x

H_inclusion_list = [str(i) for i in range(1, 129)] + ['111A', '111B', '111C', '111D', '111E', '111F', '111G', '111H', '112I', '112H', '112G', '112F', '112E', '112D', '112C', '112B', '112A', '112'] + [str(i) for i in range(113, 129)]
L_inclusion_list = [str(i) for i in range(1, 128)]

H_dict = {str(i): i - 1 for i in range(1, 112)}
H_dict.update({
    '111A': 111, '111B': 112, '111C': 113, '111D': 114, '111E': 115, '111F': 116, '111G': 117, '111H': 118,
    '112I': 119, '112H': 120, '112G': 121, '112F': 122, '112E': 123, '112D': 124, '112C': 125, '112B': 126, '112A': 127, '112': 128
})
H_dict.update({str(i): i + 128 for i in range(113, 129)})

L_dict = {str(i): i - 1 for i in range(1, 128)}

st.set_page_config(
    page_title="DeepSP App",
    layout="centered",
)

st.title('DeepSP')
st.header('Deep learning-based antibody structural properties')
st.subheader('The FASTA file format is H_seq/L_seq (variable regions)')

st.markdown('''
### EXAMPLE:
Heavy Chain: QVQLVQSGAEVKKPGASVKVSCKASGYTFTGYYMNWVRQAPGQGLEWMGWINPNSGGTNYAQKFQGRVTMTRDTSISTAYMELSRLRSDDTAVYYCARGKNSDYNWDFQHWGQGTLVTVSS
Light Chain: DIVMSQSPSSLAVSVGEKVTMSCKSSQSLLYSSNQKNYLAWYQQKPGQSPKLLIYWASTRESGVPDRFTGSGSGTDFTLTISSVKAEDLAVYYCQQYEMFGGGTKLEIK
''')

sequence_H = st.text_area("Enter the heavy chain sequence:")
sequence_L = st.text_area("Enter the light chain sequence:")


if sequence_H and sequence_L:
    sequences_H = [(name, sequence_H)]
    sequences_L = [(name, sequence_L)]
    
    results_H = anarci(sequences_H, scheme="imgt", output=False)
    results_L = anarci(sequences_L, scheme="imgt", output=False)
    numbering_H, alignment_details_H, hit_tables_H = results_H
    numbering_L, alignment_details_L, hit_tables_L = results_L

    seq_list = []
    for i in range(len(sequences_H)):    
        if numbering_H[i] is None:
            print('ANARCI did not number', sequences_H[i][0])
        else:
            domain_numbering_H, start_index_H, end_index_H = numbering_H[i][0]
            domain_numbering_L, start_index_L, end_index_L = numbering_L[i][0]
            H_tmp = 145 * ['-']
            L_tmp = 127 * ['-']
            for j in range(len(domain_numbering_H)):
                col_H = str(domain_numbering_H[j][0][0]) + domain_numbering_H[j][0][1].replace(" ", "")
                H_tmp[H_dict[col_H]] = domain_numbering_H[j][1]
            for j in range(len(domain_numbering_L)):
                col_L = str(domain_numbering_L[j][0][0]) + domain_numbering_L[j][0][1].replace(" ", "")
                L_tmp[L_dict[col_L]] = domain_numbering_L[j][1]
            seq_list.append(''.join(H_tmp + L_tmp))
    
    X = np.transpose(np.asarray([one_hot_encoder(s) for s in seq_list]), (0, 2, 1))

    def load_and_predict(json_file_name, h5_file_name, columns):
        with open(json_file_name, 'r') as json_file:
            loaded_model_json = json_file.read()
        loaded_model = model_from_json(loaded_model_json)
        loaded_model.load_weights(h5_file_name)
        loaded_model.compile(optimizer='adam', loss='mae', metrics=['mae'])
        y_pred = loaded_model.predict(X)
        return pd.DataFrame(y_pred, columns=columns)

    df_SAPpos = load_and_predict('Conv1D_regressionSAPpos.json', 'Conv1D_regression_SAPpos.h5', ['SAP_pos_CDRH1', 'SAP_pos_CDRH2', 'SAP_pos_CDRH3', 'SAP_pos_CDRL1', 'SAP_pos_CDRL2', 'SAP_pos_CDRL3', 'SAP_pos_CDR', 'SAP_pos_Hv', 'SAP_pos_Lv', 'SAP_pos_Fv'])
    df_SCMneg = load_and_predict('Conv1D_regressionSCMneg.json', 'Conv1D_regression_SCMneg.h5', ['SCM_neg_CDRH1', 'SCM_neg_CDRH2', 'SCM_neg_CDRH3', 'SCM_neg_CDRL1', 'SCM_neg_CDRL2', 'SCM_neg_CDRL3', 'SCM_neg_CDR', 'SCM_neg_Hv', 'SCM_neg_Lv', 'SCM_neg_Fv'])
    df_SCMpos = load_and_predict('Conv1D_regressionSCMpos.json', 'Conv1D_regression_SCMpos.h5', ['SCM_pos_CDRH1', 'SCM_pos_CDRH2', 'SCM_pos_CDRH3', 'SCM_pos_CDRL1', 'SCM_pos_CDRL2', 'SCM_pos_CDRL3', 'SCM_pos_CDR', 'SCM_pos_Hv', 'SCM_pos_Lv', 'SCM_pos_Fv'])

    df_name = pd.DataFrame(name_list, columns=['ID'])
    df_DeepSP = pd.concat([df_name, df_SAPpos, df_SCMneg, df_SCMpos], axis=1)
    st.dataframe(data=df_DeepSP, use_container_width=True, hide_index=True)
