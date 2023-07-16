import pandas as pd
import numpy as np
import torch

# pooling_choice = 'avg'
pooling_choice = 'max'

dir_file = '/OSM/CBR/D61_XTLAN/work/wan305/iccv_feature/1st_pred/hmdb_s1_'

tr_str_name = dir_file + 'tr_mixed_aggr_' + pooling_choice + '_feature_label.csv'
te_str_name = dir_file + 'te_mixed_aggr_' + pooling_choice + '_feature_label.csv'

seq1_info_tr = dir_file + 'tr_pred_24_' + pooling_choice + '_feature_label.csv'
seq2_info_tr = dir_file + 'tr_pred_32_' + pooling_choice + '_feature_label.csv'
seq3_info_tr = dir_file + 'tr_pred_48_' + pooling_choice + '_feature_label.csv'
seq4_info_tr = dir_file + 'tr_pred_64_' + pooling_choice + '_feature_label.csv'
seq1_info_te = dir_file + 'te_pred_24_' + pooling_choice + '_feature_label.csv'
seq2_info_te = dir_file + 'te_pred_32_' + pooling_choice + '_feature_label.csv'
seq3_info_te = dir_file + 'te_pred_48_' + pooling_choice + '_feature_label.csv'
seq4_info_te = dir_file + 'te_pred_64_' + pooling_choice + '_feature_label.csv'

s1_tr = pd.read_csv(seq1_info_tr, header = None)
s2_tr = pd.read_csv(seq2_info_tr, header = None)
s3_tr = pd.read_csv(seq3_info_tr, header = None)
s4_tr = pd.read_csv(seq4_info_tr, header = None)
s1_te = pd.read_csv(seq1_info_te, header = None)
s2_te = pd.read_csv(seq2_info_te, header = None)
s3_te = pd.read_csv(seq3_info_te, header = None)
s4_te = pd.read_csv(seq4_info_te, header = None)

mixed_label_tr = s1_tr.iloc[:, 51]
# print(mixed_label_tr)
mixed_label_te = s1_te.iloc[:, 51]
# print(mixed_label_te)

s1_tr_array = s1_tr.iloc[:, 0:51].values
s2_tr_array = s2_tr.iloc[:, 0:51].values
s3_tr_array = s3_tr.iloc[:, 0:51].values
s4_tr_array = s4_tr.iloc[:, 0:51].values

s1_te_array = s1_te.iloc[:, 0:51].values
s2_te_array = s2_te.iloc[:, 0:51].values
s3_te_array = s3_te.iloc[:, 0:51].values
s4_te_array = s4_te.iloc[:, 0:51].values

mixed_tr_array = np.zeros((mixed_label_tr.shape[0], 51, 4))
mixed_te_array = np.zeros((mixed_label_te.shape[0], 51, 4))

mixed_tr_array[:, :, 0] = s1_tr_array
mixed_tr_array[:, :, 1] = s2_tr_array
mixed_tr_array[:, :, 2] = s3_tr_array
mixed_tr_array[:, :, 3] = s4_tr_array

mixed_te_array[:, :, 0] = s1_te_array
mixed_te_array[:, :, 1] = s2_te_array
mixed_te_array[:, :, 2] = s3_te_array
mixed_te_array[:, :, 3] = s4_te_array

# print(mixed_tr_array)

# print('---------------')

# print(mixed_te_array)

if pooling_choice == 'avg':
    mixed_f_array_tr = np.mean(mixed_tr_array, axis = 2)
    mixed_f_array_te = np.mean(mixed_te_array, axis = 2)
elif pooling_choice == 'max':
    mixed_f_array_tr = np.max(mixed_tr_array, axis = 2)
    mixed_f_array_te = np.max(mixed_te_array, axis = 2)
else:
    print('pooling should be either avg or max!')

mixed_df_tr = pd.DataFrame(mixed_f_array_tr)
mixed_df_te = pd.DataFrame(mixed_f_array_te)

mixed_tr = pd.concat([mixed_df_tr, mixed_label_tr], axis=1)
mixed_te = pd.concat([mixed_df_te, mixed_label_te], axis=1)

# print(mixed_tr.shape, mixed_te.shape)

mixed_tr.to_csv(tr_str_name, header = False, index = False)
mixed_te.to_csv(te_str_name, header = False, index = False)
