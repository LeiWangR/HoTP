from match_help_32 import match_help, pooling
import torch
import pandas as pd
import os
os.environ["CUDA_VISIABLE_DEVICES"]='0, 1, 2'

pooling_choice = 'max'
feature_file = '/OSM/CBR/D61_XTLAN/work/wan305/iccv_feature/1st_pred/hmdb_s1_tr_pred_32.pth'
f1 = '/flush1/wan305/data/TrainTestSplit/hmdb_train_split_01_peter_32_step_8_samp_1_num_clip_label_tr.csv'
f2 = '/flush1/wan305/data/TrainTestSplit/hmdb_train_split_01_peter_32_step_8_samp_2_num_clip_label_tr.csv'
f3 = '/flush1/wan305/data/TrainTestSplit/hmdb_train_split_01_peter_32_step_8_samp_3_num_clip_label_tr.csv'
f4 = '/flush1/wan305/data/TrainTestSplit/hmdb_train_split_01_peter_32_step_8_samp_4_num_clip_label_tr.csv'
str_name = feature_file[:-4] + '_' + pooling_choice + '_feature_label.csv'


# for labels
y = pd.read_csv(f1, header = None)
label = y.iloc[:, 1]
# print(label)
# print(label.shape) 

x, num_clip_list = match_help(feature_file, f1, f2, f3, f4)
# print(num_clip_list)
feature = pooling(num_clip_list, x, pooling_choice)

stack_num = int(feature.shape[0] / label.shape[0])

# print(stack_num)
# print(feature)

stacked_tensor = torch.cuda.FloatTensor(label.shape[0], 51, stack_num).fill_(0)
# print(stacked_tensor)
for ii in range(stack_num):
    start_p = label.shape[0] * ii
    end_p = label.shape[0] * ii + label.shape[0]
    single_f = feature[start_p:end_p, :]
    # print(single_f)
    # print(single_f.shape)
    stacked_tensor[:, :, ii] = single_f

    # print(stacked_tensor)

if pooling_choice == 'avg':
    aggr_feature = torch.mean(stacked_tensor, 2)
elif pooling_choice == 'max':
    aggr_feature, _ = torch.max(stacked_tensor, 2)
else:
    print('pooling should be either avg or max!')

# print(aggr_feature)
# print(aggr_feature.shape)
aggr_array = aggr_feature.cpu().detach().numpy()
aggr_df = pd.DataFrame(aggr_array)
# print(aggr_df.shape)

aggr_feature_label = pd.concat([aggr_df, label], axis=1)
aggr_feature_label.to_csv(str_name, header = False, index = False)
print(aggr_feature_label.shape)

