import pandas as pd

# other_label = 'static'
other_label = 'moving'

info_list1 = pd.read_csv('/flush1/wan305/data/TrainTestSplit/yup_' + other_label + '_train_split01_peter_64_step_16_samp_1.csv', header = None)
info_list2 = pd.read_csv('/flush1/wan305/data/TrainTestSplit/yup_' + other_label + '_train_split01_peter_64_step_16_samp_2.csv', header = None)
# info_list3 = pd.read_csv('/flush1/wan305/data/TrainTestSplit/yup_' + other_label + '_train_split01_peter_48_step_16_samp_3.csv', header = None)
print(info_list1.shape, info_list2.shape)
df = pd.concat([info_list1, info_list2], axis=0)
print(df.shape)
df.to_csv('/flush1/wan305/data/TrainTestSplit/yup_' + other_label + '_train_split01_peter_64.csv', index=False, header=False)

info_list1 = pd.read_csv('/flush1/wan305/data/TrainTestSplit/yup_' + other_label + '_test_split01_peter_64_step_16_samp_1.csv', header = None)
info_list2 = pd.read_csv('/flush1/wan305/data/TrainTestSplit/yup_' + other_label + '_test_split01_peter_64_step_16_samp_2.csv', header = None)
# info_list3 = pd.read_csv('/flush1/wan305/data/TrainTestSplit/yup_' + other_label + '_test_split01_peter_48_step_16_samp_3.csv', header = None)
print(info_list1.shape, info_list2.shape)
df = pd.concat([info_list1, info_list2], axis=0)
print(df.shape)
df.to_csv('/flush1/wan305/data/TrainTestSplit/yup_' + other_label + '_test_split01_peter_64.csv', index=False, header=False)

info_list1 = pd.read_csv('/flush1/wan305/data/TrainTestSplit/yup_' + other_label + '_test_split01_peter_64_step_16_samp_1_num_clip_label.csv', header = None)
info_list2 = pd.read_csv('/flush1/wan305/data/TrainTestSplit/yup_' + other_label + '_test_split01_peter_64_step_16_samp_2_num_clip_label.csv', header = None)
# info_list3 = pd.read_csv('/flush1/wan305/data/TrainTestSplit/yup_' + other_label + '_test_split01_peter_48_step_16_samp_3_num_clip_label.csv', header = None)
print(info_list1.shape, info_list2.shape)
df = pd.concat([info_list1, info_list2], axis=0)
print(df.shape)
df.to_csv('/flush1/wan305/data/TrainTestSplit/yup_' + other_label + '_test_split01_peter_64_num_clip_label.csv', index=False, header=False)
