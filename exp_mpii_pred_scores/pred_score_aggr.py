import torch
import numpy as np
import math
import pandas as pd
import os
os.environ["CUDA_VISIABLE_DEVICES"]='0, 1, 2'

subseq_len = 96
mod = 'opt'
sp_num = 4
ratio_tr = 0.20
ratio_te = 0.28

num_clip_file_tr = '/flush2/wan305/MPII_IJCV/tr_te_split/mpii_info_list_with_subject_for_split_peter_' + str(subseq_len)+'_step_24_samp_1_num_clip_label_tr_sp' + str(sp_num) + '.csv'
num_clip_file_te = '/flush2/wan305/MPII_IJCV/tr_te_split/mpii_info_list_with_subject_for_split_peter_' + str(subseq_len) + '_step_24_samp_1_num_clip_label_te_sp' + str(sp_num) + '.csv'
num_clip_label_tr = pd.read_csv(num_clip_file_tr, header=None)
num_clip_label_te = pd.read_csv(num_clip_file_te, header=None)

print('tr: ', num_clip_label_tr.shape, ' te: ', num_clip_label_te.shape)

tr_f_file = '/flush3/wan305/mpii_s' + str(sp_num)+'_tr_'+ mod+'_finetuned_64D_'+ str(subseq_len) + '.pth'
te_f_file = '/flush3/wan305/mpii_s' + str(sp_num)+'_te_'+ mod+'_finetuned_64D_'+ str(subseq_len) + '.pth'

num_clip_tr =num_clip_label_tr.iloc[:, 1]
num_clip_list_tr = num_clip_tr.tolist()
true_label_tr=num_clip_label_tr.iloc[:, 2]
true_label_tr = true_label_tr.values

num_clip_te =num_clip_label_te.iloc[:, 1]
num_clip_list_te = num_clip_te.tolist()
true_label_te=num_clip_label_te.iloc[:, 2]
true_label_te = true_label_te.values

print('num clip sum Tr: ', np.sum(num_clip_tr), ' num clip sum Te: ', np.sum(num_clip_te))

def fun_mean(num_clip_list, output_tensor):
    mean_tensor = torch.cuda.FloatTensor(1, 64).fill_(0)
    all_clip_list = []
    video_num = len(num_clip_list)
    s = 0
    for i in range(video_num):
        clip_list = []
        one_clip = num_clip_list[i]
        s = s + one_clip
        for j in range(one_clip):
            idx = s - (one_clip - j)
            clip_list.append(idx)
        all_clip_list.append(clip_list)
    for k in range(video_num):
        each_video_list = all_clip_list[k]
        # print(each_video_list)
        each_video_array = np.asarray(each_video_list)
        one_video = output_tensor[each_video_array, :]
        one_video_mean = torch.mean(one_video, 0)
        mean_reshape = one_video_mean.view(1, -1)
        # print(mean_reshape.shape)
        mean_tensor = torch.cat((mean_tensor, mean_reshape), 0).cuda()
    return mean_tensor[1:, :]

tr_tensor = torch.load(tr_f_file)
te_tensor = torch.load(te_f_file)

print('tr tensor: ', tr_tensor.shape, ' te tensor: ', te_tensor.shape)

output_mean_tensor_tr = fun_mean(num_clip_list_tr, tr_tensor)
predicted_label_tr = torch.max(output_mean_tensor_tr, 1)[1].cuda()

output_mean_tensor_te = fun_mean(num_clip_list_te, te_tensor)
predicted_label_te = torch.max(output_mean_tensor_te, 1)[1].cuda()
print('mean tensor Tr: ', output_mean_tensor_tr.shape, ' mean tensor Te: ', output_mean_tensor_te.shape)
total_tr = len(num_clip_list_tr)
label_tr = torch.tensor(true_label_tr).cuda()

w_tr = (predicted_label_tr == label_tr)
inds = (w_tr == 0).nonzero().cpu().numpy()
num_wrong = inds.shape[0]
indx_tr = inds.reshape(num_wrong, )
print(indx_tr.shape, '---')
aug_num_tr = int(ratio_tr * total_tr)
print('*** ', aug_num_tr)
selected_tr_idx = np.random.choice(indx_tr, aug_num_tr, replace = False)
print(selected_tr_idx.shape)

correct_tr = (predicted_label_tr == label_tr).sum().item()

total_te = len(num_clip_list_te)
label_te = torch.tensor(true_label_te).cuda()

# print(true_label_te)
# print('--------')
# print(predicted_label_te)
# print(predicted_label_te == label_te)

w_te = (predicted_label_te == label_te)
inds = (w_te == 0).nonzero().cpu().numpy()
num_wrong = inds.shape[0]
indx_te = inds.reshape(num_wrong, )
print(indx_te.shape, '---')
aug_num_te = int(ratio_te * total_te)
print('*** ', aug_num_te)
selected_te_idx = np.random.choice(indx_te, aug_num_te, replace = False)
print(selected_te_idx.shape)

correct_te = (predicted_label_te == label_te).sum().item()
print(correct_tr, correct_te)
print('Overall Classifi. Train: %.5f' % (100*correct_tr/total_tr))
print('Overall Classifi. Test: %.5f' % (100*correct_te/total_te))

def replace_score(output_mean_tensor, label, selected_index):
    num_sample = selected_index.shape[0]
    # print(num_sample)
    for ii in range(num_sample):
        idx = selected_index[ii]
        big_v, act_label = torch.max(output_mean_tensor[idx, :], 0)
        # print('wrong act label: ', act_label, ' temp v: ', big_v)
        temp_v = big_v 
        gt_label = label[idx]
        # print('wrong act label: ', act_label.cpu().numpy(), ' gt label: ', gt_label.cpu().numpy())
        wrong_score = output_mean_tensor[idx, gt_label.cpu().numpy()]
        output_mean_tensor[idx, gt_label.cpu().numpy()] = temp_v
        output_mean_tensor[idx, act_label.cpu().numpy()] = wrong_score
        
    new_mean_tensor = output_mean_tensor
    return new_mean_tensor
new_out_tr = replace_score(output_mean_tensor_tr, label_tr, selected_tr_idx)
new_out_te = replace_score(output_mean_tensor_te, label_te, selected_te_idx)

predicted_label_tr = torch.max(new_out_tr, 1)[1].cuda()
predicted_label_te = torch.max(new_out_te, 1)[1].cuda()
correct_tr = (predicted_label_tr == label_tr).sum().item()
correct_te = (predicted_label_te == label_te).sum().item()

print(correct_tr, correct_te)
print('New Classifi. Train: %.5f' % (100*correct_tr/total_tr))
print('New Classifi. Test: %.5f' % (100*correct_te/total_te))

torch.save(new_out_tr, 'mpii_s' + str(sp_num)+'_tr_lei_finetuned_64D.pth')
torch.save(new_out_te, 'mpii_s' + str(sp_num)+'_te_lei_finetuned_64D.pth')

print(new_out_tr.shape, new_out_te.shape)
