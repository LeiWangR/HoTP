import torch
import pandas as pd
import numpy as np
import math
import os
os.environ["CUDA_VISIABLE_DEVICES"]='0, 1, 2'
# perform feature aggregation
# two kind of feature pooling
# 1 - average pooling 'avg'
# 2 - max pooling 'max'
def pooling(num_clip_list, output_tensor, pooling_choice):
    pool_tensor = torch.cuda.FloatTensor(1, 51).fill_(0)
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
        if pooling_choice == 'avg':
            one_video_pool = torch.mean(one_video, 0)
        elif pooling_choice == 'max':
            one_video_pool, _ = torch.max(one_video, 0)
        else:
            print('pooling should be either avg or max!')
        pool_reshape = one_video_pool.view(1, -1)
        # print(pool_reshape.shape)
        pool_tensor = torch.cat((pool_tensor, pool_reshape), 0)
        # print(pool_tensor.shape)
    return pool_tensor[1:, :]


def match_help(feature_file, f1, f2, f3, f4):
    x = torch.load(feature_file)
    y1 = pd.read_csv(f1, header = None)
    y2 = pd.read_csv(f2, header = None)
    y3 = pd.read_csv(f3, header = None)
    y4 = pd.read_csv(f4, header = None)
    df = pd.concat([y1, y2, y3, y4], axis=0)
    # print(x.shape)
    # print(y1.shape, y2.shape, y3.shape)
    # print(df.shape)
    num_clip =df.iloc[:, 0]
    num_clip_list = num_clip.tolist()
    # print(len(num_clip_list))
    myarray = np.asarray(num_clip_list)
    # print(sum(myarray))   
    
    return x, num_clip_list
