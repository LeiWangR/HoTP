import torch
from torch.utils import data
import pandas as pd
import os
import glob
import math
import numpy as np
from skimage import io, transform, img_as_float
import numpy as np
from torchvision import transforms, utils
from skimage.transform import resize

te_info_list = '/flush1/wan305/data/TrainTestSplit/yup_static_train_split01.csv'
rgb_dir = '/flush2/kon050/yup-new/rgb/'
opt_dir = '/flush2/kon050/yup-new/'

# define sub_N. stride/step and sample for the subsequence
sub_N = 96
stride = 16
sample = 1

info_list = pd.read_csv(te_info_list, header = None)

def return_start_point(info_list, rgb_dir, opt_dir, idx):
    rgb_video_path = os.path.join(rgb_dir, info_list.iloc[idx, 0])
    # print(rgb_video_path)
    video_features = info_list.iloc[idx, 1:1001]
    video_features = video_features.values
    # the class label should be (0, C-1), where C is the total number of classes
    video_label = info_list.iloc[idx, 1001] - 1
    opt_x_video_path = os.path.join(opt_dir, 'u', info_list.iloc[idx, 0])
    opt_y_video_path = os.path.join(opt_dir, 'v', info_list.iloc[idx, 0])
    # call the functions to get the starting points
    start_rgb= get_rgb_video(rgb_video_path, sub_N, stride, sample)
    start_opt= get_opt_video(opt_x_video_path, opt_y_video_path, sub_N, stride, sample)

    if start_rgb.shape[0] > start_opt.shape[0]:
        start_opt = np.concatenate((start_opt, start_opt[-1]), axis=None)
    elif start_rgb.shape[0] < start_opt.shape[0]:
        start_rgb = np.concatenate((start_rgb, start_rgb[-1]), axis=None)
    # print('rgb', start_rgb)
    # print('opt', start_opt)
    return start_rgb, start_opt


def get_rgb_video(rgb_video_path, sub_N, stride, sample):

        image_path = glob.glob(os.path.join(rgb_video_path, '*.jpg'))
        N = len(image_path)
        # print(N)
        T = (N-1-(sub_N-1)*sample)/stride
        # print(T)

        if T >= 0:
            if (T).is_integer():
                start_point = np.zeros(int(T)+1, )
                for i in range(int(T)+1):
                    start_point[i] = i * stride
            else:
                start_point = np.zeros(math.ceil(T)+1, )
                for i in range(math.floor(T)+1):
                    start_point[i] = i * stride
                # append the last adjusted starting point
                start_point[math.ceil(T)] = N-1-(sub_N - 1) * sample
        else:
            # T < 0
            start_point = np.array([-1])
        # print('rgb', start_point)
        return start_point

def get_opt_video(opt_x_video_path, opt_y_video_path, sub_N, stride, sample):

        image_x_path = glob.glob(os.path.join(opt_x_video_path, '*.jpg'))
        image_y_path = glob.glob(os.path.join(opt_y_video_path, '*.jpg'))

        N = len(image_x_path)

        # print(N)
        T = (N-1-(sub_N-1)*sample)/stride

        if T >= 0:
            if (T).is_integer():
                start_point = np.zeros(int(T)+1, )
                for i in range(int(T)+1):
                    start_point[i] = i * stride
            else:
                start_point = np.zeros(math.ceil(T)+1, )
                for i in range(math.floor(T)+1):
                    start_point[i] = i * stride
                # append the last adjusted starting point
                start_point[math.ceil(T)] = N-1-(sub_N - 1) * sample
        else:
            # T < 0
            start_point = np.array([-1])
        # print('opt', start_point)
        return start_point


num_clip_list = []
for ii in range(info_list.shape[0]):
    rgb_start, opt_start = return_start_point(info_list, rgb_dir, opt_dir, idx=ii)

    each_num = rgb_start.shape[0]
    # print(each_num)
    num_clip_list.append(each_num)
    # print(num_clip_list)
    # print('---', len(num_clip_list))
    # print(ii, '----------------------------------------------', len(num_clip_list))
num_clip_array = np.asarray(num_clip_list)
df = pd.DataFrame(num_clip_array.reshape(len(num_clip_array), -1))

traintestlist = pd.read_csv(te_info_list, sep=",", header=None)
test_label = traintestlist.iloc[:,1001:1002]-1
df_concate = pd.concat([df, test_label], axis=1)
str_name = te_info_list[: -4] + '_peter_'+ str(sub_N) + '_step_' + str(stride) + '_samp_'+str(sample)+'_num_clip_label_tr.csv'
df_concate.to_csv(str_name, header = False, index = False)
