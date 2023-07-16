import random
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


'''
Researcher: Lei Wang
This is the data loading and processing using pytorch for UCF101 and HMDB51 dataset

Some explanations:
desired_frames = 64
The input for the deep learning pipeline is RGB videos and optical flow videos
The desired frames for both videos are all 64
We resize the video to 256x256
 For training, we random crop to get a square video 224x224, also horizontal flip
 For testing, we do center crop of the video only

'''
class Rescale(object):
    """Rescale the image in a sample to a given size.
    # Rescale here may be not necessary as each frame has been resized 
    # when loading each eimage 

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size=(256, 320)):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        
        video_rgb, video_opt, video_features, video_label = sample['video_rgb'], sample['video_opt'], sample['BOW'], sample['video_label']
    
        h_rgb, w_rgb = video_rgb.shape[1],video_rgb.shape[2]
        desired_frames = video_rgb.shape[0]
        h_opt, w_opt = video_opt.shape[1],video_opt.shape[2]
        
        if isinstance(self.output_size, int):
            # for rgb video
            if h_rgb > w_rgb:
                new_h_rgb, new_w_rgb = self.output_size * h_rgb / w_rgb, self.output_size
            else:
                new_h_rgb, new_w_rgb = self.output_size, self.output_size * w_rgb / h_rgb
            # for optic flow video
            if h_opt > w_opt:
                new_h_opt, new_w_opt = self.output_size * h_opt / w_opt, self.output_size
            else:
                new_h_opt, new_w_opt = self.output_size, self.output_size * w_opt / h_opt
        else:
            new_h_rgb, new_w_rgb = self.output_size
            new_h_opt, new_w_opt = self.output_size

        new_h_rgb, new_w_rgb = int(new_h_rgb), int(new_w_rgb)
        new_video_rgb=np.zeros((desired_frames,new_h_rgb,new_w_rgb,3))
        new_h_opt, new_w_opt = int(new_h_opt), int(new_w_opt)
        new_video_opt=np.zeros((desired_frames,new_h_opt,new_w_opt,2))
        for i in range(desired_frames):
            # rgb videos
            image=video_rgb[i,:,:,:]
            img = transform.resize(image, (new_h_rgb, new_w_rgb), mode='constant')
            new_video_rgb[i,:,:,:]=img
            # optical flow
            image=video_opt[i,:,:,:]
            img = transform.resize(image, (new_h_opt, new_w_opt), mode='constant')
            new_video_opt[i,:,:,:]=img
            
        return {'video_rgb': new_video_rgb, 'video_opt': new_video_opt, 'BOW': video_features, 'video_label': video_label}

class RandomCrop(object):
    """Crop randomly the image in a sample.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, output_size=(224,224)):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        # desired_frames
        video_rgb, video_opt, video_features, video_label = sample['video_rgb'], sample['video_opt'], sample['BOW'], sample['video_label']
        # RGB and optical flow all with the same spatial scale
        h, w = video_rgb.shape[1],video_rgb.shape[2]
        new_h, new_w = self.output_size
        desired_frames = video_rgb.shape[0]
        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)
         
        new_video_rgb=np.zeros((desired_frames,new_h,new_w,3))
        new_video_opt=np.zeros((desired_frames,new_h,new_w,2))
        for i in range(desired_frames):
            image=video_rgb[i,:,:,:]
            image = image[top: top + new_h,left: left + new_w]
            new_video_rgb[i,:,:,:]=image

            image=video_opt[i,:,:,:]
            image = image[top: top + new_h,left: left + new_w]
            new_video_opt[i,:,:,:]=image

        return {'video_rgb': new_video_rgb, 'video_opt': new_video_opt, 'BOW': video_features, 'video_label': video_label}

class CenterCrop(object):
    """Crop the given video at the center
    This function is used for testing

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, output_size=(224,224)):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        # 
        video_rgb, video_opt, video_features, video_label = sample['video_rgb'], sample['video_opt'], sample['BOW'], sample['video_label']
        # RGB and optical flow all with the same spatial scale
        h, w = video_rgb.shape[1],video_rgb.shape[2]
        new_h, new_w = self.output_size
        desired_frames = video_rgb.shape[0]
        top = int(np.round((h - new_h) / 2.))
        left = int(np.round((w - new_w) / 2.))

        new_video_rgb=np.zeros((desired_frames,new_h,new_w,3))
        new_video_opt=np.zeros((desired_frames,new_h,new_w,2))
        for i in range(desired_frames):
            image=video_rgb[i,:,:,:]
            image = image[top: top + new_h,left: left + new_w]
            new_video_rgb[i,:,:,:]=image

            image=video_opt[i,:,:,:]
            image = image[top: top + new_h,left: left + new_w]
            new_video_opt[i,:,:,:]=image

        return {'video_rgb': new_video_rgb, 'video_opt': new_video_opt, 'BOW': video_features, 'video_label': video_label}


class ToTensor(object):
    def __call__(self, sample):
        video_rgb, video_opt, video_bow, video_label = sample['video_rgb'], sample['video_opt'], sample['BOW'], sample['video_label']
        video_rgb = video_rgb.transpose((3, 0, 1, 2))
        video_opt = video_opt.transpose((3, 0, 1, 2))
        video_rgb = np.array(video_rgb)
        video_opt = np.array(video_opt)
        # video_label = [video_label]
        video_bow = video_bow.astype(np.float32)
        return {'video_rgb': torch.from_numpy(video_rgb), 'video_opt': torch.from_numpy(video_opt), 'BOW': torch.from_numpy(video_bow), 'video_label': torch.from_numpy(np.array(video_label))}


class ActionDataset(data.Dataset):
    def __init__(self, info_list, rgb_dir, opt_dir, mode='train', transforms = None):
        self.info_list = pd.read_csv(info_list, header = None)
        self.rgb_dir = rgb_dir
        self.opt_dir = opt_dir
        self.mode = mode
        self.transforms = transforms

    def __len__(self):
        return len(self.info_list)


    def __getitem__(self, idx):
        rgb_video_path = os.path.join(self.rgb_dir, self.info_list.iloc[idx, 0])
        video_features = self.info_list.iloc[idx, 1:1001]
        rgb_start = self.info_list.iloc[idx, 1002]
        opt_start = self.info_list.iloc[idx, 1003]
        sub_N = self.info_list.iloc[idx, 1004]
        stride = self.info_list.iloc[idx, 1005]
        sample = self.info_list.iloc[idx, 1006]
        video_features = video_features.values
        # the class label should be (0, C-1), where C is the total number of classes
        video_label = self.info_list.iloc[idx, 1001] - 1
        # print(rgb_video_path, 'index: ', idx)
        opt_x_video_path = os.path.join(self.opt_dir, 'u', self.info_list.iloc[idx, 0])
        opt_y_video_path = os.path.join(self.opt_dir, 'v', self.info_list.iloc[idx, 0])
        video_rgb = self.get_rgb_video(rgb_video_path, rgb_start, sub_N, stride, sample)
        video_opt = self.get_opt_video(opt_x_video_path, opt_y_video_path, opt_start, sub_N, stride, sample)
        # if it is in train mode, do horizontal flip for the video
        # otherwise (in test mode), do not flip
        # if self.mode == 'train':
        #     # horizontal flip
        #     if random.randint(1, 10) % 2 == 0:
        #         video_rgb = self.left_right_flip(video_rgb)
        #         video_opt = self.left_right_flip(video_opt)
 
        sample = {'video_rgb': video_rgb, 'video_opt': video_opt, 'BOW': video_features, 'video_label': video_label}

        if self.transforms:
            sample = self.transforms(sample)
        return sample

    def get_rgb_video(self, rgb_video_path, rgb_start, sub_N, stride, sample):
        # define the number of frames needed
        desired_frames = sub_N
        # print(sub_N)
        start_point = int(rgb_start)
        # the following parameters are fixed for experiments
        height = 256
        width = 320
        
        # define an empty array
        video_rgb = np.zeros((desired_frames, height, width, 3))
        joined_path = os.path.join(rgb_video_path, 'frame*.jpg')
        image_path = sorted(glob.glob(joined_path))
        # print(image_path) 
        actual_frames = len(image_path)
        # print(actual_frames, start_point)
        if start_point == -1:
            if actual_frames >= desired_frames:
                # print(1)
                evensList = list(range(0, desired_frames, 1))
            else:
                # print(2)
                added_num = desired_frames - actual_frames
                oldList = list(range(0, actual_frames, 1))
                addedArray = np.random.choice(oldList, added_num)
                addedList = addedArray.tolist()
                evensList = oldList + addedList
                evensList.sort()
        else:
            # print(3)
            evensList = list(range(start_point, start_point + desired_frames*sample, sample))
        # print(evensList)
        image_path = sorted(glob.glob(joined_path))
        for index in range(desired_frames):
                # get the index from the number list
                im_idx = evensList[index]
                # get the frame path from the index
                frame_idx = image_path[im_idx]
                tmp_image = io.imread(frame_idx)
                # scale pixel values in range(-1, 1)
                tmp_image = (tmp_image/255.)*2 - 1
                # resize each frame image into 256 x 256
                tmp_image = resize(tmp_image, (height, width), mode='constant')
                
                video_rgb[index, :, :, :] = tmp_image    
        
        return video_rgb

        
    def get_opt_video(self, opt_x_video_path, opt_y_video_path, opt_start, sub_N, stride, sample):
        # define the number of frames needed
        desired_frames = sub_N
        start_point = int(opt_start)
        # the following parameters are fixed for experiments
        height = 256
        width = 320
        
        # define an empty array
        video_opt = np.zeros((desired_frames, height, width, 2))
        joined_path_x = os.path.join(opt_x_video_path, 'frame*.jpg')
        joined_path_y = os.path.join(opt_y_video_path, 'frame*.jpg')
        image_x_path = sorted(glob.glob(joined_path_x))
        image_y_path = sorted(glob.glob(joined_path_y))
        # print('--------', image_x_path)        
        actual_frames = len(image_x_path)
        # print('--------', actual_frames, start_point)
        if start_point == -1:
            if actual_frames >= desired_frames:
                evensList = list(range(0, desired_frames, 1))
            else:

                added_num = desired_frames - actual_frames
                oldList = list(range(0, actual_frames, 1))
                addedArray = np.random.choice(oldList, added_num)
                addedList = addedArray.tolist()
                evensList = oldList + addedList
                evensList.sort()
        else:
            evensList = list(range(start_point, start_point + desired_frames*sample, sample))
                
        # print(evensList)
        image_x_path = sorted(glob.glob(joined_path_x))
        image_y_path = sorted(glob.glob(joined_path_y))
        for index in range(desired_frames):
                # get the index from the number list
                im_idx = evensList[index]
                # combine x and y components for optic flow
                one_frame = np.zeros((height, width, 2))
                # get the frame path from the index
                frame_x_idx = image_x_path[im_idx]
                frame_y_idx = image_y_path[im_idx]
                tmp_x_image = io.imread(frame_x_idx)
                tmp_y_image = io.imread(frame_y_idx)
                
                # scale the pixel values in range (-1, 1)
                tmp_x_image = (tmp_x_image/255.)*2 - 1
                tmp_y_image = (tmp_y_image/255.)*2 - 1
                # resize each frame image into 256 x 256
                tmp_x_image = resize(tmp_x_image, (height, width), mode='constant')
                tmp_y_image = resize(tmp_y_image, (height, width), mode='constant')

                one_frame[:, :, 0] = tmp_x_image
                one_frame[:, :, 1] = tmp_y_image
                video_opt[index, :, :, :] = one_frame
        
        return video_opt
   
    def left_right_flip(self, video):
        # horizontal flip of the video
        # RGB video left-right flip for each channel
        # Optical flow video left-right flip for each channel, but
        # also reverse the x components
        frames = video.shape[0]
        height, width = video.shape[1], video.shape[2]
        channels = video.shape[3]
        video_flipped = np.zeros((frames, height, width, channels))
        for fi in range(frames):
            # print(fi)
            channel_im = np.zeros((height, width, channels))
            for ci in range(channels):
                flip_c = video[fi, :, :, ci]
                temp_flip_c = np.flip(flip_c, 1)
                if channels == 2 and ci == 0:
                    temp_flip_c = -1 * temp_flip_c
                channel_im[:, :, ci] = temp_flip_c
            video_flipped[fi, :, :, :] = channel_im
        return(video_flipped) 
