import torch
import torch.nn as nn
import torch.nn.functional as F
from action_dataloader import ActionDataset, ToTensor, RandomCrop, CenterCrop
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from torch.autograd import Variable
from torch.optim import lr_scheduler
from pytorch_i3d import InceptionI3d
import numpy as np
import math
from collections import OrderedDict
import pandas as pd
import os
os.environ["CUDA_VISIABLE_DEVICES"]='0, 1, 2'

class TwoStreamNet(nn.Module):
    def __init__(self):
        super(TwoStreamNet, self).__init__()
        self.i3d_rgb = InceptionI3d(400, in_channels = 3)
        self.i3d_opt = InceptionI3d(400, in_channels = 2)

    def forward(self, x1, x2):
        x_rgb = self.i3d_rgb(x1)
        x_opt = self.i3d_opt(x2)
        # print(x_rgb.shape)
        # print(x_opt.shape)

        return x_rgb, x_opt

# define device: cpu at this stage
# device = torch.device('cuda:0')

#-------------------------------------------
# weights initialization
def weights_init(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))
        if m.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(m.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(m.bias, -bound, bound)
    elif isinstance(m, nn.Conv1d):
        nn.init.normal_(m.weight.data)
        if m.bias is not None:
            nn.init.normal_(m.bias.data)
    elif isinstance(m, nn.Conv2d):
        nn.init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            nn.init.normal_(m.bias.data)
    elif isinstance(m, nn.Conv3d):
        nn.init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            nn.init.normal_(m.bias.data)

#--------------------------------------------

#----------------------------------------------

# define some parameters her

# N is the batch size
N = 20
N_test = 20
# C1 is the channels for rgb videos
C1 = 3
# C2 is the channels for opt videos
C2 = 2
# height and width for the input videos
H, W = 224, 224

# dropout_keep_prob=0.5

num_epochs = 1
#----------------------------------------------

#----------------------------------------------
# dataset
tr_info_list = '/flush2/wan305/MPII_IJCV/mpii_info_list_with_subject_for_split_peter_80.csv'

rgb_dir = '/datasets/work/D61_XTLAN_work/wan305/MPII_COOKING/rgb/'
opt_dir = '/datasets/work/D61_XTLAN_work/wan305/MPII_COOKING/'
bb_dir = '/datasets/work/D61_XTLAN_work/wan305/MPII_COOKING/mpii_bb_premium/'

tr_hmdb51 = ActionDataset(tr_info_list, rgb_dir, opt_dir, bb_dir, mode = 'train', transforms=transforms.Compose([ToTensor()]))

train_dataloader=DataLoader(tr_hmdb51,batch_size=N,shuffle=False, drop_last = False)

#---------------------------------------------
net = TwoStreamNet()
net.i3d_rgb.load_state_dict(torch.load('/flush1/wan305/pretrained_I3D/rgb_imagenet.pt'))
net.i3d_opt.load_state_dict(torch.load('/flush1/wan305/pretrained_I3D/flow_imagenet.pt'))

net.cuda()
net = nn.DataParallel(net)
#---------------------------------------------

for i in range(num_epochs):
    # Testing---------------------------------
    
    net.eval()

    with torch.no_grad():    
        # x_f_tensor = torch.cuda.FloatTensor(1, 1024, 16, 7, 7).fill_(0)
        # x_p_tensor = torch.cuda.FloatTensor(1, 1024, 15, 1, 1).fill_(0)
        x_rgb_tensor = torch.cuda.FloatTensor(1, 400).fill_(0)
        x_opt_tensor = torch.cuda.FloatTensor(1, 400).fill_(0)
        for t, sample_batched in enumerate(train_dataloader):
            # test the model
            rgb_video = Variable(sample_batched['video_rgb'], requires_grad = False).float().cuda()
            opt_video = Variable(sample_batched['video_opt'], requires_grad = False).float().cuda()
            # bow = Variable(sample_batched['BOW'])
            # add action labels (ground truth) for prediction pipeline
            label = Variable(sample_batched['video_label'], requires_grad = False).cuda()
        
            x_rgb, x_opt = net(rgb_video, opt_video)
            # print(x_rgb)
            # print('------', x_opt)
            # x_f_tensor = torch.cat((x_f_tensor, x_f), 0)
            
            # x_p_tensor = torch.cat((x_p_tensor, x_p), 0)
            x_rgb_tensor = torch.cat((x_rgb_tensor, x_rgb), 0)
            x_opt_tensor = torch.cat((x_opt_tensor, x_opt), 0)
            # ind = t % 15
            # if t > 0 and ind == 0:
            #     x_f_tensor = x_f_tensor[1:, :, :, :, :]
            #     filename = '/flush2/wan305/hmdb_s1_te_fusion_64_' + str(t) + '.pth'
            #     torch.save(x_f_tensor, filename)
            #     print('saved', t, x_f_tensor.shape)
            #     x_f_tensor = torch.cuda.FloatTensor(1, 1024, 16, 7, 7).fill_(0)
                                 
            print(t, '-----------------')
        # x_f_tensor = x_f_tensor[1:, :, :, :, :]
        # filename = '/flush2/wan305/hmdb_s1_te_fusion_64_' + str(t) + '.pth'
        # torch.save(x_f_tensor, filename)
        # print('saved', t, x_f_tensor.shape)
    
        # x_p_tensor = x_p_tensor[1:, :, :, :, :]
        x_rgb_tensor = x_rgb_tensor[1:, :]
        x_opt_tensor = x_opt_tensor[1:, :]
        print('x_rgb_tensor: ', x_rgb_tensor.shape)
        print('x_opt_tensor: ', x_opt_tensor.shape)
        # torch.save(x_p_tensor, '/flush2/wan305/hmdb_s1_te_pool_64.pth')
        torch.save(x_rgb_tensor, '/flush2/wan305/MPII_IJCV/Raw_400D/mpii_pred_80_rgb_m.pth')
        torch.save(x_opt_tensor, '/flush2/wan305/MPII_IJCV/Raw_400D/mpii_pred_80_opt_m.pth')
