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

    def forward(self, x, option):
        if option == 'rgb':
            x = self.i3d_rgb(x)
        else:
            x = self.i3d_opt(x)
        # print(x_rgb.shape)
        # print(x_opt.shape)

        return x

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
option_str = 'rgb'
sub_N = 100
stride = 50
N = 5
N_test = 5
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
tr_info_list = '/flush3/wan305/Charades_bb_extraction/charades_info_list_new_tr_peter_' + str(sub_N) + '_step_' + str(stride) + '.csv'
te_info_list = '/flush3/wan305/Charades_bb_extraction/charades_info_list_new_te_peter_' + str(sub_N) + '_step_' + str(stride) + '.csv'

rgb_dir = '/scratch1/wan305/Charades/rgb/'
opt_dir = '/scratch1/wan305/Charades/'

tr_hmdb51 = ActionDataset(tr_info_list, rgb_dir, opt_dir, option = option_str, transforms=transforms.Compose([ToTensor()]))
te_hmdb51 = ActionDataset(te_info_list, rgb_dir, opt_dir, option = option_str, transforms=transforms.Compose([ToTensor()]))

train_dataloader=DataLoader(tr_hmdb51,batch_size=N,shuffle=False, drop_last = False)
test_dataloader=DataLoader(te_hmdb51,batch_size=N,shuffle=False, drop_last = False)
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
        x_tensor = torch.cuda.FloatTensor(1, 400).fill_(0)
        # print(train_dataloader, '******')
        for t, sample_batched in enumerate(train_dataloader):
            # test the model
            
            seq_video = Variable(sample_batched['video_seq'], requires_grad = False).float().cuda()
            # bow = Variable(sample_batched['BOW'])
            # add action labels (ground truth) for prediction pipeline
            label = Variable(sample_batched['video_label'], requires_grad = False).cuda()
        
            x_seq = net(seq_video, option_str)
            # print(x_rgb)
            # print('------', x_opt)
            # x_f_tensor = torch.cat((x_f_tensor, x_f), 0)
            
            # x_p_tensor = torch.cat((x_p_tensor, x_p), 0)
            x_tensor = torch.cat((x_tensor, x_seq), 0)
            # ind = t % 15
            # if t > 0 and ind == 0:
            #     x_f_tensor = x_f_tensor[1:, :, :, :, :]
            #     filename = '/flush2/wan305/hmdb_s1_te_fusion_64_' + str(t) + '.pth'
            #     torch.save(x_f_tensor, filename)
            #     print('saved', t, x_f_tensor.shape)
            #     x_f_tensor = torch.cuda.FloatTensor(1, 1024, 16, 7, 7).fill_(0)
                                 
            print(t, '-----------------')
            if t != 0 and t % 50000 == 0:
                print('part saved ...')
                x_tensor_p = x_tensor[1:, :]
                torch.save(x_tensor_p, '/flush3/wan305/charades_subN_' + str(sub_N) + '_stride_' + str(stride) + '_tr_' + option_str + '.pth')
        # x_f_tensor = x_f_tensor[1:, :, :, :, :]
        # filename = '/flush2/wan305/hmdb_s1_te_fusion_64_' + str(t) + '.pth'
        # torch.save(x_f_tensor, filename)
        # print('saved', t, x_f_tensor.shape)
    
        # x_p_tensor = x_p_tensor[1:, :, :, :, :]
        x_tensor = x_tensor[1:, :]
        print('x_tensor: ', x_tensor.shape)
        # torch.save(x_p_tensor, '/flush2/wan305/hmdb_s1_te_pool_64.pth')
        torch.save(x_tensor, '/flush3/wan305/charades_subN_' + str(sub_N) + '_stride_' + str(stride) + '_tr_' + option_str + '.pth')

    with torch.no_grad():    
        # x_f_tensor = torch.cuda.FloatTensor(1, 1024, 16, 7, 7).fill_(0)
        # x_p_tensor = torch.cuda.FloatTensor(1, 1024, 15, 1, 1).fill_(0)
        x_tensor = torch.cuda.FloatTensor(1, 400).fill_(0)
        # print(train_dataloader, '******')
        for t, sample_batched in enumerate(test_dataloader):
            # test the model
            
            seq_video = Variable(sample_batched['video_seq'], requires_grad = False).float().cuda()
            # bow = Variable(sample_batched['BOW'])
            # add action labels (ground truth) for prediction pipeline
            label = Variable(sample_batched['video_label'], requires_grad = False).cuda()
        
            x_seq = net(seq_video, option_str)
            # print(x_rgb)
            # print('------', x_opt)
            # x_f_tensor = torch.cat((x_f_tensor, x_f), 0)
            
            # x_p_tensor = torch.cat((x_p_tensor, x_p), 0)
            x_tensor = torch.cat((x_tensor, x_seq), 0)
            # ind = t % 15
            # if t > 0 and ind == 0:
            #     x_f_tensor = x_f_tensor[1:, :, :, :, :]
            #     filename = '/flush2/wan305/hmdb_s1_te_fusion_64_' + str(t) + '.pth'
            #     torch.save(x_f_tensor, filename)
            #     print('saved', t, x_f_tensor.shape)
            #     x_f_tensor = torch.cuda.FloatTensor(1, 1024, 16, 7, 7).fill_(0)
                                 
            print('-----------------', t)
            if t != 0 and t % 50000 == 0:
                print('part saved ...')
                x_tensor_p = x_tensor[1:, :]
                torch.save(x_tensor_p, '/flush3/wan305/charades_subN_' + str(sub_N) + '_stride_' + str(stride) + '_te_' + option_str + '.pth')
        # x_f_tensor = x_f_tensor[1:, :, :, :, :]
        # filename = '/flush2/wan305/hmdb_s1_te_fusion_64_' + str(t) + '.pth'
        # torch.save(x_f_tensor, filename)
        # print('saved', t, x_f_tensor.shape)
    
        # x_p_tensor = x_p_tensor[1:, :, :, :, :]
        x_tensor = x_tensor[1:, :]
        print('x_tensor: ', x_tensor.shape)
        # torch.save(x_p_tensor, '/flush2/wan305/hmdb_s1_te_pool_64.pth')
        torch.save(x_tensor, '/flush3/wan305/charades_subN_' + str(sub_N) + '_stride_' + str(stride) + '_te_' + option_str + '.pth')
