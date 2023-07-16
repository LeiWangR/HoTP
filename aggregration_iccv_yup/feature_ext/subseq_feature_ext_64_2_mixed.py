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
        # self.dropout = nn.Dropout(0.5)
        # self.bn = nn.BatchNorm3d(1024)
        self.ln = nn.LayerNorm([1024, 16, 7, 7])
        self.avgpool = nn.AvgPool3d((2, 7, 7), stride=(1, 1, 1))
        self.logits = nn.Conv3d(1024,20, kernel_size=(1,1,1), padding=0,bias=True)
    
    def forward(self, x1, x2):
        x1 = F.relu(self.i3d_rgb(x1))
        x2 = F.relu(self.i3d_opt(x2))
        x = torch.cat((x1, x2), 2)
        # print(x.shape)
        x_f = self.ln(x)
        x_p = self.avgpool(x_f)
        # x = self.dropout(self.avgpool(x))
        # print(x.shape)
        x = self.logits(x_p)
        # print(x.shape)
        x = x.mean(2)
        # print(x.shape)
        x = x.view(-1, 20)
        
        return x_f, x_p, x

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

# define some parameters here
# N is the batch size
N = 32
N_test = 32
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
tr_info_list = '/flush1/wan305/data/TrainTestSplit/yup_mixed_train_split01_peter_64.csv'
te_info_list = '/flush1/wan305/data/TrainTestSplit/yup_mixed_test_split01_peter_64.csv'
rgb_dir = '/flush1/kon050/yup-new/rgb/'
opt_dir = '/flush1/kon050/yup-new/'

tr_hmdb51 = ActionDataset(tr_info_list, rgb_dir, opt_dir, mode = 'train', transforms=transforms.Compose([CenterCrop(), ToTensor()]))

te_hmdb51 = ActionDataset(te_info_list, rgb_dir, opt_dir, mode = 'test', transforms=transforms.Compose([CenterCrop(), ToTensor()]))

train_dataloader=DataLoader(tr_hmdb51,batch_size=N,shuffle=False, drop_last = False)
test_dataloader=DataLoader(te_hmdb51,batch_size = N_test,shuffle=False, drop_last = False)

#------------------------
# num_clip_label = pd.read_csv('/flush1/wan305/data/TrainTestSplit/hmdb_test_split_01_aggr_f64_p_0.5_num_clip_label.csv', header=None)

# num_clip =num_clip_label.iloc[:, 0]
# num_clip_list = num_clip.tolist()
# true_label=num_clip_label.iloc[:, 1]
# true_label = true_label.values
#------------------------


# dataloader=DataLoader(tr_hmdb51,batch_size=N,shuffle=True, drop_last = False, num_workers = 24, pin_memory = True)
# test_dataloader=DataLoader(te_hmdb51,batch_size = N_test,shuffle=True, drop_last = False, num_workers = 24, pin_memory = True)
#---------------------------------------------
# learning_rate = 1e-4
net = TwoStreamNet()
# net.apply(weights_init)
# net.i3d_rgb.load_state_dict(torch.load('/flush1/wan305/pretrained_I3D/rgb_imagenet.pt'))
# net.i3d_opt.load_state_dict(torch.load('/flush1/wan305/pretrained_I3D/flow_imagenet.pt'))

# original saved file with DataParallel
state_dict = torch.load('/flush3/wan305/Lei_trained_model/yup_mixed_two_stream_split1_64_new.pt')
# create new OrderedDict that does not contain `module.`
new_state_dict = OrderedDict()
for k, v in state_dict.items():
    name = k[7:] # remove `module.`
    new_state_dict[name] = v
# load params
net.load_state_dict(new_state_dict)

# ct = 0
# for child in net.i3d_rgb.children():
#     ct += 1
#     # print(ct)
#     # print(child)
#     if ct > 2 and ct < 19:
#         for param in child.parameters():
#             param.requires_grad = False
# ct = 0
# for child in net.i3d_opt.children():
#     ct += 1
#     # print(ct)
#     # print(child)
#     if ct > 2 and ct < 19:
#         for param in child.parameters():
#             param.requires_grad = False

net.cuda()
net = nn.DataParallel(net)

# pred_criterion = nn.CrossEntropyLoss()

# pred_optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad,net.parameters()), lr = learning_rate, momentum = 0.9, weight_decay = 1e-7)

# pred_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad,net.parameters()), lr = learning_rate, weight_decay = 1e-7)

#---------------------------------------------

for i in range(num_epochs):
    # Testing---------------------------------
    
    net.eval()

    with torch.no_grad():    
        # x_f_tensor = torch.cuda.FloatTensor(1, 1024, 16, 7, 7).fill_(0)
        x_p_tensor = torch.cuda.FloatTensor(1, 1024).fill_(0)
        # x_pred_tensor = torch.cuda.FloatTensor(1, 51).fill_(0)
        for t, sample_batched in enumerate(train_dataloader):
            # test the model
            rgb_video = Variable(sample_batched['video_rgb'], requires_grad = False).float().cuda()
            opt_video = Variable(sample_batched['video_opt'], requires_grad = False).float().cuda()
            # bow = Variable(sample_batched['BOW'])
            # add action labels (ground truth) for prediction pipeline
            label = Variable(sample_batched['video_label'], requires_grad = False).cuda()
        
            x_f, x_p, pred_label = net(rgb_video, opt_video)
            # x_f_tensor = torch.cat((x_f_tensor, x_f), 0)
            x_p = torch.mean(x_p, 2)
            x_p_tensor = torch.cat((x_p_tensor, x_p.view(-1, 1024)), 0)
            # x_p_tensor = torch.cat((x_p_tensor, x_p), 0)
            # x_pred_tensor = torch.cat((x_pred_tensor, pred_label), 0)
            
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
    
        x_p_tensor = x_p_tensor[1:, :]
        # x_pred_tensor = x_pred_tensor[1:, :]
        print('x_p_tensor saved', x_p_tensor.shape)
        # torch.save(x_p_tensor, '/flush2/wan305/hmdb_s1_te_pool_64.pth')
        torch.save(x_p_tensor, '/flush3/wan305/yup_mixed_s1_tr_pool_64.pth')
        
    with torch.no_grad():
        # x_f_tensor = torch.cuda.FloatTensor(1, 1024, 16, 7, 7).fill_(0)
        x_p_tensor = torch.cuda.FloatTensor(1, 1024).fill_(0)
        # x_pred_tensor = torch.cuda.FloatTensor(1, 51).fill_(0)
        for t, sample_batched in enumerate(test_dataloader):
            # test the model
            rgb_video = Variable(sample_batched['video_rgb'], requires_grad = False).float().cuda()
            opt_video = Variable(sample_batched['video_opt'], requires_grad = False).float().cuda()
            # bow = Variable(sample_batched['BOW'])
            # add action labels (ground truth) for prediction pipeline
            label = Variable(sample_batched['video_label'], requires_grad = False).cuda()

            x_f, x_p, pred_label = net(rgb_video, opt_video)
            # x_f_tensor = torch.cat((x_f_tensor, x_f), 0)
            x_p = torch.mean(x_p, 2)
            x_p_tensor = torch.cat((x_p_tensor, x_p.view(-1, 1024)), 0)
            # x_p_tensor = torch.cat((x_p_tensor, x_p), 0)
            # x_pred_tensor = torch.cat((x_pred_tensor, pred_label), 0)
            
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

        x_p_tensor = x_p_tensor[1:, :]
        # x_pred_tensor = x_pred_tensor[1:, :]
        print('x_p_tensor saved', x_p_tensor.shape)
        # torch.save(x_p_tensor, '/flush2/wan305/hmdb_s1_te_pool_64.pth')
        torch.save(x_p_tensor, '/flush3/wan305/yup_mixed_s1_te_pool_64.pth')
