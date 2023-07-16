import torch
import torch.nn as nn
import torch.nn.functional as F
from action_dataloader import ActionDataset, ToTensor, RandomCrop, CenterCrop
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from torch.autograd import Variable
from pytorch_i3d import InceptionI3d
import numpy as np
import math
from collections import OrderedDict
import pandas as pd
import os
os.environ["CUDA_VISIABLE_DEVICES"]='0, 1, 2'


class I3DSingleStream(nn.Module):
    def __init__(self):
        super(I3DSingleStream, self).__init__()
        self.i3d_rgb = InceptionI3d(400, in_channels = 3)
        self.dropout = nn.Dropout(0.5)
        self.logits = nn.Conv3d(1024,64, kernel_size=(1,1,1), padding=0, bias=True)

    def forward(self, x):
        x = self.i3d_rgb(x)
        x = self.dropout(x)
        x = self.logits(x)
        x = x.mean(2)
        # print(x.shape)
        x = x.view(-1, 64)
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

num_epochs = 1000
#----------------------------------------------

#----------------------------------------------
# dataset
tr_info_list = '/flush2/wan305/MPII_IJCV/tr_te_split/mpii_info_list_with_subject_for_split_peter_64_tr_sp3.csv'
te_info_list = '/flush2/wan305/MPII_IJCV/tr_te_split/mpii_info_list_with_subject_for_split_peter_64_te_sp3.csv'

rgb_dir = '/datasets/work/D61_XTLAN_work/wan305/MPII_COOKING/rgb/'
opt_dir = '/datasets/work/D61_XTLAN_work/wan305/MPII_COOKING/'
bb_dir = '/datasets/work/D61_XTLAN_work/wan305/MPII_COOKING/mpii_bb_premium/'

tr_hmdb51 = ActionDataset(tr_info_list, rgb_dir, opt_dir, bb_dir, mode = 'train', transforms=transforms.Compose([ToTensor()]))

te_hmdb51 = ActionDataset(te_info_list, rgb_dir, opt_dir, bb_dir, mode = 'test', transforms=transforms.Compose([ToTensor()]))

train_dataloader=DataLoader(tr_hmdb51,batch_size=N,shuffle=True, drop_last = False)
test_dataloader=DataLoader(te_hmdb51,batch_size = N_test,shuffle=False, drop_last = False)

#---------------------------------------------
learning_rate = 1e-5

net = I3DSingleStream()
net.apply(weights_init)
net.i3d_rgb.load_state_dict(torch.load('/flush1/wan305/pretrained_I3D/rgb_imagenet.pt'))

# # original saved file with DataParallel
# state_dict = torch.load('/flush3/wan305/Lei_trained_model/hmdb51_split1_opt.pt')
# # create new OrderedDict that does not contain `module.`
# new_state_dict = OrderedDict()
# for k, v in state_dict.items():
#     name = k[7:] # remove `module.`
#     new_state_dict[name] = v
# # load params
# net.load_state_dict(new_state_dict)

for params in net.i3d_rgb.parameters():
    params.requires_grad = False

net.cuda()
net = nn.DataParallel(net)

pred_criterion = nn.CrossEntropyLoss()

pred_optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad,net.parameters()), lr = learning_rate, momentum = 0.9, weight_decay = 1e-7)
#---------------------------------------------

for i in range(num_epochs):
    # Training -------------------------------
    net.train()
    pred_optimizer.zero_grad()
    for t, sample_batched in enumerate(train_dataloader):
        rgb_video = Variable(sample_batched['video_rgb'], requires_grad = False).float().cuda()
        # opt_video = Variable(sample_batched['video_opt'], requires_grad = False).float().cuda()
        # bow = Variable(sample_batched['BOW'])
        # add action labels (ground truth) for prediction pipeline
        label = Variable(sample_batched['video_label'], requires_grad = False).cuda()
        y_pred = net(rgb_video)

        # pred_loss = pred_criterion(Variable(y_pred.float()).to(device), Variable(label.long()).to(device))
        pred_loss = pred_criterion(y_pred, label.long())
        # pred_optimizer.zero_grad()
        # pred_loss = Variable(pred_loss, requires_grad = True).to(device)
        pred_loss.backward()

        pred_optimizer.step()

        print('epoch: ', i, '', 'batch_id: ',t, 'Classifi. loss: %.5f' % pred_loss.item())

    torch.save(net.state_dict(), '/flush3/wan305/Lei_trained_model/mpii_s3_rgb_64.pt')
    # Testing---------------------------------
    net.eval()
    correct = 0
    total = 0
    with torch.no_grad():    
        for t, sample_batched in enumerate(test_dataloader):
            # test the model
            rgb_video = Variable(sample_batched['video_rgb'], requires_grad = False).float().cuda()
            # opt_video = Variable(sample_batched['video_opt'], requires_grad = False).float().cuda()
            # bow = Variable(sample_batched['BOW'])
            # add action labels (ground truth) for prediction pipeline
            label = Variable(sample_batched['video_label'], requires_grad = False).cuda()
        
            pred_label = net(rgb_video)
            predicted_label = torch.max(pred_label, 1)[1].cuda()
            total += label.size(0)
            correct += (predicted_label == label).sum().item()

            print('epoch: ', i, '', 'batch_id: ', t, 'Classifi. acc.: %.5f' % (100*correct/total))
        
