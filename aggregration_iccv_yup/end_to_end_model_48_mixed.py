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

def sig_norm(x):
    sig_func = 1/(1+torch.exp(-x))
    x_normed = F.normalize(sig_func, p=1, dim=1)
    return x_normed

def fun_mean(num_clip_list, output_tensor):
    mean_tensor = torch.cuda.FloatTensor(1, 20).fill_(0)
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

class TwoStreamNet(nn.Module):
    def __init__(self):
        super(TwoStreamNet, self).__init__()
        self.i3d_rgb = InceptionI3d(400, in_channels = 3)
        self.i3d_opt = InceptionI3d(400, in_channels = 2)
        # self.dropout = nn.Dropout(0.5)
        # self.bn = nn.BatchNorm3d(1024)
        self.ln = nn.LayerNorm([1024, 12, 7, 7])
        self.avgpool = nn.AvgPool3d((2, 7, 7), stride=(1, 1, 1))
        self.logits = nn.Conv3d(1024,20, kernel_size=(1,1,1), padding=0,bias=True)

    def forward(self, x1, x2):
        x1 = F.relu(self.i3d_rgb(x1))
        x2 = F.relu(self.i3d_opt(x2))
        x = torch.cat((x1, x2), 2)
        # print(x.shape)
        x = self.ln(x)
        x = self.avgpool(x)
        # x = self.dropout(self.avgpool(x))        
        # print(x.shape)
        x = self.logits(x)
        x = x.mean(2)
        # print(x.shape)
        x = x.view(-1, 20)

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

num_epochs = 100
#----------------------------------------------

#----------------------------------------------
# dataset
# tr_info_list = '/flush1/wan305/data/TrainTestSplit/hmdb_train_split_01_peter_24_step_6_samp_1.csv'
# te_info_list = '/flush1/wan305/data/TrainTestSplit/hmdb_train_split_01_peter_24_step_6_samp_1.csv'
tr_info_list = '/flush1/wan305/data/TrainTestSplit/yup_mixed_train_split01_peter_48.csv'
te_info_list = '/flush1/wan305/data/TrainTestSplit/yup_mixed_test_split01_peter_48.csv'
rgb_dir = '/flush1/kon050/yup-new/rgb/'
opt_dir = '/flush1/kon050/yup-new/'

tr_hmdb51 = ActionDataset(tr_info_list, rgb_dir, opt_dir, mode = 'train', transforms=transforms.Compose([RandomCrop(), ToTensor()]))

te_hmdb51 = ActionDataset(te_info_list, rgb_dir, opt_dir, mode = 'test', transforms=transforms.Compose([CenterCrop(), ToTensor()]))

dataloader=DataLoader(tr_hmdb51,batch_size=N,shuffle=True, drop_last = False)
test_dataloader=DataLoader(te_hmdb51,batch_size = N_test,shuffle=False, drop_last = False)

#------------------------
num_clip_label1 = pd.read_csv('/flush1/wan305/data/TrainTestSplit/yup_static_test_split01_peter_48_num_clip_label.csv', header=None)
num_clip_label2 = pd.read_csv('/flush1/wan305/data/TrainTestSplit/yup_moving_test_split01_peter_48_num_clip_label.csv', header=None)
num_clip_label = pd.concat([num_clip_label1, num_clip_label2], axis = 0)

num_clip =num_clip_label.iloc[:, 0]
num_clip_list = num_clip.tolist()
true_label=num_clip_label.iloc[:, 1]
true_label = true_label.values

#---------------------------------------------
learning_rate = 1e-4
net = TwoStreamNet()
# net.apply(weights_init)
# net.i3d_rgb.load_state_dict(torch.load('/flush1/wan305/pretrained_I3D/rgb_imagenet.pt'))
# net.i3d_opt.load_state_dict(torch.load('/flush1/wan305/pretrained_I3D/flow_imagenet.pt'))

# original saved file with DataParallel
state_dict = torch.load('/flush3/wan305/Lei_trained_model/yup_mixed_two_stream_split1_48.pt')
# create new OrderedDict that does not contain `module.`
new_state_dict = OrderedDict()
for k, v in state_dict.items():
    name = k[7:] # remove `module.`
    new_state_dict[name] = v
# load params
net.load_state_dict(new_state_dict)

ct = 0
for child in net.i3d_rgb.children():
    ct += 1
    # print(ct)
    # print(child)
    if ct > 2 and ct < 19:
        for param in child.parameters():
            param.requires_grad = False
ct = 0
for child in net.i3d_opt.children():
    ct += 1
    # print(ct)
    # print(child)
    if ct > 2 and ct < 19:
        for param in child.parameters():
            param.requires_grad = False

net.cuda()
net = nn.DataParallel(net)

pred_criterion = nn.CrossEntropyLoss()

pred_optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad,net.parameters()), lr = learning_rate, momentum = 0.9, weight_decay = 1e-7)

# pred_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad,net.parameters()), lr = learning_rate, weight_decay = 1e-7)

#---------------------------------------------

for i in range(num_epochs):
    # Training -------------------------------
    net.train()
    pred_optimizer.zero_grad()
    for t, sample_batched in enumerate(dataloader):
        with torch.no_grad():
            
            rgb_video = Variable(sample_batched['video_rgb'], requires_grad = False).float().cuda()
            opt_video = Variable(sample_batched['video_opt'], requires_grad = False).float().cuda()
        # bow = Variable(sample_batched['BOW'])
        # add action labels (ground truth) for prediction pipeline
            label = Variable(sample_batched['video_label'], requires_grad = False).cuda()
        
        y_pred = net(rgb_video, opt_video)

        # pred_loss = pred_criterion(Variable(y_pred.float()).to(device), Variable(label.long()).to(device))
        pred_loss = pred_criterion(y_pred, label.long())
        # pred_optimizer.zero_grad()
        # pred_loss = Variable(pred_loss, requires_grad = True).to(device)
        pred_loss.backward()

        pred_optimizer.step()

        print('epoch: ', i, '', 'batch_id: ',t, 'Classifi. loss: %.5f' % pred_loss.item())

    torch.save(net.state_dict(), '/flush3/wan305/Lei_trained_model/yup_mixed_two_stream_split1_48_new.pt')
    # Testing---------------------------------
    
    net.eval()
    correct = 0
    total = 0
    
    output_tensor = torch.cuda.FloatTensor(1, 20).fill_(0)
    with torch.no_grad():    
        for t, sample_batched in enumerate(test_dataloader):
            # test the model
            rgb_video = Variable(sample_batched['video_rgb'], requires_grad = False).float().cuda()
            opt_video = Variable(sample_batched['video_opt'], requires_grad = False).float().cuda()
            # bow = Variable(sample_batched['BOW'])
            # add action labels (ground truth) for prediction pipeline
            label = Variable(sample_batched['video_label'], requires_grad = False).cuda()
        
            pred_label = net(rgb_video, opt_video)
 
            output_tensor = torch.cat((output_tensor, pred_label), 0)

            predicted_label = torch.max(pred_label, 1)[1].cuda()

            total += label.size(0)
            correct += (predicted_label == label).sum().item()

            print('epoch: ', i, '', 'batch_id: ', t, 'Classifi. acc.: %.5f' % (100*correct/total))
        output_tensor = output_tensor[1:, :]
        output_tensor_norm = sig_norm(output_tensor)
        output_mean_tensor = fun_mean(num_clip_list, output_tensor_norm)
        # torch.save(output_mean_tensor, '/flush3/wan305/label_soft_results_relu_24.pth')
        predicted_label_soft = torch.max(output_mean_tensor, 1)[1].cuda()
        # torch.save(predicted_label_soft, '/flush3/wan305/label_soft_voting_relu_48.pth')
        total_soft = len(num_clip_list)
        label_soft = torch.tensor(true_label).cuda()
        # print('true label: ', label)
        correct_soft = (predicted_label_soft == label_soft).sum().item()

        print('Overall Classifi. Acc. Soft Voting: %.5f' % (100*correct_soft/total_soft))

        output_array = np.array(output_tensor)
        b = np.zeros_like(output_array)
        b[np.arange(len(output_array)), output_array.argmax(1)] = 1
        output_tensor_norm = torch.tensor(b).cuda()
        output_mean_tensor = fun_mean(num_clip_list, output_tensor_norm)

        predicted_label_hard = torch.max(output_mean_tensor, 1)[1].cuda()
        # torch.save(predicted_label_hard, '/flush3/wan305/label_hard_voting_relu_48.pth')
        # print('pred. label: ', predicted_label)
        total_hard = len(num_clip_list)
        label_hard = torch.tensor(true_label).cuda()
        # print('true label: ', label)
        correct_hard = (predicted_label_hard == label_hard).sum().item()

        print('Overall Classifi. Acc. Hard Voting: %.5f' % (100*correct_hard/total_hard))
#---------------------------------------------

