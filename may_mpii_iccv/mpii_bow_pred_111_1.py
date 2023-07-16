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
from sklearn import metrics
#-------------------------------------------
import warnings
import sklearn.exceptions
warnings.filterwarnings("ignore", category=sklearn.exceptions.UndefinedMetricWarning)

# define device: cpu at this stage
device = torch.device('cuda:0')

split = '1'

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
# setup the i3d model for rgb stream
i3d_rgb = InceptionI3d(400, in_channels = 3).to(device)
# replace the number of classes
# use 400 as the Kinetics dataset has 400 classes
i3d_rgb.replace_logits(400)
# load the pretrained i3d rgb model
i3d_rgb.load_state_dict(torch.load('/flush1/wan305/pretrained_I3D/rgb_imagenet.pt'))

# setup the i3d model for opt stream
i3d_opt = InceptionI3d(400, in_channels = 2).to(device)
# replace the number of classes
# use 400 as the Kinetics dataset has 400 classes
i3d_opt.replace_logits(400)
# load the pretrained i3d opt model
i3d_opt.load_state_dict(torch.load('/flush1/wan305/pretrained_I3D/flow_imagenet.pt'))

# 
for params in i3d_rgb.parameters():
    params.requires_grad = False
for params in i3d_opt.parameters():
    params.requires_grad = False

#----------------------------------------------
# extract features from two-stream I3D AveragePooling Layer
def TwoStreamI3DAvgPool(rgb_video, opt_video, batch_size):
    # use the pretrained network to extract features
    # the dimension of the extracted features is (N, 1024, 7, 1, 1)
    i3d_rgb_features = i3d_rgb.extract_avgpool_features(rgb_video)
    i3d_opt_features = i3d_opt.extract_avgpool_features(opt_video)

    i3d_rgb_features = i3d_rgb_features.view(-1, 1024, 7)
    i3d_opt_features = i3d_opt_features.view(-1, 1024, 7)

    i3d_features = np.zeros((batch_size, 1024, 7, 2))
    i3d_features[:, :, :, 0] = i3d_rgb_features
    i3d_features[:, :, :, 1] = i3d_opt_features

    return torch.from_numpy(i3d_features)

#----------------------------------------------
# def TwoStreamI3DInception(rgb_video, opt_video):
    # return

#----------------------------------------------
# define some parameters here
# N is the batch size
N = 32
# C1 is the channels for rgb videos
C1 = 3
# C2 is the channels for opt videos
C2 = 2
# T is the number of frames (we use 64 fixed frames)
T = 64
# height and width for the input videos
H, W = 224, 224

input_size = 1024
output_size = 1000

num_epochs = 100

#----------------------------------------------
# dataset
tr_info_list = '/flush2/wan305/MPII_IJCV/tr_te_split/mpii_info_list_with_subject_for_split_tr_sp' + split + '.csv'

te_info_list = '/flush2/wan305/MPII_IJCV/tr_te_split/mpii_info_list_with_subject_for_split_te_sp' + split + '.csv'

rgb_dir = '/datasets/work/D61_XTLAN_work/wan305/MPII_COOKING/rgb/'
opt_dir = '/datasets/work/D61_XTLAN_work/wan305/MPII_COOKING/'
bb_dir = '/datasets/work/D61_XTLAN_work/wan305/MPII_COOKING/mpii_bb_premium/'

tr_f_sk = '/datasets/work/D61_XTLAN_work/wan305/MPII_bow_16000D_raw_feature/mpii_bow_sketched_4000D_tr_' + split + '.pth'
te_f_sk = '/datasets/work/D61_XTLAN_work/wan305/MPII_bow_16000D_raw_feature/mpii_bow_sketched_4000D_te_' + split + '.pth'

bow_tr = torch.load(tr_f_sk)

bow_te = torch.load(te_f_sk)

tr_hmdb51 = ActionDataset(tr_info_list, rgb_dir, opt_dir, bb_dir, bow_tr, mode = 'train', transforms=transforms.Compose([ToTensor()]))

te_hmdb51 = ActionDataset(te_info_list, rgb_dir, opt_dir, bb_dir, bow_te, mode = 'test', transforms=transforms.Compose([ToTensor()]))


tr_dataloader=DataLoader(tr_hmdb51,batch_size=N,shuffle=True, drop_last = False)

te_dataloader=DataLoader(te_hmdb51,batch_size=N,shuffle=False, drop_last = False)

#---------------------------------------------
# BOW Regression pipeline (BOWFCNet and BOWCasConvNet)
class TrajCasConvNet(nn.Module):

    def __init__(self):
        super(TrajCasConvNet, self).__init__()
        self.conv1 = nn.Conv2d(1024, 1016, (3, 2))
        self.conv2 = nn.Conv2d(1016, 1008, (3, 1))
        self.conv3 = nn.Conv2d(1008, 1000, (3, 1))

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.conv3(x)
        x = x.view(-1, 1000)
        return x

class HOGCasConvNet(nn.Module):

    def __init__(self):
        super(HOGCasConvNet, self).__init__()
        self.conv1 = nn.Conv2d(1024, 1016, (3, 2))
        self.conv2 = nn.Conv2d(1016, 1008, (3, 1))
        self.conv3 = nn.Conv2d(1008, 1000, (3, 1))

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.conv3(x)
        x = x.view(-1, 1000)
        return x

class TrajFCNet(nn.Module):

    def __init__(self):
        super(TrajFCNet, self).__init__()
        self.conv = nn.Conv2d(1024, 1024, (7, 2))
        self.fc = nn.Linear(1024, 1000)

    def forward(self, x):
        x = F.relu(self.conv(x))
        x = x.view(-1, 1024)
        x = self.fc(x)
        return x

class HOGFCNet(nn.Module):

    def __init__(self):
        super(HOGFCNet, self).__init__()
        self.conv = nn.Conv2d(1024, 1024, (7, 2))
        self.fc = nn.Linear(1024, 1000)

    def forward(self, x):
        x = F.relu(self.conv(x))
        x = x.view(-1, 1024)
        x = self.fc(x)
        return x

class HOFFCNet(nn.Module):

    def __init__(self):
        super(HOFFCNet, self).__init__()
        self.conv = nn.Conv2d(1024, 1024, (7, 2))
        self.fc = nn.Linear(1024, 1000)

    def forward(self, x):
        x = F.relu(self.conv(x))
        x = x.view(-1, 1024)
        x = self.fc(x)
        return x

class HOFCasConvNet(nn.Module):

    def __init__(self):
        super(HOFCasConvNet, self).__init__()
        self.conv1 = nn.Conv2d(1024, 1016, (3, 2))
        self.conv2 = nn.Conv2d(1016, 1008, (3, 1))
        self.conv3 = nn.Conv2d(1008, 1000, (3, 1))

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.conv3(x)
        x = x.view(-1, 1000)
        return x

class MBHFCNet(nn.Module):
    
    def __init__(self):
        super(MBHFCNet, self).__init__()
        self.conv = nn.Conv2d(1024, 1024, (7, 2))
        self.fc = nn.Linear(1024, 1000)
    
    def forward(self, x):
        x = F.relu(self.conv(x))
        x = x.view(-1, 1024)
        x = self.fc(x)
        return x

class MBHCasConvNet(nn.Module):
    
    def __init__(self):
        super(MBHCasConvNet, self).__init__()
        self.conv1 = nn.Conv2d(1024, 1016, (3, 2))
        self.conv2 = nn.Conv2d(1016, 1008, (3, 1))
        self.conv3 = nn.Conv2d(1008, 1000, (3, 1))
    
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.conv3(x)
        x = x.view(-1, 1000)
        return x

#---------------------------------------------
# Prediction pipeline

class PredFCNet_S1(nn.Module):

    def __init__(self):
        super(PredFCNet_S1, self).__init__()
        self.conv = nn.Conv2d(1024, 1024, (7, 2))
        self.fc = nn.Linear(1024, 1000)

    def forward(self, x):
        x = F.relu(self.conv(x))
        x = x.view(-1, 1024)
        x = self.fc(x)
        return x


class PredCasConvNet_S1(nn.Module):

    def __init__(self):
        super(PredCasConvNet_S1, self).__init__()
        self.conv1 = nn.Conv2d(1024, 1016, (3, 2))
        self.conv2 = nn.Conv2d(1016, 1008, (3, 1))
        self.conv3 = nn.Conv2d(1008, 1000, (3, 1))

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.conv3(x)
        x = x.view(-1, 1000)
        # x = F.relu(x)
        return x

#---------------------------------------------
#class FusionPredConvNet_S2(nn.Module):
#
#    def __init__(self):
#        super(FusionPredConvNet_S2, self).__init__()
#        self.conv1 = nn.Conv1d(1000, 1000, (1, 2))
#        # for ucf101 dataset as it has 101 actions
#        self.conv2 = nn.Conv1d(1000, 101, (1, 1))
#
#    def forward(self, x):
#        x = F.relu(self.conv1(x))
#        x = self.conv2(x)
#        x = x.view(-1, 51)
#        return x

class FusionPredFCNet_S2(nn.Module):

    def __init__(self):
        super(FusionPredFCNet_S2, self).__init__()
        self.fc = nn.Linear(5000, 64)

    def forward(self, x):
        x = x.view(-1, 5000)
        x = self.fc(x)

        return x

#---------------------------------------------
class PredNet(nn.Module):

    def __init__(self):
        super(PredNet, self).__init__()
        self.stage1 = PredFCNet_S1()
        self.bn = nn.BatchNorm1d(5000)
        self.stage2 = FusionPredFCNet_S2()

    def forward(self, x, traj, hog, hof, mbh):
        x = self.stage1(x)
        #----------Option 1---------
        # The following three lines are only for stage 2 using FusionPredConvNet_S2
        # x = x.view(-1, 1000, 1, 1)
        # bow_features = bow_features.view(-1, 1000, 1, 1)
        # x = torch.cat((x, bow_features), 3)
        #----------Option 2----------
        # This is used when stage 2 use fully connected layer
        x = torch.cat((x, traj, hog, hof, mbh), 1)
        x = self.bn(x)
        x = self.stage2(x)

        return x
        
#---------------------------------------------
regr_criterion = nn.MSELoss()
learning_rate_traj = 1e-3

trajnet = TrajFCNet().to(device)
trajnet.apply(weights_init)

# bownet.load_state_dict(torch.load('/flush1/wan305/Lei_trained_model/hmdb51_bow_model.pt'))
for params in trajnet.parameters():
    params.requires_grad = True

regr_optimizer = torch.optim.SGD(trajnet.parameters(), lr = learning_rate_traj, momentum = 0.9, weight_decay = 1e-7)

#--------------------
regr_criterion_hog = nn.MSELoss()
learning_rate_hog = 1e-3

hognet = HOGFCNet().to(device)
# hognet = HOGCasConvNet().to(device)

hognet.apply(weights_init)

# bownet.load_state_dict(torch.load('/flush1/wan305/Lei_trained_model/hmdb51_bow_model.pt'))
for params in hognet.parameters():
    params.requires_grad = True

regr_optimizer_hog= torch.optim.SGD(hognet.parameters(), lr = learning_rate_hog, momentum = 0.9, weight_decay = 1e-7)

regr_criterion_hof = nn.MSELoss()
learning_rate_hof = 1e-3

hofnet = HOFFCNet().to(device)
# hofnet = HOFCasConvNet().to(device)

hofnet.apply(weights_init)

# bownet.load_state_dict(torch.load('/flush1/wan305/Lei_trained_model/hmdb51_bow_model.pt'))
for params in hofnet.parameters():
    params.requires_grad = True

regr_optimizer_hof = torch.optim.SGD(hofnet.parameters(), lr = learning_rate_hof, momentum = 0.9, weight_decay = 1e-7)

regr_criterion_mbh = nn.MSELoss()
learning_rate_mbh = 1e-3

mbhnet = MBHFCNet().to(device)
# mbhnet = MBHCasConvNet().to(device)

mbhnet.apply(weights_init)

# bownet.load_state_dict(torch.load('/flush1/wan305/Lei_trained_model/hmdb51_bow_model.pt'))
for params in mbhnet.parameters():
    params.requires_grad = True

regr_optimizer_mbh = torch.optim.SGD(mbhnet.parameters(), lr = learning_rate_mbh, momentum = 0.9, weight_decay = 1e-7)

pred_criterion = nn.CrossEntropyLoss()
learning_rate_pred = 1e-3

prednet = PredNet().to(device)
prednet.apply(weights_init)
# prednet.load_state_dict(torch.load('/flush1/wan305/Lei_trained_model/hmdb51_pred_model.pt'))
for params in prednet.parameters():
    params.requires_grad = True

pred_optimizer = torch.optim.SGD(prednet.parameters(), lr = learning_rate_pred, momentum = 0.9, weight_decay = 1e-7)
# pred_optimizer = torch.optim.Adam(prednet.parameters(), lr = learning_rate_pred, weight_decay = 1e-5)
#---------------------------------------------

for i in range(num_epochs):
    
    # Training -------------------------------
    trajnet.train()
    hognet.train()
    hofnet.train()
    mbhnet.train()
    prednet.train()
    for t, sample_batched in enumerate(tr_dataloader):
        rgb_video = Variable(sample_batched['video_rgb']).float().to(device)
        opt_video = Variable(sample_batched['video_opt']).float().to(device)
        traj = Variable(sample_batched['traj']).float().to(device)
        hog = Variable(sample_batched['hog']).float().to(device)
        hof = Variable(sample_batched['hof']).float().to(device)
        mbh = Variable(sample_batched['mbh']).float().to(device)
        # add action labels (ground truth) for prediction pipeline
        label = Variable(sample_batched['video_label']).to(device)
        N = label.shape[0]

        i3d_features = TwoStreamI3DAvgPool(rgb_video, opt_video, N)
        i3d_features = Variable(i3d_features).float().to(device)

        traj_pred = trajnet(i3d_features).to(device)
        # temp_bow_pred = bow_pred
        regr_loss = regr_criterion(traj_pred, traj)
        hog_pred = hognet(i3d_features).to(device)
        regr_loss_hog = regr_criterion_hog(hog_pred, hog)
        hof_pred = hofnet(i3d_features).to(device)
        regr_loss_hof = regr_criterion_hof(hof_pred, hof)
        mbh_pred = mbhnet(i3d_features).to(device)
        regr_loss_mbh = regr_criterion_mbh(mbh_pred, mbh)

        regr_optimizer.zero_grad()
        regr_loss.backward(retain_graph=True)
        regr_optimizer.step()

        regr_optimizer_hog.zero_grad()
        regr_loss_hog.backward(retain_graph=True)
        regr_optimizer_hog.step()

        regr_optimizer_hof.zero_grad()
        regr_loss_hof.backward(retain_graph=True)
        regr_optimizer_hof.step()
        
        regr_optimizer_mbh.zero_grad()
        regr_loss_mbh.backward(retain_graph=True)
        regr_optimizer_mbh.step()

        # torch.save(trajnet.state_dict(), '')
        # torch.save(hognet.state_dict(), '')
        # torch.save(hofnet.state_dict(), '')
        # torch.save(mbhnet.state_dict(), '')
        # torch.save(regr_optimizer.state_dict(), '/flush1/wan305/Lei_trained_model/hmdb51_bow_optimizer.pt')
        
        traj_pred = Variable(traj_pred).float().to(device)
        hog_pred = Variable(hog_pred).float().to(device)
        hof_pred = Variable(hof_pred).float().to(device)
        mbh_pred = Variable(mbh_pred).float().to(device)

        # extract features from prediction pipeline for later fusion
        pred_label = prednet(i3d_features, traj_pred, hog_pred, hof_pred, mbh_pred).to(device)
        pred_loss = pred_criterion(pred_label, label.long())
        
        pred_optimizer.zero_grad()
        # pred_loss = Variable(pred_loss, requires_grad = True).to(device)
        pred_loss.backward()

        pred_optimizer.step()

        # torch.save(prednet.state_dict(), '')

        # torch.save(pred_optimizer.state_dict(), '/flush1/wan305/Lei_trained_model/hmdb51_pred_optimizer.pt')

        print('e: ', i, '', 'b: ',t, 'traj: %.5f' %  regr_loss.item(), 'hog: %.5f' %  regr_loss_hog.item(), 'hof: %.5f' %  regr_loss_hof.item(), 'mbh: %.5f' %  regr_loss_mbh.item(), 'pred.: %.5f' % pred_loss.item())
    

    torch.save(trajnet.state_dict(), '/flush2/wan305/may_mpii_iccv/Lei_trained_model/mpii_cooking_traj_111_s'+split+'.pt')
    torch.save(hognet.state_dict(), '/flush2/wan305/may_mpii_iccv/Lei_trained_model/mpii_cooking_hog_111_s'+split+'.pt')
    torch.save(hofnet.state_dict(), '/flush2/wan305/may_mpii_iccv/Lei_trained_model/mpii_cooking_hof_111_s'+split+'.pt')
    torch.save(mbhnet.state_dict(), '/flush2/wan305/may_mpii_iccv/Lei_trained_model/mpii_cooking_mbh_111_s'+split+'.pt')
    torch.save(prednet.state_dict(), '/flush2/wan305/may_mpii_iccv/Lei_trained_model/mpii_cooking_pred_111_s'+split+'.pt')

    # Testing --------------------------------
    trajnet.eval()
    hognet.eval()
    hofnet.eval()
    mbhnet.eval()
    prednet.eval()

    correct = 0
    total = 0
    with torch.no_grad():
        for t, sample_batched in enumerate(te_dataloader):
            rgb_video = Variable(sample_batched['video_rgb']).float().to(device)
            opt_video = Variable(sample_batched['video_opt']).float().to(device)
            traj = Variable(sample_batched['traj']).float().to(device)
            hog = Variable(sample_batched['hog']).float().to(device)
            hof = Variable(sample_batched['hof']).float().to(device)
            mbh = Variable(sample_batched['mbh']).float().to(device)
            # add action labels (ground truth) for prediction pipeline
            label = Variable(sample_batched['video_label']).to(device)
            N = label.shape[0]
            i3d_features = TwoStreamI3DAvgPool(rgb_video, opt_video, N)
            i3d_features = Variable(i3d_features).float().to(device)

            traj_pred = trajnet(i3d_features).to(device)
            hog_pred = hognet(i3d_features).to(device)
            hof_pred = hofnet(i3d_features).to(device)
            mbh_pred = mbhnet(i3d_features).to(device)
            mse1 = ((traj_pred - traj) ** 2).mean() #mse loss
            mse2 = ((hog_pred - hog) ** 2).mean() #mse loss
            mse3 = ((hof_pred - hof) ** 2).mean() #mse loss
            mse4 = ((mbh_pred - mbh) ** 2).mean() #mse loss
            # extract features from prediction pipeline for later fusion
            traj_pred = Variable(traj_pred).float().to(device)
            hog_pred = Variable(hog_pred).float().to(device)
            hof_pred = Variable(hof_pred).float().to(device)
            mbh_pred = Variable(mbh_pred).float().to(device)
            pred_label = prednet(i3d_features, traj_pred, hog_pred, hof_pred, mbh_pred).to(device)
        
            predicted_label = torch.max(pred_label, 1)[1].to(device)
            predicted_label = predicted_label.cpu().detach().numpy().tolist()
            label = label.cpu().detach().numpy().tolist()
            mean_ap = metrics.precision_score(predicted_label, label, average='macro')
            # print(label, '***')
            # print(predicted_label.shape, type(predicted_label), label.shape, type(label))
            # total += label.size(0)
            # correct += (predicted_label == label).sum().item()

            print('e: ', i, '', 'b: ',t, 'traj: %.5f' % mse1, 'hog: %.5f' % mse2, 'hof: %.5f' % mse3, 'mbh: %.5f' % mse4, 'acc.: %.5f' % (100*mean_ap))

#---------------------------------------------
