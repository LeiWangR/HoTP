import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from torch.autograd import Variable

import numpy as np
import math
import pandas as pd

from AggrNet import AggrNet
from weights_init import weights_init
from test import test
from train import train
from feature_loader import FeatureLoader
import os
# os.environ["CUDA_VISIABLE_DEVICES"]='0, 1, 2'

batch_train = 32
batch_test = 32

pooling_choice = 'max'
# tr_info_list = '/OSM/CBR/D61_XTLAN/work/wan305/iccv_feature/1st_pred/hmdb_s1_tr_pred_48_max_feature_label.csv'
# te_info_list = '/OSM/CBR/D61_XTLAN/work/wan305/iccv_feature/1st_pred/hmdb_s1_te_pred_48_max_feature_label.csv'

tr_info_list = '/OSM/CBR/D61_XTLAN/work/wan305/iccv_feature/1st_pred/hmdb_s1_tr_mixed_aggr_max_feature_label.csv'
te_info_list = '/OSM/CBR/D61_XTLAN/work/wan305/iccv_feature/1st_pred/hmdb_s1_te_mixed_aggr_max_feature_label.csv'

learning_rate = 1e-3
n_epochs = 200
model_path = '/OSM/CBR/D61_XTLAN/work/wan305/iccv_feature/1st_pred/hmdb_s1_48_' + pooling_choice + '_model.pt'

tr = FeatureLoader(tr_info_list)
te = FeatureLoader(te_info_list)

tr_dataloader = DataLoader(tr, batch_size = batch_train, shuffle = True, drop_last = False)
te_dataloader = DataLoader(te, batch_size = batch_test, shuffle = False, drop_last = False)

net = AggrNet()
# print(net)
if os.path.exists(model_path):
    os.remove(model_path)

for ii in range(n_epochs):
    print('Training Epoch: ', ii)
    train(batch_size = batch_train, dataloader = tr_dataloader, learning_rate = learning_rate, model_path = model_path, net = net)

    print('Testing Epoch: ', ii)

    net.load_state_dict(torch.load(model_path))
    test(batch_size = batch_test, dataloader = te_dataloader, net = net)

