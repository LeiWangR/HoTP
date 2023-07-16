import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from torch.autograd import Variable

import numpy as np
import math
import pandas as pd

from torch.utils.data import Dataset, DataLoader
from AggrNet import AggrNet
from weights_init import weights_init
from feature_loader import FeatureLoader
from test import test
from torch.optim.lr_scheduler import MultiStepLR
import os.path

def train(batch_size, dataloader, learning_rate, model_path, net):
    print('Train')
    if os.path.exists(model_path):
        net.load_state_dict(torch.load(model_path))
    else:
        net.apply(weights_init)
    pred_criterion = nn.CrossEntropyLoss()
    pred_optimizer = torch.optim.SGD(net.parameters(), lr = learning_rate)
    scheduler = MultiStepLR(pred_optimizer, milestones=[20,50], gamma=0.1)
    scheduler.step()
    net.train()

    pred_optimizer.zero_grad()
    for t, sample_batched in enumerate(dataloader):
        with torch.no_grad():
            features = Variable(sample_batched['features'], requires_grad = False).float()
            label = Variable(sample_batched['label'], requires_grad = False)
            # print(features.shape)
            # print(label)
        y_pred = net(features)
        # print(y_pred)
        pred_loss = pred_criterion(y_pred, label.long())
        pred_loss.backward()
        pred_optimizer.step()
            
        print('batch id: ', t, 'loss: %.5f' %pred_loss.item())

        torch.save(net.state_dict(), model_path)

