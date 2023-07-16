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

def test(batch_size, dataloader, net):
    print('Test')
    net.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for t, sample_batched in enumerate(dataloader):
            features = Variable(sample_batched['features'], requires_grad = False).float()
            label = Variable(sample_batched['label'], requires_grad = False)
            pred_label = net(features)
            predicted_label = torch.max(pred_label, 1)[1]
            total += label.size(0)
            correct += (predicted_label == label).sum().item()
            print('batch id: ', t, 'Acc.: %.5f' %(100*correct/total))

