import torch
from torch.utils import data
import pandas as pd
import numpy as np

class FeatureLoader(data.Dataset):
    def __init__(self, info_list):
        self.info_list = pd.read_csv(info_list, header = None)

    def __len__(self):
        return len(self.info_list)


    def __getitem__(self, idx):
        features = self.info_list.iloc[idx, 0:51]
        label = self.info_list.iloc[idx, 51]
        features_temp = features.astype(np.float32)
        features = torch.from_numpy(features_temp.values)
        label = torch.from_numpy(np.array(label))
        sample = {
            'features': features.view(1, -1), 
            'label': label
            }
        # print(sample)


        return sample
        
