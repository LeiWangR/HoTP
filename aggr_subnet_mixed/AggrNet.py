import torch
import torch.nn as nn
import torch.nn.functional as F

# sub-network for feature aggregation

class AggrNet(nn.Module):
    def __init__(self):
        super(AggrNet, self).__init__()
        self.fc = nn.Linear(51, 51)

    def forward(self, x):
        x = F.relu(x)
        x = self.fc(x)
        x = x.view(-1, 51)

        return x


