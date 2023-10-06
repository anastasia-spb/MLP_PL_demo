import torch.nn as nn
import torch.nn.functional as F


class MPLNet(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(MPLNet, self).__init__()
        self.fc1 = nn.Linear(input_dim, 250)
        self.fc2 = nn.Linear(250, 100)
        self.fc3 = nn.Linear(100, output_dim)

    def forward(self, x):
        # flatten image input
        batch_size = x.shape[0]
        x = x.view(batch_size, -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
