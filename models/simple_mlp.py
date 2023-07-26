import torch.nn as nn
import torch
import einops

class SimpleSequential(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(SimpleSequential, self).__init__()

        self.nfe = 0
        
        self.net = nn.Sequential(
            nn.Linear(in_dim, 100),
            nn.ReLU(),
            nn.Linear(100, 100),
            nn.ReLU(),
            nn.Linear(100, out_dim))

        
    def forward(self, x, args=None):
        x1 = self.net(x)
        self.nfe += 1
        return x1
    
    
class SimpleSequentialTimed(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(SimpleSequentialTimed, self).__init__()

        self.nfe = 0
        
        self.net = nn.Sequential(
            nn.Linear(in_dim + 1, 100),
            nn.ReLU(),
            nn.Linear(100, 100),
            nn.ReLU(),
            nn.Linear(100, out_dim))

        
    def forward(self, t, x, args=None):
        # print(t, t.shape)
        if len(torch.tensor(t).shape) == 0 :
            t = einops.repeat(torch.tensor([t]), '1 -> b 1', b=x.shape[0])
        else:
            t = einops.repeat(t, 'b -> b 1', b=x.shape[0])
        x1 = torch.cat((x, t.to(x)), dim=1)
        x1 = self.net(x1)
        self.nfe += 1
        return x1