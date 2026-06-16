import torch
import torch.nn as nn
import torch.nn.functional as F
from ..utilities import *


class BaseFNO2d(nn.Module):
    def __init__(self, width, layers, in_channel, out_channel, modes1, modes2):
        super(BaseFNO2d, self).__init__()

        """
        The overall network. It contains 4 layers of the Fourier layer.
        1. Lift the input to the desire channel dimension by self.fc0 .
        2. 4 layers of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv .
        3. Project from the channel space to the output space by self.fc1 and self.fc2 .
        
        input: the solution of the coefficient function and locations (a(x, y), x, y)
        input shape: (batchsize, x=s, y=s, c=3)
        output: the solution 
        output shape: (batchsize, x=s, y=s, c=1)
        """

        self.modes1 = modes1
        self.modes2 = modes2
        self.layers = layers
        self.width = width
        self.fc0 = nn.Linear(in_channel, self.width) # input channel is 3: (a(x, y), x, y)
        self.conv = nn.ModuleList()
        self.w = nn.ModuleList()

        for _ in range(self.layers):
            self.conv.append(SpectralConv2d(self.width, self.width, self.modes1, self.modes2))
            self.w.append(nn.Conv1d(self.width, self.width, 1))


    def forward(self, x):
        batchsize = x.shape[0]
        size_x, size_y = x.shape[1], x.shape[2]

        x = self.fc0(x)
        x = x.permute(0, 3, 1, 2)

        for index, (conv, w) in enumerate( zip(self.conv, self.w) ):
            x = conv(x) + w(x.view(batchsize, self.width, -1)).view(batchsize, self.width, size_x, size_y) 
            if index != self.layers - 1:     # Final layer has no activation    
                x = F.relu(x)                # Shape: Batch * Channel * x * y

        return x

class SingleDownstreamTask(nn.Module):
    def __init__(self, base_model, width):
        super(SingleDownstreamTask, self).__init__()

        """
        The DownstreamTask has 1 operator that takes a base model with l layers that spits an intermediate function value
        then processes it with an MLP head
    
        """

        self.base_model = base_model
        self.width = width
  
        self.mlp = MLP(self.width, 1, 128)

    def forward(self, x): 
        x = self.base_model(x.to(torch.float32))
        x = x.permute(0, 2, 3, 1)            
        x = self.mlp(x)
        return x
    
class MultiHeadMultiDownstreamTask(nn.Module):
    def __init__(self, base_model, width, num_of_tasks):
        super(MultiHeadMultiDownstreamTask, self).__init__()

        """
        The DownstreamTask has multiple operators that share a base model with l layers that spits an intermediate function value
        for multiple tasks and processes them with separate heads
        """

        self.base_model = base_model
        self.width = width
        self.num_tasks = num_of_tasks

        self.mlps = nn.ModuleList()
        for _ in range(num_of_tasks):
            self.mlps.append(MLP(self.width, 1, 128))

    def forward(self, xs):  
        
        inter_xs = []
        for x in xs:
          inter_xs.append(self.base_model(x.to(torch.float32)).permute(0,2,3,1))

        outs = []
        for i, x in enumerate(inter_xs):
          outs.append(self.mlps[i](x))

        return outs
    
class SingleHeadMultiDownstreamTask(nn.Module):
    def __init__(self, base_model, width, num_of_tasks):
        super(SingleHeadMultiDownstreamTask, self).__init__()

        """
        The DownstreamTask has multiple operators that share a base model with l layers that spits an intermediate function value
        for multiple tasks and processes them with a single head
        """

        self.base_model = base_model
        self.width = width
        self.num_tasks = num_of_tasks

        self.mlp = MLP(self.width, 1, 128)

    def forward(self, xs):  
        
        inter_xs = []
        for x in xs:
          inter_xs.append(self.base_model(x.to(torch.float32)).permute(0,2,3,1))

        outs = []
        for x in inter_xs:
          outs.append(self.mlp(x))

        return outs
    
class DownstreamTaskFrozenBaseNewHead(nn.Module):
    def __init__(self, base_model, width):
        super(DownstreamTaskFrozenBaseNewHead, self).__init__()

        """
        The DownstreamTask2 has 1 operator that takes Frozen base model with l layers that spits an intermediate function value
        then processes it with a newly initialized head
        """

        self.base_model = base_model
        for param in self.base_model.parameters():
            param.requires_grad = False

        self.width = width

        self.mlp = MLP(self.width, 1, 128)

    def forward(self, x):  
        x = self.base_model(x.to(torch.float32))
        x = x.permute(0, 2, 3, 1)            # Shape: Batch * x * y * Channel
        x = self.mlp(x)
        return x
    
class DownstreamTaskFrozenBaseOldHead(nn.Module):
    def __init__(self, base_model, width, mlp_head):
        super(DownstreamTaskFrozenBaseOldHead, self).__init__()

        """
        The DownstreamTask2 has 1 operator that takes Frozen base model with l layers that spits an intermediate function value
        then processes it with an old head
        """

        self.base_model = base_model
        for param in self.base_model.parameters():
            param.requires_grad = False

        self.width = width

        self.mlp = mlp_head

    def forward(self, x):  
        x = self.base_model(x.to(torch.float32))
        x = x.permute(0, 2, 3, 1)            # Shape: Batch * x * y * Channel
        x = self.mlp(x)
        return x
