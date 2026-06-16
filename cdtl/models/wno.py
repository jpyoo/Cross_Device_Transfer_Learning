import torch
import torch.nn as nn
import torch.nn.functional as F
from ..utilities import *


""" The forward operation """
class BaseWNO2d(nn.Module):
    def __init__(self, width, level, layers, size, wavelet, in_channel, out_channel, grid_range, omega, padding=0):
        super(BaseWNO2d, self).__init__()

        """
        The WNO network. It contains l-layers of the Wavelet integral layer.
        1. Lift the input using v(x) = self.fc0 .
        2. l-layers of the integral operators v(j+1)(x,y) = g(K.v + W.v)(x,y).
            --> W is defined by self.w; K is defined by self.conv.
        3. Project the output of last layer using self.fc1 and self.fc2.
        
        Input : 3-channel tensor, Initial input and location (a(x,y), x,y)
              : shape: (batchsize * x=width * x=height * c=3)
        Output: Solution of a later timestep (u(x,y))
              : shape: (batchsize * x=width * x=height * c=1)
              
        Input parameters:
        -----------------
        width : scalar, lifting dimension of input
        level : scalar, number of wavelet decomposition
        layers: scalar, number of wavelet kernel integral blocks
        size  : list with 2 elements (for 2D), image size
        wavelet: string, wavelet filter
        in_channel: scalar, channels in input including grid
        grid_range: list with 2 elements (for 2D), right supports of 2D domain
        padding   : scalar, size of zero padding
        """

        self.level = level
        self.width = width
        self.layers = layers
        self.size = size
        self.wavelet1 = wavelet[0]
        self.wavelet2 = wavelet[1]
        self.omega = omega
        self.in_channel = in_channel
        self.grid_range = grid_range 
        self.padding = padding
        
        self.conv = nn.ModuleList()
        self.w = nn.ModuleList()
        
        self.fc0 = nn.Linear(self.in_channel, self.width) # input channel is 3: (a(x, y), x, y)
        for _ in range( self.layers ):
            self.conv.append(WaveConv2dCwt(self.width, self.width, self.level, size=self.size,
                                            wavelet1=self.wavelet1, wavelet2=self.wavelet2, omega=self.omega) )
            self.w.append( nn.Conv2d(self.width, self.width, 1) )

    def forward(self, x):  
        x = self.fc0(x)                      # Shape: Batch * x * y * Channel
        x = x.permute(0, 3, 1, 2)            # Shape: Batch * Channel * x * y
        
        for index, (convl, wl) in enumerate( zip(self.conv, self.w) ):
            x = convl(x) + wl(x) 
            if index != self.layers - 1:     # Final layer has no activation    
                x = F.mish(x)                # Shape: Batch * Channel * x * y
        return x 
    

""" The forward operation """
class BaseWNO2dDropout(nn.Module):
    def __init__(self, width, level, layers, size, wavelet, in_channel, out_channel, grid_range, omega, prob = 0.05, padding=0):
        super(BaseWNO2dDropout, self).__init__()

        """
        The WNO network. It contains l-layers of the Wavelet integral layer.
        1. Lift the input using v(x) = self.fc0 .
        2. l-layers of the integral operators v(j+1)(x,y) = g(K.v + W.v)(x,y).
            --> W is defined by self.w; K is defined by self.conv.
        3. Project the output of last layer using self.fc1 and self.fc2.
        
        Input : 3-channel tensor, Initial input and location (a(x,y), x,y)
              : shape: (batchsize * x=width * x=height * c=3)
        Output: Solution of a later timestep (u(x,y))
              : shape: (batchsize * x=width * x=height * c=1)
              
        Input parameters:
        -----------------
        width : scalar, lifting dimension of input
        level : scalar, number of wavelet decomposition
        layers: scalar, number of wavelet kernel integral blocks
        size  : list with 2 elements (for 2D), image size
        wavelet: string, wavelet filter
        in_channel: scalar, channels in input including grid
        grid_range: list with 2 elements (for 2D), right supports of 2D domain
        padding   : scalar, size of zero padding
        """

        self.level = level
        self.width = width
        self.layers = layers
        self.size = size
        self.wavelet1 = wavelet[0]
        self.wavelet2 = wavelet[1]
        self.omega = omega
        self.in_channel = in_channel
        self.grid_range = grid_range 
        self.padding = padding
        
        self.conv = nn.ModuleList()
        self.w = nn.ModuleList()
        self.dropout = nn.Dropout(prob)
        
        self.fc0 = nn.Linear(self.in_channel, self.width) # input channel is 3: (a(x, y), x, y)
        for _ in range( self.layers ):
            self.conv.append(WaveConv2dCwt(self.width, self.width, self.level, size=self.size,
                                            wavelet1=self.wavelet1, wavelet2=self.wavelet2, omega=self.omega) )
            self.w.append( nn.Conv2d(self.width, self.width, 1) )

    def forward(self, x):  
        x = self.fc0(x)                      # Shape: Batch * x * y * Channel
        x = x.permute(0, 3, 1, 2)            # Shape: Batch * Channel * x * y
        
        for index, (convl, wl) in enumerate( zip(self.conv, self.w) ):
            x = convl(x) + wl(x) 
            if index != self.layers - 1:     # Final layer has no activation    
                x = F.mish(x)                # Shape: Batch * Channel * x * y
                x = self.dropout(x)

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
    
class SingleDownstreamTaskDropout(nn.Module):
    def __init__(self, base_model, width, prob=0.05):
        super(SingleDownstreamTaskDropout, self).__init__()

        """
        The DownstreamTask has 1 operator that takes a base model with l layers that spits an intermediate function value
        then processes it with an MLP head
    
        """

        self.base_model = base_model
        self.width = width
  
        self.mlp = MLPDropout(self.width, 1, 128, prob)

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
    
class MultiHeadMultiDownstreamTaskDropout(nn.Module):
    def __init__(self, base_model, width, num_of_tasks, prob=0.05):
        super(MultiHeadMultiDownstreamTaskDropout, self).__init__()

        """
        The DownstreamTask has multiple operators that share a base model with l layers that spits an intermediate function value
        for multiple tasks and processes them with separate heads
        """

        self.base_model = base_model
        self.width = width
        self.num_tasks = num_of_tasks

        self.mlps = nn.ModuleList()
        for _ in range(num_of_tasks):
            self.mlps.append(MLPDropout(self.width, 1, 128, prob))

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
    
class SingleHeadMultiDownstreamTaskDropout(nn.Module):
    def __init__(self, base_model, width, num_of_tasks, prob=0.05):
        super(SingleHeadMultiDownstreamTaskDropout, self).__init__()

        """
        The DownstreamTask has multiple operators that share a base model with l layers that spits an intermediate function value
        for multiple tasks and processes them with a single head
        """

        self.base_model = base_model
        self.width = width
        self.num_tasks = num_of_tasks

        self.mlp = MLPDropout(self.width, 1, 128, prob)

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
