import torch
import torch.nn as nn
import torch.nn.functional as F
from ..utilities import *
from torch_geometric.nn.conv import MessagePassing
import torch_geometric.nn as gnn


### Spectral Component

"""
This class represents the spectral global convolution of the Sp2GNO Block.
The Graph Fourier Transform Operator (U) is already defined (full eigendecomposition/partial)
Input is bN x d node matrix (bN total nodes with d dimensional feature embeddings)
bn represents --> b different N-node graphs (Might want to relax the requirement of same node size)
The transform operator acts on the input function node values
The convolution weight in spectral domain is an d x d x m 3D tensor with each dxd matrix representing a graph spectral "frequency" --> these parameters are trainable
"""
class GraphFourierLayer(nn.Module):

    def __init__(self, in_channels, out_channels, width, num_freq, device):

        super(GraphFourierLayer, self).__init__()

        # These represent the convolution tensor dimensions, both equal d in reality but can represent different input and output function dimensions for the block
        self.in_channels = in_channels
        self.out_channels = out_channels

        # This represents the d dimension of the input function
        self.width = width

        self.device = device # Cuda/cpu
        self.no_low_freq = num_freq # the first low frequency spectral components included in the U transform

        # Skip convolution to spectral block
        self.w = nn.Linear(self.width, self.width)

        self.scale = (1 / (in_channels * out_channels))

        self.mlp = MLP(self.width, self.width, self.width)

        self.weights = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.no_low_freq, dtype=torch.float)) # dxdxm 3d tensor for convolution
        self.norm2 = nn.LayerNorm(self.width) # Layer normalization after the spectral convolution


        """Commented out layers that include a dropout and mlp layers after the spectral convolution"""
        # self.norm1 = nn.LayerNorm(self.width)
        # self.mlp = MLP(self.width, self.width, self.width)
        # self.mlp_dropout = MLPDropout(self.width, self.width, self.width)
        # self.mlp_dropout1 = MLPDropout1(self.width, self.width, self.width)
        # self.ft_dropout = nn.Dropout(0.1)

    def compl_mul2d(self, input, weights):
        """For computing the 3D tensor convolution multiplication in spectral domain"""

        out = torch.einsum("bmi,iom->bmo", input, weights) # b, m, d for node and d x d x m for 3D kernel --> output is b, m, d
        return out
    
    def graph_fourier_transform(self, x, U, b):
        """U^T acting on x ==> x is (bN, d) while U should be b x N x m, where N is the number of nodes, m is the fourier spectrum, and b is batch size
        Same graph is assumed for each element in the batch"""

        batch_size = b
        x = x.view(batch_size, -1, self.in_channels) # b,N,d
        x_wt = torch.stack([torch.mm(U[b].t(), x[b]) for b in range(batch_size)]) # Output has dimension b, m, d
        return x_wt
    
    def inverse_graph_fourier_transform(self, x_wt, U):
        """x_wt should be b,m,d"""

        batch_size = x_wt.shape[0]
        x_wt = torch.stack([torch.mm(U[b], x_wt[b]) for b in range(batch_size)]) # Size is b x N x d
        x_wt = x_wt.view(-1, self.in_channels) # Output is bN,d
        return x_wt

    def forward(self, x, U, b, edge_index=None):
        """x is node data (bN, d) and U should be (bN, m) (after indexing in next line)"""

        U = U[:, :self.no_low_freq] # just incase the decomposition included more frequencies
        U = U.view(b, -1, self.no_low_freq) # bN,m to bxNxm
        x_ft = self.graph_fourier_transform(x, U, b)
        out_ft = torch.zeros_like(x_ft) # Size is b x m (num_low_freq) x d
        out_ft = self.compl_mul2d(x_ft, self.weights)
        x1 = self.inverse_graph_fourier_transform(out_ft, U)
        U = U.view(-1, self.no_low_freq) # back to bN x d

        """Commented out layers"""
        # x1 = self.ft_dropout(x1)
        # x1 = self.norm1(x1)
        # x1 = self.mlp_dropout(x1)
        # x1 = F.relu(x1)
        # x_out = x1 + self.w(x)

        x_out = x1 + self.w(x) # combining the skip connection
        x_out = F.gelu(x_out) # Nonlinearity
        x_out = self.norm2(x_out) # Layer norm

        return x_out
    

### Spatial Convolution

"""
This class defines a gated local spatial convolution for graph data, ARMA Conv can also be potentiall used with gnn module for torch geometric
This model extracts local features with its message passaging architecture
"""

class FrigateConv(MessagePassing):
    def __init__(this, in_channels, out_channels, lip_nodes):
        super(FrigateConv, this).__init__(aggr='add')

        # Weight Matrix used in convolution
        this.lin = nn.Linear(in_channels, out_channels)

        # Not used
        this.lin_r = nn.Linear(in_channels, out_channels)
        this.lin_rout = nn.Linear(out_channels, out_channels)

        # Edge weight initial embedding
        this.lin_ew = nn.Linear(1, 16)

        # Gating Mechanism for Edge Weight (give 0-1 importance score for edge connections)
        # Input is weight embedding and 2 * number of nodes for lipschitz embedding
        this.gate = nn.Sequential(
                nn.Linear(16 + 2*lip_nodes, 3),
                nn.ReLU(),
                nn.Linear(3, 1),
                nn.Sigmoid(),
                )

    def forward(this, x, edge_index, edge_weight, lipschitz_embeddings):
        """x is bN, d, edge_index is 2, bE, lip_embeds are (bN, lip_nodes), and edge_weight is bE, b represents batch_size"""
        """bN --> b N-node graphs, each graph must have E edges --> FUTURE IMPROVEMENT IS TO RELAX THIS"""
        """Values of edge_index are adjusted for multiple graphs --> add N --> acts like b disconnected graphs"""

        # Defines the spatial convolution layer acting on the input data (bN, d) --> edge index is 2, bN 
        # STILL A LITTLE CONFUSED HOW BATCHING WORKS FOR THIS
        x = this.lin(x)
        out = this.propagate(edge_index, x=x, edge_weight=edge_weight, lipschitz_embeddings=lipschitz_embeddings) # Propagation of message
        #out += this.lin_rout(x_r)
        out = F.normalize(out, p=2., dim=-1)
        return out
    
    def message(this, x_j, edge_index_i, edge_index_j, edge_weight, lipschitz_embeddings):

        edge_weight_j = edge_weight.view(-1, 1) # Edge weights
        edge_weight_j = this.lin_ew(edge_weight_j) # Linear embedding of edge weights
        gating_input = torch.cat((edge_weight_j, lipschitz_embeddings[edge_index_i],
            lipschitz_embeddings[edge_index_j]), dim=1) # Include lipschitz embeddings

        gating = this.gate(gating_input) # gate the edges --> remove float when dataloader
        output = x_j * gating
        # output = x_j * edge_weight_j
        return output

""" The forward operation """
class BaseSp2GNO(nn.Module):
    def __init__(self, num_fourier_layers, num_input_func, width, num_freq, device, output_dim=1, lip_nodes = 16, domain_dim = 2):
        super(BaseSp2GNO, self).__init__()

        """
        Input : 3-channel tensor, Initial input and location (a(x,y), x,y)
              : shape: (batchsize * x=width * x=height * c=3)
        Output: Intermediate Function Output at x,y
              : shape: (batchsize * x=width * x=height * c=width)
        
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
        
        self.num_fourier_layers = num_fourier_layers # How many sp2GNO block layers
        self.width = width # intermediate feature dimension after lifting projection MLP
        self.device = device # cuda/cpu
        self.domain_dim = domain_dim
        

        # self.p = MLP(num_input_func, self.width, self.width)
        self.p = nn.Linear(num_input_func, self.width)
        # Projection of node info to width dimension
    
        # Spectral Fourier layers
        self.graph_fourier_layers = nn.ModuleList()
        self.no_low_freq = num_freq
        self.lip_nodes = lip_nodes # define number of lipschitz nodes --> O((log(N))**2)
        for _ in range(num_fourier_layers):
            self.graph_fourier_layers.append(GraphFourierLayer(self.width, self.width, self.width,
                                                               self.no_low_freq, self.device))

        self.spatial_convs = nn.ModuleList()
        for i in range(num_fourier_layers):
            # self.spatial_convs.append(gnn.ARMAConv(self.width, self.width, num_stacks=1, num_layers=1, shared_weights=False)) # ARMA spatial conv
            self.spatial_convs.append(FrigateConv(self.width, self.width, self.lip_nodes))

        # Combines spatial and spectral info
        self.projection = nn.Linear(2*self.width, self.width)

    def forward(self, x, U, edge_index, edge_weight, lip_embed, device):  

        """
        x --> (b, s, s, 3), a = a(x,y), x, y
        U --> (s*s, max_mode)
        edge_index --> (2, num_edges)
        edge_weight --> (num_edges)
        lip_embed --> (s*s, num_lip_nodes)
        """


        s = x.shape[1]
        num_nodes = s*s
        num_edges = edge_index.shape[1]
        b = x.shape[0]

        edge_index = edge_index.repeat(1, b).to(device) # (2, bE)
        edge_weight = edge_weight.repeat(b).to(device) # (bE)
        lip_embed = lip_embed.repeat(b, 1).to(device) # (bN, lip_nodes)
        U = U.repeat(b, 1).to(device) # (bN, max_modes)

        for i in range(b):
            if i == b-1:
                edge_index[:, num_edges*i:] += i*num_nodes
            else:
                edge_index[:, num_edges*i:num_edges*(i+1)] += i*num_nodes

        x = x.view(b*num_nodes, -1)
        x = self.p(x.to(torch.float32))

        U = U.to(torch.float32)
        edge_weight = edge_weight.to(torch.float32)
        lip_embed = lip_embed.to(torch.float32)

        # sp2gno blocks
        for i, graph_fourier_layer in enumerate(self.graph_fourier_layers):

            x_fourier = graph_fourier_layer(x, U, b, edge_index)
            x_spatial = self.spatial_convs[i](x, edge_index, edge_weight, lip_embed)
            x_combined = torch.cat([x_fourier, x_spatial], dim=1)
            #x = self.projection(x_combined) + x
            x = self.projection(x_combined)

        return x.view(b, s, s, -1) # batch, x, y, channel
    

""" The forward operation """
class BaseSp2GNOSpectral(nn.Module):
    def __init__(self, num_fourier_layers, num_input_func, width, num_freq, device, output_dim=1, lip_nodes = 16, domain_dim = 2):
        super(BaseSp2GNOSpectral, self).__init__()

        """
        Input : 3-channel tensor, Initial input and location (a(x,y), x,y)
              : shape: (batchsize * x=width * x=height * c=3)
        Output: Intermediate Function Output at x,y
              : shape: (batchsize * x=width * x=height * c=width)
        
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
        
        self.num_fourier_layers = num_fourier_layers # How many sp2GNO block layers
        self.width = width # intermediate feature dimension after lifting projection MLP
        self.device = device # cuda/cpu
        self.domain_dim = domain_dim
        

        #self.p = MLP(num_input_func, self.width, self.width)
        self.p = nn.Linear(num_input_func, self.width)
        # Projection of node info to width dimension
    
        # Spectral Fourier layers
        self.graph_fourier_layers = nn.ModuleList()
        self.no_low_freq = num_freq
        self.lip_nodes = lip_nodes # define number of lipschitz nodes --> O((log(N))**2)
        for _ in range(num_fourier_layers):
            self.graph_fourier_layers.append(GraphFourierLayer(self.width, self.width, self.width,
                                                               self.no_low_freq, self.device))

    def forward(self, x, U, edge_index, edge_weight, lip_embed, device):  

        """
        x --> (b, s, s, 3), a = a(x,y), x, y
        U --> (s*s, max_mode)
        edge_index --> (2, num_edges)
        edge_weight --> (num_edges)
        lip_embed --> (s*s, num_lip_nodes)
        """


        s = x.shape[1]
        num_nodes = s*s
        num_edges = edge_index.shape[1]
        b = x.shape[0]

        edge_index = edge_index.repeat(1, b).to(device) # (2, bE)
        edge_weight = edge_weight.repeat(b).to(device) # (bE)
        lip_embed = lip_embed.repeat(b, 1).to(device) # (bN, lip_nodes)
        U = U.repeat(b, 1).to(device) # (bN, max_modes)

        for i in range(b):
            if i == b-1:
                edge_index[:, num_edges*i:] += i*num_nodes
            else:
                edge_index[:, num_edges*i:num_edges*(i+1)] += i*num_nodes

        x = x.view(b*num_nodes, -1)
        x = self.p(x.to(torch.float32))

        U = U.to(torch.float32)
        edge_weight = edge_weight.to(torch.float32)
        lip_embed = lip_embed.to(torch.float32)

        # sp2gno blocks
        for i, graph_fourier_layer in enumerate(self.graph_fourier_layers):

            x = graph_fourier_layer(x, U, b, edge_index)

        return x.view(b, s, s, -1) # batch, x, y, channel


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

    def forward(self, x, U, edge_index, edge_weight, lip_embed, device): 
        x = self.base_model(x, U, edge_index, edge_weight, lip_embed, device)
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

    def forward(self, xs, Us, edge_indices, edge_weights, lip_embeds, device):  
        
        inter_xs = []
        for i, x in enumerate(xs):
          inter_xs.append(self.base_model(x, Us[i], edge_indices[i], edge_weights[i], lip_embeds[i], device))

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

    def forward(self, xs, Us, edge_indices, edge_weights, lip_embeds, device):  
        
        inter_xs = []
        for i, x in enumerate(xs):
          inter_xs.append(self.base_model(x, Us[i], edge_indices[i], edge_weights[i], lip_embeds[i], device))

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

    def forward(self, x, U, edge_index, edge_weight, lip_embed, device):  
        x = self.base_model(x, U, edge_index, edge_weight, lip_embed, device)
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

    def forward(self, x, U, edge_index, edge_weight, lip_embed, device):  
        x = self.base_model(x, U, edge_index, edge_weight, lip_embed, device)
        x = self.mlp(x)
        return x
    

class DownstreamTaskFrozenBaseNewHeadCollab(nn.Module):
    def __init__(self, base_model, width):
        super(DownstreamTaskFrozenBaseNewHeadCollab, self).__init__()

        """
        The DownstreamTask2 has 1 operator that takes Frozen base model with l layers that spits an intermediate function value
        then processes it with an old head
        """

        self.base_model = base_model
        for param in self.base_model.parameters():
            param.requires_grad = False
        
        self.width = width

        self.base_model.projection = nn.Linear(2*self.width, self.width)
        for param in self.base_model.projection.parameters():
            param.requires_grad = True


        self.mlp = MLP(self.width, 1, 128)


    def forward(self, x, U, edge_index, edge_weight, lip_embed, device):  
        x = self.base_model(x, U, edge_index, edge_weight, lip_embed, device)
        x = self.mlp(x)
        return x

class DownstreamTaskFrozenBaseOldHeadCollab(nn.Module):
    def __init__(self, base_model, width, mlp_head):
        super(DownstreamTaskFrozenBaseOldHeadCollab, self).__init__()

        """
        The DownstreamTask2 has 1 operator that takes Frozen base model with l layers that spits an intermediate function value
        then processes it with an old head
        """

        self.base_model = base_model
        for param in self.base_model.parameters():
            param.requires_grad = False

        for param in self.base_model.projection.parameters():
            param.requires_grad = True

        self.width = width

        self.mlp = mlp_head

    def forward(self, x, U, edge_index, edge_weight, lip_embed, device):  
        x = self.base_model(x, U, edge_index, edge_weight, lip_embed, device)
        x = self.mlp(x)
        return x


