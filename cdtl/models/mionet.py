import torch
import torch.nn as nn
import torch.nn.functional as F
from ..utilities import *

class BaseMIONet(nn.Module):
    def __init__(self, 
                 coord_dim=2, 
                 branch_dim=1,
                 hidden_dim=128, 
                 output_dim=1, 
                 num_trunk_layers=4, 
                 num_branch_layers=4):
        super().__init__()

        self.trunk_net = self.build_mlp(coord_dim, hidden_dim, hidden_dim, num_trunk_layers-1)
        self.branch_net_a = self.build_mlp(branch_dim, hidden_dim, hidden_dim, num_branch_layers-1)
        self.branch_net_e = self.build_mlp(branch_dim, hidden_dim, hidden_dim, num_branch_layers-1) 
        self.branch_net_k = self.build_mlp(branch_dim, hidden_dim, hidden_dim, num_branch_layers-1)
        self.branch_net_d = self.build_mlp(branch_dim, hidden_dim, hidden_dim, num_branch_layers-1)  

    def build_mlp(self, in_dim, hidden_dim, out_dim, num_layers):
        layers = []
        dims = [in_dim] + [hidden_dim] * (num_layers - 1) + [out_dim]
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(nn.ReLU())
        return nn.Sequential(*layers)

    def forward(self, trunk_input, branch_input_a, branch_input_e, branch_input_k, branch_input_d):
 
        N = trunk_input.shape[1]
        trunk_feat = F.relu(self.trunk_net(trunk_input.to(torch.float32)))                                            # [B, N, H]
        branch_feat_a = F.relu(self.branch_net_a(branch_input_a.to(torch.float32)))
        branch_feat_e = F.relu(self.branch_net_e(branch_input_e.to(torch.float32)))
        branch_feat_k = F.relu(self.branch_net_k(branch_input_k.to(torch.float32)))
        branch_feat_d = F.relu(self.branch_net_d(branch_input_d.to(torch.float32)))
        return trunk_feat, branch_feat_a, branch_feat_e, branch_feat_k, branch_feat_d

class SingleDownstreamTask(nn.Module):
    def __init__(self, base_model, width):
        super(SingleDownstreamTask, self).__init__()

        """
        The DownstreamTask has 1 operator that takes a base model with l layers that spits an intermediate function value
        then processes it with an MLP head
    
        """

        self.base_model = base_model
        self.width = width
  
        self.trunk_layer = nn.Linear(self.width, self.width)
        self.branch_layer_a = nn.Linear(self.width, self.width)
        self.branch_layer_e = nn.Linear(self.width, self.width)
        self.branch_layer_k = nn.Linear(self.width, self.width)
        self.branch_layer_d = nn.Linear(self.width, self.width)

    def forward(self, trunk_input, branch_input_a, branch_input_e, branch_input_k, branch_input_d): 
        trunk_feat, branch_feat_a, branch_feat_e, branch_feat_k, branch_feat_d = self.base_model(trunk_input.to(torch.float), branch_input_a.to(torch.float), branch_input_e.to(torch.float), branch_input_k.to(torch.float).to(torch.float), branch_input_d.to(torch.float))
        
        trunk_feat = self.trunk_layer(trunk_feat)
        branch_feat_a = self.branch_layer_a(branch_feat_a)
        branch_feat_e = self.branch_layer_e(branch_feat_e)
        branch_feat_k = self.branch_layer_k(branch_feat_k)
        branch_feat_d = self.branch_layer_d(branch_feat_d)

        branch_results = branch_feat_a * branch_feat_e * branch_feat_k * branch_feat_d


        output = torch.einsum('bi,bpi->bp', branch_results, trunk_feat)
        return output.unsqueeze(-1) # final shape: (B, P, O)
    
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

        self.trunk_layers = nn.ModuleList()
        for _ in range(num_of_tasks):
            self.trunk_layers.append(nn.Linear(self.width, self.width))

        self.a_layers = nn.ModuleList()
        for _ in range(num_of_tasks):
            self.a_layers.append(nn.Linear(self.width, self.width))

        self.e_layers = nn.ModuleList()
        for _ in range(num_of_tasks):
            self.e_layers.append(nn.Linear(self.width, self.width))

        self.k_layers = nn.ModuleList()
        for _ in range(num_of_tasks):
            self.k_layers.append(nn.Linear(self.width, self.width))

        self.d_layers = nn.ModuleList()
        for _ in range(num_of_tasks):
            self.d_layers.append(nn.Linear(self.width, self.width))

    def forward(self, trunk_inputs, branch_input_as, branch_input_es, branch_input_ks, branch_input_ds):  
        
        inter_trunks = []
        inter_as = []
        inter_es = []
        inter_ks = []
        inter_ds = []
        for i, trunk_input in enumerate(trunk_inputs):
            trunk_feat, branch_feat_a, branch_feat_e, branch_feat_k, branch_feat_d = self.base_model(trunk_input.to(torch.float), branch_input_as[i].to(torch.float), 
                                                                                                    branch_input_es[i].to(torch.float), branch_input_ks[i].to(torch.float).to(torch.float), 
                                                                                                    branch_input_ds[i].to(torch.float))
            inter_trunks.append(trunk_feat)
            inter_as.append(branch_feat_a)
            inter_es.append(branch_feat_e)
            inter_ks.append(branch_feat_k)
            inter_ds.append(branch_feat_d)

        outs = []
        for i, tr in enumerate(inter_trunks):
            branch_res = self.a_layers[i](inter_as[i]) * self.e_layers[i](inter_es[i]) * self.k_layers[i](inter_ks[i]) * self.d_layers[i](inter_ds[i])
            trunk_res = self.trunk_layers[i](tr)  
            outs.append(torch.einsum('bi,bpi->bp', branch_res, trunk_res).unsqueeze(-1))

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

        self.trunk_layer = nn.Linear(self.width, self.width)
        self.branch_layer_a = nn.Linear(self.width, self.width)
        self.branch_layer_e = nn.Linear(self.width, self.width)
        self.branch_layer_k = nn.Linear(self.width, self.width)
        self.branch_layer_d = nn.Linear(self.width, self.width)

    def forward(self, trunk_inputs, branch_input_as, branch_input_es, branch_input_ks, branch_input_ds):  
        
        inter_trunks = []
        inter_as = []
        inter_es = []
        inter_ks = []
        inter_ds = []
        for i, trunk_input in enumerate(trunk_inputs):
            trunk_feat, branch_feat_a, branch_feat_e, branch_feat_k, branch_feat_d = self.base_model(trunk_input.to(torch.float), branch_input_as[i].to(torch.float), 
                                                                                                    branch_input_es[i].to(torch.float), branch_input_ks[i].to(torch.float).to(torch.float), 
                                                                                                    branch_input_ds[i].to(torch.float))
            inter_trunks.append(trunk_feat)
            inter_as.append(branch_feat_a)
            inter_es.append(branch_feat_e)
            inter_ks.append(branch_feat_k)
            inter_ds.append(branch_feat_d)

        outs = []
        for i, tr in enumerate(inter_trunks):
            branch_res = self.branch_layer_a(inter_as[i]) * self.branch_layer_e(inter_es[i]) * self.branch_layer_k(inter_ks[i]) * self.branch_layer_d(inter_ds[i])
            trunk_res = self.trunk_layer(tr)  
            outs.append(torch.einsum('bi,bpi->bp', branch_res, trunk_res).unsqueeze(-1))

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

        self.trunk_layer = nn.Linear(self.width, self.width)
        self.branch_layer_a = nn.Linear(self.width, self.width)
        self.branch_layer_e = nn.Linear(self.width, self.width)
        self.branch_layer_k = nn.Linear(self.width, self.width)
        self.branch_layer_d = nn.Linear(self.width, self.width)

    def forward(self, trunk_input, branch_input_a, branch_input_e, branch_input_k, branch_input_d):  
        trunk_feat, branch_feat_a, branch_feat_e, branch_feat_k, branch_feat_d = self.base_model(trunk_input.to(torch.float), branch_input_a.to(torch.float), branch_input_e.to(torch.float), branch_input_k.to(torch.float).to(torch.float), branch_input_d.to(torch.float))
        
        trunk_feat = self.trunk_layer(trunk_feat)
        branch_feat_a = self.branch_layer_a(branch_feat_a)
        branch_feat_e = self.branch_layer_e(branch_feat_e)
        branch_feat_k = self.branch_layer_k(branch_feat_k)
        branch_feat_d = self.branch_layer_d(branch_feat_d)

        branch_results = branch_feat_a * branch_feat_e * branch_feat_k * branch_feat_d

        output = torch.einsum('bi,bpi->bp', branch_results, trunk_feat)
        return output.unsqueeze(-1) # final shape: (B, P, O)
    
class DownstreamTaskFrozenBaseOldHead(nn.Module):
    def __init__(self, base_model, width, trunk_head, a_head, e_head, k_head, d_head):
        super(DownstreamTaskFrozenBaseOldHead, self).__init__()

        """
        The DownstreamTask2 has 1 operator that takes Frozen base model with l layers that spits an intermediate function value
        then processes it with an old head
        """

        self.base_model = base_model
        for param in self.base_model.parameters():
            param.requires_grad = False

        self.width = width

        self.trunk_layer = trunk_head
        self.branch_layer_a = a_head
        self.branch_layer_e = e_head
        self.branch_layer_k = k_head
        self.branch_layer_d = d_head

    def forward(self, trunk_input, branch_input_a, branch_input_e, branch_input_k, branch_input_d):  
        trunk_feat, branch_feat_a, branch_feat_e, branch_feat_k, branch_feat_d = self.base_model(trunk_input.to(torch.float), branch_input_a.to(torch.float), branch_input_e.to(torch.float), branch_input_k.to(torch.float).to(torch.float), branch_input_d.to(torch.float))
        
        trunk_feat = self.trunk_layer(trunk_feat)
        branch_feat_a = self.branch_layer_a(branch_feat_a)
        branch_feat_e = self.branch_layer_e(branch_feat_e)
        branch_feat_k = self.branch_layer_k(branch_feat_k)
        branch_feat_d = self.branch_layer_d(branch_feat_d)

        branch_results = branch_feat_a * branch_feat_e * branch_feat_k * branch_feat_d

        output = torch.einsum('bi,bpi->bp', branch_results, trunk_feat)
        return output.unsqueeze(-1) # final shape: (B, P, O)
