import torch
import torch.nn as nn
import torch.nn.functional as F
from ..utilities import *


class BaseNOMAD(nn.Module):
    def __init__(self, 
                 coord_dim=2, 
                 branch_dim=1,
                 hidden_dim=128, 
                 output_dim=1, 
                 num_trunk_layers=3, 
                 num_branch_layers=3,
                 comb_init_layers=3):
        super().__init__()

        self.trunk_net_init = self.build_mlp(coord_dim, hidden_dim, hidden_dim, num_trunk_layers-2)
        self.branch_net_a = self.build_mlp(branch_dim, hidden_dim, hidden_dim, num_branch_layers)
        self.branch_net_e = self.build_mlp(branch_dim, hidden_dim, hidden_dim, num_branch_layers) 
        self.branch_net_k = self.build_mlp(branch_dim, hidden_dim, hidden_dim, num_branch_layers)
        self.branch_net_d = self.build_mlp(branch_dim, hidden_dim, hidden_dim, num_branch_layers)  
        self.comb_layer_init = self.build_mlp(hidden_dim*5, hidden_dim, hidden_dim, comb_init_layers-2)

    def build_mlp(self, in_dim, hidden_dim, out_dim, num_layers):
        layers = []
        dims = [in_dim] + [hidden_dim] * (num_layers - 1) + [out_dim]
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:  # No activation after final layer
                layers.append(nn.ReLU())
        return nn.Sequential(*layers)

    def forward(self, trunk_input, branch_input_a, branch_input_e, branch_input_k, branch_input_d):
 
        N = trunk_input.shape[1]
        trunk_feat_init = self.trunk_net_init(trunk_input.to(torch.float32))                                            # [B, N, H]
        branch_feat_a = self.branch_net_a(branch_input_a.to(torch.float32)).unsqueeze(1).repeat(1,N,1)       # [B, N, H]
        branch_feat_e = self.branch_net_e(branch_input_e.to(torch.float32)).unsqueeze(1).repeat(1,N,1)        # [B, N, H]
        branch_feat_k = self.branch_net_k(branch_input_k.to(torch.float32)).unsqueeze(1).repeat(1,N,1)        # [B, N, H]
        branch_feat_d = self.branch_net_d(branch_input_d.to(torch.float32)).unsqueeze(1).repeat(1,N,1)        # [B, N, H]
        branch_feat = torch.cat([branch_feat_a, branch_feat_e, branch_feat_k, branch_feat_d] , dim=-1)               # [B, N, 3H] 
                                                                              # [B, N, H]  
        return trunk_feat_init, branch_feat

class SingleDownstreamTask(nn.Module):
    def __init__(self, base_model, width):
        super(SingleDownstreamTask, self).__init__()

        """
        The DownstreamTask has 1 operator that takes a base model with l layers that spits an intermediate function value
        then processes it with an MLP head
    
        """

        self.base_model = base_model
        self.width = width
  
        self.mlp_trunk = nn.Sequential(
            nn.Linear(self.width, self.width),
            nn.ReLU(),
            nn.Linear(self.width, self.width)
        )

        self.mlp_comb = nn.Sequential(
            nn.Linear(self.width, self.width),
            nn.ReLU(),
            nn.Linear(self.width, 1)
        )

    def forward(self, trunk_input, branch_input_a, branch_input_e, branch_input_k, branch_input_d): 
        trunk_feat, branch_feat = self.base_model(trunk_input.to(torch.float), branch_input_a.to(torch.float), branch_input_e.to(torch.float), branch_input_k.to(torch.float).to(torch.float), branch_input_d.to(torch.float))
        trunk_final = self.mlp_trunk(F.relu(trunk_feat))
        x = torch.cat([trunk_final, branch_feat], dim=-1)
        x = self.mlp_comb(F.relu(self.base_model.comb_layer_init(x)))
        return x
    
class MultiDownstreamTask(nn.Module):
    def __init__(self, base_model, width, num_of_tasks, multiple_trunks, multiple_combs):
        super(MultiDownstreamTask, self).__init__()

        """
        The DownstreamTask has multiple operators that share a base model with l layers that spits an intermediate function value
        for multiple tasks and processes them with separate heads
        """

        self.base_model = base_model
        self.width = width
        self.num_tasks = num_of_tasks
        self.multiple_trunks = multiple_trunks
        self.multiple_combs = multiple_combs

        if multiple_trunks:
            self.mlps_trunk = nn.ModuleList()
            for _ in range(num_of_tasks):
                self.mlps_trunk.append(nn.Sequential(nn.Linear(self.width, self.width),
                                                    nn.ReLU(),
                                                    nn.Linear(self.width, self.width)))
        else:
            print("single trunk")
            self.mlps_trunk = nn.Sequential(nn.Linear(self.width, self.width),
                                            nn.ReLU(),
                                            nn.Linear(self.width, self.width))
        
        if multiple_combs:
            self.mlps_comb = nn.ModuleList()
            for _ in range(num_of_tasks):
                self.mlps_comb.append(nn.Sequential(nn.Linear(self.width, self.width),
                                                    nn.ReLU(),
                                                    nn.Linear(self.width, 1)))
        else:
            print("single comb")
            self.mlps_comb = nn.Sequential(nn.Linear(self.width, self.width),
                                            nn.ReLU(),
                                            nn.Linear(self.width, 1))

    def forward(self, trunk_inputs, branch_input_as, branch_input_es, branch_input_ks, branch_input_ds):  
        
        inter_trunk = []
        inter_branch = []
        for i, trunk_input in enumerate(trunk_inputs):
            trunk, branch = self.base_model(trunk_input.to(torch.float), branch_input_as[i].to(torch.float), 
                                          branch_input_es[i].to(torch.float), branch_input_ks[i].to(torch.float).to(torch.float), branch_input_ds[i].to(torch.float))
            inter_trunk.append(trunk)
            inter_branch.append(branch)

        outs = []
        for i, trunk in enumerate(inter_trunk):
            if self.multiple_combs and self.multiple_trunks:
                outs.append(self.mlps_comb[i](F.relu(self.base_model.comb_layer_init(torch.cat([self.mlps_trunk[i](F.relu(trunk)), inter_branch[i]], dim=-1)))))
            elif self.multiple_combs and (not self.multiple_trunks):
                outs.append(self.mlps_comb[i](F.relu(self.base_model.comb_layer_init(torch.cat([self.mlps_trunk(F.relu(trunk)), inter_branch[i]], dim=-1)))))
            elif (not self.multiple_combs) and self.multiple_trunks:
                outs.append(self.mlps_comb(F.relu(self.base_model.comb_layer_init(torch.cat([self.mlps_trunk[i](F.relu(trunk)), inter_branch[i]], dim=-1)))))
            else:
                outs.append(self.mlps_comb(F.relu(self.base_model.comb_layer_init(torch.cat([self.mlps_trunk(F.relu(trunk)), inter_branch[i]], dim=-1)))))
        return outs
    
# class SingleHeadMultiDownstreamTask(nn.Module):
#     def __init__(self, base_model, width, num_of_tasks):
#         super(SingleHeadMultiDownstreamTask, self).__init__()

#         """
#         The DownstreamTask has multiple operators that share a base model with l layers that spits an intermediate function value
#         for multiple tasks and processes them with a single head
#         """

#         self.base_model = base_model
#         self.width = width
#         self.num_tasks = num_of_tasks

#         self.mlp_trunk = nn.Sequential(
#             nn.Linear(self.width, self.width),
#             nn.ReLU(),
#             nn.Linear(self.width, self.width)
#         )

#         self.mlp_comb = nn.Sequential(
#             nn.Linear(self.width, self.width),
#             nn.ReLU(),
#             nn.Linear(self.width, 1)
#         )

#     def forward(self, trunk_inputs, branch_input_as, branch_input_es, branch_input_ks, branch_input_ds):  
        
#         inter_trunk = []
#         inter_branch = []
#         for i, trunk_input in enumerate(trunk_inputs):
#             trunk, branch = self.base_model(trunk_input.to(torch.float), branch_input_as[i].to(torch.float), 
#                                           branch_input_es[i].to(torch.float), branch_input_ks[i].to(torch.float).to(torch.float), branch_input_ds[i].to(torch.float))
#             inter_trunk.append(trunk)
#             inter_branch.append(branch)

#         outs = []
#         for i, trunk in enumerate(inter_trunk):
#             outs.append(self.mlp_comb(F.relu(self.base_model.comb_layer_init(torch.cat([self.mlp_trunk(F.relu(trunk)), inter_branch[i]], dim=-1)))))

#         return outs
    
class DownstreamTaskFrozenBaseNewHead(nn.Module):
    def __init__(self, base_model, width, mlp_head_trunk, mlp_head_comb, new_trunk, new_comb):
        super(DownstreamTaskFrozenBaseNewHead, self).__init__()

        """
        The DownstreamTask2 has 1 operator that takes Frozen base model with l layers that spits an intermediate function value
        then processes it with a newly initialized head
        """

        self.base_model = base_model
        for param in self.base_model.parameters():
            param.requires_grad = False

        self.width = width

        if new_trunk:
            self.mlp_trunk = nn.Sequential(
                nn.Linear(self.width, self.width),
                nn.ReLU(),
                nn.Linear(self.width, self.width)
            )
        else:
            self.mlp_trunk = mlp_head_trunk
            print("frozen old trunk")
            for param in self.mlp_trunk.parameters():
                param.requires_grad = False

        if new_comb:
            self.mlp_comb = nn.Sequential(
                nn.Linear(self.width, self.width),
                nn.ReLU(),
                nn.Linear(self.width, 1)
            )
        else:
            self.mlp_comb = mlp_head_comb
            print("frozen old comb")
            for param in self.mlp_comb.parameters():
                param.requires_grad = False

    def forward(self, trunk_input, branch_input_a, branch_input_e, branch_input_k, branch_input_d):  
        trunk_feat, branch_feat = self.base_model(trunk_input.to(torch.float), branch_input_a.to(torch.float), branch_input_e.to(torch.float), branch_input_k.to(torch.float).to(torch.float), branch_input_d.to(torch.float))
        trunk_final = self.mlp_trunk(F.relu(trunk_feat))
        x = torch.cat([trunk_final, branch_feat], dim=-1)
        x = self.mlp_comb(F.relu(self.base_model.comb_layer_init(x)))
        return x
    
class DownstreamTaskFrozenBaseOldHead(nn.Module):
    def __init__(self, base_model, width, mlp_head_trunk, mlp_head_comb, freeze_trunk, freeze_comb):
        super(DownstreamTaskFrozenBaseOldHead, self).__init__()

        """
        The DownstreamTask2 has 1 operator that takes Frozen base model with l layers that spits an intermediate function value
        then processes it with an old head
        """

        self.base_model = base_model
        for param in self.base_model.parameters():
            param.requires_grad = False

        self.width = width

        self.mlp_trunk = mlp_head_trunk
        self.mlp_comb = mlp_head_comb

        if freeze_trunk:
            print("freeze trunk")
            for param in self.mlp_trunk.parameters():
                param.requires_grad = False
        if freeze_comb:
            print("freeze comb")
            for param in self.mlp_comb.parameters():
                param.requires_grad = False

    def forward(self, trunk_input, branch_input_a, branch_input_e, branch_input_k, branch_input_d):  
        trunk_feat, branch_feat = self.base_model(trunk_input.to(torch.float), branch_input_a.to(torch.float), branch_input_e.to(torch.float), branch_input_k.to(torch.float).to(torch.float), branch_input_d.to(torch.float))
        trunk_final = self.mlp_trunk(F.relu(trunk_feat))
        x = torch.cat([trunk_final, branch_feat], dim=-1)
        x = self.mlp_comb(F.relu(self.base_model.comb_layer_init(x)))
        return x
