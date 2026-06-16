"""
This code belongs to the paper:
-- Tripura, T., & Chakraborty, S. (2022). Wavelet Neural Operator for solving 
   parametric partialdifferential equations in computational mechanics problems.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from timeit import default_timer
from scipy.sparse import lil_matrix
import scipy
import time
from torch_geometric.utils import get_laplacian


""" Load required packages 
-- "PyWavelets"
    https://pywavelets.readthedocs.io/en/latest/install.html
    ($ conda install pywavelets)
"""

from torch.nn.parameter import Parameter


import ptwt, pywt
from ptwt.conv_transform_3 import wavedec3, waverec3
from pytorch_wavelets import DWT1D, IDWT1D
from pytorch_wavelets import DTCWTForward, DTCWTInverse
from pytorch_wavelets import DWT, IDWT 

def truncated_eigen_decomposition(L, k1, k2):

    lambdas_low, U_low = torch.lobpcg(L, k = k1, largest = False, tol = 1E-3)

    # lambdas_high, U_high = torch.lobpcg(L, k = k2, largest = True, tol = 1E-3)
    # lambdas = np.concatenate((lambdas_low, lambdas_high))
    # U = np.concatenate((U_low, U_high), axis=1)
    return lambdas_low, U_low

def calculate_lambdas_U_truncated_sparse(edge_index, num_nodes, max_mode = 64):
    t1 = time.time()
    L_norm_index , L_norm_values = get_laplacian(edge_index, normalization='sym')
    L_norm = torch.sparse_coo_tensor(indices = L_norm_index, values = L_norm_values, size=[num_nodes, num_nodes])
    print(L_norm.shape)
    t2 = time.time()
    print('time for Laplacian construction from edge_index', (t2 -t1))
    t1 = time.time()
    lambdas, U = truncated_eigen_decomposition(L_norm , k1 = max_mode , k2 = 1)
    t2 = time.time()
    print('time for eigen value decomp', (t2 -t1))
    return lambdas, U

def calculate_lambdas_U_truncated_edgeweight_sparse(edge_index, edge_weight, num_nodes, max_mode = 64):
    t1 = time.time()
    L_norm_index , L_norm_values = get_laplacian(edge_index, edge_weight, normalization='sym')
    L_norm = torch.sparse_coo_tensor(indices = L_norm_index, values = L_norm_values, size=[num_nodes, num_nodes])
    t2 = time.time()
    print('time for Laplacian construction from edge_index', (t2 -t1))
    t1 = time.time()
    lambdas, U = truncated_eigen_decomposition(L_norm , k1 = max_mode , k2 = 1)
    t2 = time.time()
    print('time for eigen value decomp', (t2 -t1))
    return lambdas, U

class MLP(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels):
        super(MLP, self).__init__()
        self.mlp1 = nn.Linear(in_channels, mid_channels)
        self.mlp2 = nn.Linear(mid_channels, out_channels)

    def forward(self, x):
        x = self.mlp1(x)
        x = F.gelu(x)
        x = self.mlp2(x)
        return x

class MLPDropout(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels, prob=0.05):
        super(MLPDropout, self).__init__()
        self.mlp1 = nn.Linear(in_channels, mid_channels)
        self.mlp2 = nn.Linear(mid_channels, out_channels)
        self.dropout = nn.Dropout(prob)

    def forward(self, x):
        x = self.mlp1(x)
        x = F.gelu(x)
        x = self.dropout(x)
        x = self.mlp2(x)
        return x


""" Def: 2d Wavelet convolutional layer (discrete) """
class WaveConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, level, size, wavelet, mode='symmetric', omega=8):
        super(WaveConv2d, self).__init__()

        """
        2D Wavelet layer. It does DWT, linear transform, and Inverse dWT. 
        
        Input parameters: 
        -----------------
        in_channels  : scalar, input kernel dimension
        out_channels : scalar, output kernel dimension
        level        : scalar, levels of wavelet decomposition
        size         : scalar, length of input 1D signal
        wavelet      : string, wavelet filters
        mode         : string, padding style for wavelet decomposition
        
        It initializes the kernel parameters: 
        -------------------------------------
        self.weights1 : tensor, shape-[in_channels * out_channels * x * y]
                        kernel weights for Approximate wavelet coefficients
        self.weights2 : tensor, shape-[in_channels * out_channels * x * y]
                        kernel weights for Horizontal-Detailed wavelet coefficients
        self.weights3 : tensor, shape-[in_channels * out_channels * x * y]
                        kernel weights for Vertical-Detailed wavelet coefficients
        self.weights4 : tensor, shape-[in_channels * out_channels * x * y]
                        kernel weights for Diagonal-Detailed wavelet coefficients
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.level = level
        if isinstance(size, list):
            if len(size) != 2:
                raise Exception('size: WaveConv2dCwt accepts the size of 2D signal in list with 2 elements')
            else:
                self.size = size
        else:
            raise Exception('size: WaveConv2dCwt accepts size of 2D signal is list')
        self.wavelet = wavelet       
        self.mode = mode
        dummy_data = torch.randn( 1,1,*self.size )        
        dwt_ = DWT(J=self.level, mode=self.mode, wave=self.wavelet)
        mode_data, _ = dwt_(dummy_data)
        self.modes1 = mode_data.shape[-2]
        self.modes2 = mode_data.shape[-1]
        self.omega = omega
        self.effective_modes_x = self.modes1//self.omega+1
        self.effective_modes_y = self.modes2//self.omega+1
        
        # Parameter initilization
        self.scale = (1 / (in_channels * out_channels))
        self.weights_a1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.effective_modes_x, self.effective_modes_x, dtype=torch.cfloat))
        self.weights_a2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.effective_modes_x, self.effective_modes_x, dtype=torch.cfloat))
        self.weights_h1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.effective_modes_x, self.effective_modes_x, dtype=torch.cfloat))
        self.weights_h2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.effective_modes_x, self.effective_modes_x, dtype=torch.cfloat))
        self.weights_v1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.effective_modes_x, self.effective_modes_x, dtype=torch.cfloat))
        self.weights_v2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.effective_modes_x, self.effective_modes_x, dtype=torch.cfloat))
        self.weights_d1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.effective_modes_x, self.effective_modes_x, dtype=torch.cfloat))
        self.weights_d2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.effective_modes_x, self.effective_modes_x, dtype=torch.cfloat))

    # Element-wise multiplication
    def mul2d(self, input, weights):
        """
        Performs element-wise multiplication

        Input Parameters
        ----------------
        input   : tensor, shape-(batch * in_channel * x * y )
                  2D wavelet coefficients of input signal
        weights : tensor, shape-(in_channel * out_channel * x * y)
                  kernel weights of corresponding wavelet coefficients

        Returns
        -------
        convolved signal : tensor, shape-(batch * out_channel * x * y)
        """
        return torch.einsum("bixy,ioxy->boxy", input, weights)
    
    # Spectral Convolution
    def spectralconv(self, waves, weights1, weights2):
        """
        Performs spectral convolution using Fourier decomposition

        Input Parameters
        ----------
        waves : tensor, shape-[Batch * Channel * size(x)]
                signal to be convolved, here the wavelet coefficients.
        weights : tensor, shape-[in_channel * out_channel * size(x)]
                The weights/kernel of the neural network.

        Returns
        -------
        convolved signal : tensor, shape-[batch * out_channel * size(x)]

        """
        # Get the frequency componenets
        xw = torch.fft.rfft2(waves)
        
        # Initialize the output
        conv_out = torch.zeros(waves.shape[0], self.out_channels, waves.shape[-2], waves.shape[-1]//2+1, dtype=torch.cfloat, device=waves.device)
        
        # Perform Element-wise multiplication in spectral doamin
        conv_out[:,:,:self.effective_modes_x,:self.effective_modes_y] = self.mul2d(xw[:,:,:self.effective_modes_x,:self.effective_modes_y], weights1)
        conv_out[:,:,-self.effective_modes_x:,:self.effective_modes_y] = self.mul2d(xw[:,:,-self.effective_modes_x:,:self.effective_modes_y], weights2)
        return torch.fft.irfft2(conv_out, s=(waves.shape[-2], waves.shape[-1]))
    
    def forward(self, x):
        """
        Input parameters: 
        -----------------
        x : tensor, shape-[Batch * Channel * x * y]
        Output parameters: 
        ------------------
        x : tensor, shape-[Batch * Channel * x * y]
        """
        if x.shape[-1] > self.size[-1]:
            factor = int(np.log2(x.shape[-1] // self.size[-1]))
            
            # Compute single tree Discrete Wavelet coefficients using some wavelet
            dwt = DWT(J=self.level+factor, mode=self.mode, wave=self.wavelet).to(x.device)
            x_ft, x_coeff = dwt(x)
            
        elif x.shape[-1] < self.size[-1]:
            factor = int(np.log2(self.size[-1] // x.shape[-1]))
            
            # Compute single tree Discrete Wavelet coefficients using some wavelet
            dwt = DWT(J=self.level-factor, mode=self.mode, wave=self.wavelet).to(x.device)
            x_ft, x_coeff = dwt(x)
        
        else:
            # Compute single tree Discrete Wavelet coefficients using some wavelet
            dwt = DWT(J=self.level, mode=self.mode, wave=self.wavelet).to(x.device)
            x_ft, x_coeff = dwt(x)

        # Instantiate higher level coefficients as zeros
        out_ft = torch.zeros_like(x_ft, device= x.device)
        out_coeff = [torch.zeros_like(coeffs, device= x.device) for coeffs in x_coeff]
        
        # Multiply the final approximate Wavelet modes
        out_ft = self.spectralconv(x_ft, self.weights_a1, self.weights_a2)
        
        # Multiply the final detailed wavelet coefficients
        out_coeff[-1][:,:,0,:,:] = self.spectralconv(x_coeff[-1][:,:,0,:,:].clone(), self.weights_h1, self.weights_h2)
        out_coeff[-1][:,:,1,:,:] = self.spectralconv(x_coeff[-1][:,:,1,:,:].clone(), self.weights_v1, self.weights_v2)
        out_coeff[-1][:,:,2,:,:] = self.spectralconv(x_coeff[-1][:,:,2,:,:].clone(), self.weights_d1, self.weights_d2)
        
        # Return to physical space        
        idwt = IDWT(mode=self.mode, wave=self.wavelet).to(x.device)
        x = idwt((out_ft, out_coeff))
        return x

     
    
""" Def: 2d Wavelet convolutional layer (slim continuous) """
class WaveConv2dCwt(nn.Module):
    def __init__(self, in_channels, out_channels, level, size, wavelet1, wavelet2, omega=8):
        super(WaveConv2dCwt, self).__init__()

        """
        !! It is computationally expensive than the discrete "WaveConv2d" !!
        2D Wavelet layer. It does SCWT (Slim continuous wavelet transform),
                                linear transform, and Inverse dWT. 
        
        Input parameters: 
        -----------------
        in_channels  : scalar, input kernel dimension
        out_channels : scalar, output kernel dimension
        level        : scalar, levels of wavelet decomposition
        size         : scalar, length of input 1D signal
        wavelet1     : string, Specifies the first level biorthogonal wavelet filters
        wavelet2     : string, Specifies the second level quarter shift filters
        mode         : string, padding style for wavelet decomposition
        
        It initializes the kernel parameters: 
        -------------------------------------
        self.weights0 : tensor, shape-[in_channels * out_channels * x * y]
                        kernel weights for Approximate wavelet coefficients
        self.weights- 15r, 45r, 75r, 105r, 135r, 165r : tensor, shape-[in_channels * out_channels * x * y]
                        kernel weights for REAL wavelet coefficients at 15, 45, 75, 105, 135, 165 angles
        self.weights- 15c, 45c, 75c, 105c, 135c, 165c : tensor, shape-[in_channels * out_channels * x * y]
                        kernel weights for COMPLEX wavelet coefficients at 15, 45, 75, 105, 135, 165 angles
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.level = level
        if isinstance(size, list):
            if len(size) != 2:
                raise Exception('size: WaveConv2dCwt accepts the size of 2D signal in list with 2 elements')
            else:
                self.size = size
        else:
            raise Exception('size: WaveConv2dCwt accepts size of 2D signal is list')
        self.wavelet_level1 = wavelet1
        self.wavelet_level2 = wavelet2        
        dummy_data = torch.randn( 1,1,*self.size ) 
        dwt_ = DTCWTForward(J=self.level, biort=self.wavelet_level1, qshift=self.wavelet_level2)
        mode_data, mode_coef = dwt_(dummy_data)
        self.modes1 = mode_data.shape[-2]
        self.modes2 = mode_data.shape[-1]
        self.modes21 = mode_coef[-1].shape[-3]
        self.modes22 = mode_coef[-1].shape[-2]
        self.omega = omega
        self.effective_modes_x = self.modes1//self.omega+1
        self.effective_modes_y = self.modes2//self.omega+1
        self.effective_modes_xx = self.modes21//self.omega+1
        self.effective_modes_yy = self.modes22//self.omega+1
        
        # Parameter initilization
        self.scale = (1 / (in_channels * out_channels))
        self.weights_01 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.effective_modes_x, self.effective_modes_y, dtype=torch.cfloat))
        self.weights_02 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.effective_modes_x, self.effective_modes_y, dtype=torch.cfloat))
        self.weights_15r1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.effective_modes_xx, self.effective_modes_yy, dtype=torch.cfloat))
        self.weights_15r2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.effective_modes_xx, self.effective_modes_yy, dtype=torch.cfloat))
        self.weights_15c1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.effective_modes_xx, self.effective_modes_yy, dtype=torch.cfloat))
        self.weights_15c2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.effective_modes_xx, self.effective_modes_yy, dtype=torch.cfloat))
        self.weights_45r1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.effective_modes_xx, self.effective_modes_yy, dtype=torch.cfloat))
        self.weights_45r2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.effective_modes_xx, self.effective_modes_yy, dtype=torch.cfloat))
        self.weights_45c1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.effective_modes_xx, self.effective_modes_yy, dtype=torch.cfloat))
        self.weights_45c2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.effective_modes_xx, self.effective_modes_yy, dtype=torch.cfloat))
        self.weights_75r1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.effective_modes_xx, self.effective_modes_yy, dtype=torch.cfloat))
        self.weights_75r2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.effective_modes_xx, self.effective_modes_yy, dtype=torch.cfloat))
        self.weights_75c1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.effective_modes_xx, self.effective_modes_yy, dtype=torch.cfloat))
        self.weights_75c2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.effective_modes_xx, self.effective_modes_yy, dtype=torch.cfloat))
        self.weights_105r1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.effective_modes_xx, self.effective_modes_yy, dtype=torch.cfloat))
        self.weights_105r2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.effective_modes_xx, self.effective_modes_yy, dtype=torch.cfloat))
        self.weights_105c1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.effective_modes_xx, self.effective_modes_yy, dtype=torch.cfloat))
        self.weights_105c2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.effective_modes_xx, self.effective_modes_yy, dtype=torch.cfloat))
        self.weights_135r1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.effective_modes_xx, self.effective_modes_yy, dtype=torch.cfloat))
        self.weights_135r2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.effective_modes_xx, self.effective_modes_yy, dtype=torch.cfloat))
        self.weights_135c1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.effective_modes_xx, self.effective_modes_yy, dtype=torch.cfloat))
        self.weights_135c2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.effective_modes_xx, self.effective_modes_yy, dtype=torch.cfloat))
        self.weights_165r1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.effective_modes_xx, self.effective_modes_yy, dtype=torch.cfloat))
        self.weights_165r2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.effective_modes_xx, self.effective_modes_yy, dtype=torch.cfloat))
        self.weights_165c1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.effective_modes_xx, self.effective_modes_yy, dtype=torch.cfloat))
        self.weights_165c2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.effective_modes_xx, self.effective_modes_yy, dtype=torch.cfloat))

    # Convolution
    def mul2d(self, input, weights):
        """
        Performs element-wise multiplication

        Input Parameters
        ----------------
        input   : tensor, shape-(batch * in_channel * x * y )
                  2D wavelet coefficients of input signal
        weights : tensor, shape-(in_channel * out_channel * x * y)
                  kernel weights of corresponding wavelet coefficients

        Returns
        -------
        convolved signal : tensor, shape-(batch * out_channel * x * y)
        """
        return torch.einsum("bixy,ioxy->boxy", input, weights)
    
    # Spectral Convolution
    def spectralconv(self, waves, weights1, weights2):
        """
        Performs spectral convolution using Fourier decomposition

        Input Parameters
        ----------
        waves : tensor, shape-[Batch * Channel * size(x)]
                signal to be convolved, here the wavelet coefficients.
        weights : tensor, shape-[in_channel * out_channel * size(x)]
                The weights/kernel of the neural network.

        Returns
        -------
        convolved signal : tensor, shape-[batch * out_channel * size(x)]

        """
        # Get the frequency componenets
        modes1, modes2 = weights2.shape[-2], weights2.shape[-1]
        xw = torch.fft.rfft2(waves)
        
        # Initialize the output
        conv_out = torch.zeros(waves.shape[0], self.out_channels, waves.shape[-2], waves.shape[-1]//2+1, dtype=torch.cfloat, device=waves.device)
        
        # Perform Element-wise multiplication in spectral doamin
        conv_out[:,:,:modes1,:modes2] = self.mul2d(xw[:,:,:modes1,:modes2], weights1)
        conv_out[:,:,-modes1:,:modes2] = self.mul2d(xw[:,:,-modes1:,:modes2], weights2)
        return torch.fft.irfft2(conv_out, s=(waves.shape[-2], waves.shape[-1]))

    def forward(self, x):
        """
        Input parameters: 
        -----------------
        x : tensor, shape-[Batch * Channel * x * y]
        Output parameters: 
        ------------------
        x : tensor, shape-[Batch * Channel * x * y]
        """      
        if x.shape[-1] > self.size[-1]:
            factor = int(np.log2(x.shape[-1] // self.size[-1]))
            
            # Compute dual tree continuous Wavelet coefficients
            cwt = DTCWTForward(J=self.level+factor, biort=self.wavelet_level1, qshift=self.wavelet_level2).to(x.device)
            x_ft, x_coeff = cwt(x)
            
        elif x.shape[-1] < self.size[-1]:
            factor = int(np.log2(self.size[-1] // x.shape[-1]))
            
            # Compute dual tree continuous Wavelet coefficients
            cwt = DTCWTForward(J=self.level-factor, biort=self.wavelet_level1, qshift=self.wavelet_level2).to(x.device)
            x_ft, x_coeff = cwt(x)            
        else:
            # Compute dual tree continuous Wavelet coefficients 
            cwt = DTCWTForward(J=self.level, biort=self.wavelet_level1, qshift=self.wavelet_level2).to(x.device)
            x_ft, x_coeff = cwt(x)
        
        # Instantiate higher level coefficients as zeros
        out_ft = torch.zeros_like(x_ft, device= x.device)
        out_coeff = [torch.zeros_like(coeffs, device= x.device) for coeffs in x_coeff]
        
        # Multiply the final approximate Wavelet modes
        out_ft = self.spectralconv(x_ft, self.weights_01, self.weights_02)
        # Multiply the final detailed wavelet coefficients        
        out_coeff[-1][:,:,0,:,:,0] = self.spectralconv(x_coeff[-1][:,:,0,:,:,0].clone(), self.weights_15r1, self.weights_15r2)
        out_coeff[-1][:,:,0,:,:,1] = self.spectralconv(x_coeff[-1][:,:,0,:,:,1].clone(), self.weights_15c1, self.weights_15c2)
        out_coeff[-1][:,:,1,:,:,0] = self.spectralconv(x_coeff[-1][:,:,1,:,:,0].clone(), self.weights_45r1, self.weights_45r2)
        out_coeff[-1][:,:,1,:,:,1] = self.spectralconv(x_coeff[-1][:,:,1,:,:,1].clone(), self.weights_45c1, self.weights_45c2)
        out_coeff[-1][:,:,2,:,:,0] = self.spectralconv(x_coeff[-1][:,:,2,:,:,0].clone(), self.weights_75r1, self.weights_75r2)
        out_coeff[-1][:,:,2,:,:,1] = self.spectralconv(x_coeff[-1][:,:,2,:,:,1].clone(), self.weights_75c1, self.weights_75c2)
        out_coeff[-1][:,:,3,:,:,0] = self.spectralconv(x_coeff[-1][:,:,3,:,:,0].clone(), self.weights_105r1, self.weights_105r2)
        out_coeff[-1][:,:,3,:,:,1] = self.spectralconv(x_coeff[-1][:,:,3,:,:,1].clone(), self.weights_105c1, self.weights_105c2)
        out_coeff[-1][:,:,4,:,:,0] = self.spectralconv(x_coeff[-1][:,:,4,:,:,0].clone(), self.weights_135r1, self.weights_135r2)
        out_coeff[-1][:,:,4,:,:,1] = self.spectralconv(x_coeff[-1][:,:,4,:,:,1].clone(), self.weights_135c1, self.weights_135c2)
        out_coeff[-1][:,:,5,:,:,0] = self.spectralconv(x_coeff[-1][:,:,5,:,:,0].clone(), self.weights_165r1, self.weights_165r2)
        out_coeff[-1][:,:,5,:,:,1] = self.spectralconv(x_coeff[-1][:,:,5,:,:,1].clone(), self.weights_165c1, self.weights_165c2)        
        
        # Reconstruct the signal
        icwt = DTCWTInverse(biort=self.wavelet_level1, qshift=self.wavelet_level2).to(x.device)
        x = icwt((out_ft, out_coeff))
        return x
    
    
#loss function with rel/abs Lp loss
class LpLoss(object):
    def __init__(self, d=2, p=2, size_average=True, reduction=True):
        super(LpLoss, self).__init__()

        #Dimension and Lp-norm type are postive
        assert d > 0 and p > 0

        self.d = d
        self.p = p
        self.reduction = reduction
        self.size_average = size_average

    def abs(self, x, y):
        num_examples = x.size()[0]

        #Assume uniform mesh
        h = 1.0 / (x.size()[1] - 1.0)

        all_norms = (h**(self.d/self.p))*torch.norm(x.reshape(num_examples,-1) - y.reshape(num_examples,-1), self.p, 1)

        if self.reduction:
            if self.size_average:
                return torch.mean(all_norms)
            else:
                return torch.sum(all_norms)

        return all_norms

    def rel(self, x, y):
        num_examples = x.size()[0]

        diff_norms = torch.norm(x.reshape(num_examples,-1) - y.reshape(num_examples,-1), self.p, 1)
        y_norms = torch.norm(y.reshape(num_examples,-1), self.p, 1)

        if self.reduction:
            if self.size_average:
                return torch.mean(diff_norms/y_norms)
            else:
                return torch.sum(diff_norms/y_norms)

        return diff_norms/y_norms

    def __call__(self, x, y):
        return self.rel(x, y)
    

################################################################
# fourier layer
################################################################
class SpectralConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super(SpectralConv2d, self).__init__()

        """
        2D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1 #Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes2 = modes2

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))

    # Complex multiplication
    def compl_mul2d(self, input, weights):
        # (batch, in_channel, x,y ), (in_channel, out_channel, x,y) -> (batch, out_channel, x,y)
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        #Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfft2(x)

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels,  x.size(-2), x.size(-1)//2 + 1, dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :self.modes1, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, :self.modes1, :self.modes2], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, -self.modes1:, :self.modes2], self.weights2)

        #Return to physical space
        x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))
        return x
