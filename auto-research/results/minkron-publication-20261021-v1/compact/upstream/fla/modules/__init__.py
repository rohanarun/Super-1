"""Explicit CPU compatibility backend for the unmodified authors' layer.

These are native PyTorch equivalents of the used convolution and normalization
operations, not the authors' fused kernels. No packed/padded sequence support.
The official implementation normalizes BEFORE applying its sigmoid output gate.
"""
import torch
from torch import nn
from torch.nn import functional as F

class RMSNorm(nn.Module):
    def __init__(self,hidden_size,eps=1e-5,**kwargs):
        super().__init__();self.weight=nn.Parameter(torch.ones(hidden_size));self.eps=eps
    def forward(self,x):
        acc=x if x.dtype==torch.float64 else x.float()
        return (acc*torch.rsqrt(acc.square().mean(-1,keepdim=True)+self.eps)*self.weight).to(x.dtype)

class FusedRMSNormGated(RMSNorm):
    def __init__(self,hidden_size,activation='sigmoid',eps=1e-5,**kwargs):
        assert activation=='sigmoid';super().__init__(hidden_size,eps);self.activation=activation
    def forward(self,x,g):return super().forward(x)*torch.sigmoid(g)

class ShortConvolution(nn.Conv1d):
    def __init__(self,hidden_size,kernel_size,bias=False,activation='silu',**kwargs):
        super().__init__(hidden_size,hidden_size,kernel_size,padding=kernel_size-1,groups=hidden_size,bias=bias)
        assert activation in ('silu','swish');self.activation=activation
    def forward(self,x,cache=None,output_final_state=False,cu_seqlens=None,**kwargs):
        assert cu_seqlens is None
        W=self.kernel_size[0];xt=x.transpose(1,2)
        if cache is None:cache=x.new_zeros(x.shape[0],x.shape[-1],W)
        combined=torch.cat((cache,xt),-1)
        # The cache owns W inputs; W-1 are necessary for the next convolution.
        o=F.conv1d(combined[:,:,1:],self.weight,self.bias,groups=self.groups).transpose(1,2)
        state=combined[:,:,-W:].clone() if output_final_state else None
        return F.silu(o),state
