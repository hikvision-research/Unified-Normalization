# Copyright (c) Hangzhou Hikvision Digital Technology Co., Ltd. All rights reserved.
#
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
"""
Abstract: Implementation of Unified Normalization
"""

import torch
import torch.nn as nn

# implementation of FP/BP
class UN2dFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, weight, bias, running_var, ema_gz, eps, momentum, buffer_x2, buffer_gz, iters, fp_buffer_size, bp_buffer_size, warmup_iters, outlier_filtration, skip_counter, mask_x):
        ctx.eps = eps
        ctx.bp_buffer_size = bp_buffer_size
        current_iter = iters.item()
        ctx.current_iter = current_iter
        ctx.warmup_iters = warmup_iters
        ctx.momentum = momentum

        skip = False if fp_buffer_size > 1 else True

        B, C, H, W = x.size()
        x2 = (mask_x * mask_x).mean(dim=0)

        # outlier
        recorded_var = torch.var(torch.sqrt(buffer_x2))                                 # variance of sequence {sqrt(x2)}
        buffer_x2[current_iter % fp_buffer_size].copy_(x2)                              # update buffer
        if current_iter >= warmup_iters and outlier_filtration:
            am = buffer_x2.mean(dim=0).view(1, C, 1, 1)                                 # arithmetic mean
            gm = torch.exp(torch.mean(torch.log(buffer_x2), dim=0)).view(1, C, 1, 1)    # geometric mean
            diff = am - gm
            if (diff > recorded_var * fp_buffer_size).sum() > 0:                        # compare with recorded variance
                # skip current batch, and update buffer with running statistic
                buffer_x2[current_iter % fp_buffer_size].copy_(running_var.view(-1))
                skip = True
                skip_counter.copy_(skip_counter+1)          # counting

        if current_iter <= fp_buffer_size or current_iter < warmup_iters or skip:
            var = x2.view(1, C, 1, 1)
        else:
            var = torch.exp(torch.mean(torch.log(buffer_x2), dim=0)).view(1, C, 1, 1)   # numerically-stable version of the geometric mean
        
        # pass the scale to BP
        r = (var + eps).sqrt() / (running_var + eps).sqrt()
        if current_iter <= max(1000, warmup_iters) or skip:
            r = torch.clamp(r, 1, 1)
        else:
            r = torch.clamp(r, 1/5, 5)
        
        z = x /(var + eps).sqrt()

        ctx.smas_skip = skip
        ctx.save_for_backward(z, var, weight, buffer_gz, ema_gz, r)
        
        running_var.copy_(momentum*running_var + (1-momentum)*var)

        y = weight.view(1,C,1,1) * z + bias.view(1,C,1,1)
        return y
    
    @staticmethod
    def backward(ctx, grad_output):
        eps = ctx.eps
        bp_buffer_size = ctx.bp_buffer_size
        current_iter = ctx.current_iter
        warmup_iters = ctx.warmup_iters
        momentum = ctx.momentum
        smas_skip = ctx.smas_skip

        N, C, H, W = grad_output.size()
        z, var, weight, buffer_gz, ema_gz, r = ctx.saved_variables

        # gradient compensation
        y = r * z
        g = grad_output * weight.view(1, C, 1, 1)
        g = g * r
        gz = (g * z).mean(dim=3).mean(dim=2).mean(dim=0)
        
        if smas_skip:
            buffer_gz[current_iter % bp_buffer_size].copy_(ema_gz.view(-1))
        else:
            buffer_gz[current_iter % bp_buffer_size].copy_(gz)
        
        if current_iter <= bp_buffer_size or current_iter < warmup_iters or smas_skip:
            mean_gz = gz.view(1, C, 1, 1)
        else:
            mean_gz = buffer_gz.mean(dim=0).view(1, C, 1, 1)
        
        ema_gz.copy_(ema_gz * momentum + (1 - momentum) * mean_gz)
        if smas_skip:
            approx_grad_g = (g - (mean_gz * z))
        else:
            approx_grad_g = (g - (ema_gz * z))

        gx = 1. / torch.sqrt(var + eps) * approx_grad_g 
        return gx, (grad_output * y).sum(dim=3).sum(dim=2).sum(dim=0), grad_output.sum(dim=3).sum(dim=2).sum(dim=0),  None, None, None, None, None, None, None, None, None, None, None, None, None

class UN1d(nn.Module):
    """
    Args:
        channels: :math:`C` from an expected input of size :math:`(B, N, C)`
        window_size: the window size to save batch statistics from last several iterations.
            default: 4
        warmup_iters: the number of iterations before using moving average strategies to normalize input.
            Default: 4000
        outlier_filtration: adaptive threshold for applying moving averaging strategies in FP/BP; Suggest to set as False for NLP tasks, and True for CV tasks.
            Default: False
        eps: a small value added to the denominator for numerical stability.
            Default: 1e-5
        momentum: the value used for the exponential moving average. 
            It should be in the limit of :math`(0, 1)`. 
            Default: 0.9
    """
    def __init__(self, channels, window_size=4, warmup_iters=4000, outlier_filtration=False, eps=1e-5, momentum=0.9):
        super(UN1d, self).__init__()
        assert (isinstance(window_size, int) 
                or (isinstance(window_size, list) and len(window_size) == 2)
                or isinstance(window_size, tuple)), "window_size should be in the type of int/list/tuple, but got {} {}".format(type(window_size), window_size)
        
        if isinstance(window_size, int):
            self.fp_buffer_size = window_size
            self.bp_buffer_size = window_size
        else:
            self.fp_buffer_size = window_size[0]
            self.bp_buffer_size = window_size[1]
        
        self.register_parameter('weight', nn.Parameter(torch.ones(channels)))
        self.register_parameter('bias', nn.Parameter(torch.zeros(channels)))
        self.register_buffer('running_var', torch.ones(channels))
        self.register_buffer('ema_gz', torch.zeros(channels))
        self.register_buffer('iters', torch.zeros(1).type(torch.LongTensor))
        self.register_buffer('buffer_x2', torch.zeros(self.fp_buffer_size, channels))
        self.register_buffer('buffer_gz', torch.zeros(self.bp_buffer_size, channels))

        self.num_features = channels
        
        self.eps = eps
        self.momentum = momentum
        self.warmup_iters = warmup_iters
        self.outlier_filtration = outlier_filtration
        self.register_buffer('skip_counter', torch.zeros(1).type(torch.LongTensor))
    
    def extra_repr(self):
        return '{num_features}, fp_win={fp_buffer_size}, bp_win={bp_buffer_size}, eps={eps}, momentum={momentum}, warmup={warmup_iters}, outlier_filtration={outlier_filtration}'.format(**self.__dict__)

    def forward(self, x, pad_mask=None):
        """
        input:  B x N x C
        pad_mask: B x N (padding is True)
        """

        shaped_input = (len(x.shape) == 2)
        if shaped_input:
            x = x.unsqueeze(0)
        B, N, C = x.shape

        # construct the mask_input, size to be (BxN) x C: N is the real length here
        if pad_mask is None:
            mask_input = x.clone()
        else:
            bn_mask = ~pad_mask
            mask_input = x[bn_mask, :]
        
        mask_input = mask_input.reshape(-1, self.num_features)

        x = x.permute(0, 2, 1).contiguous()                 # BxNxC -> BxCxN
        input_shape = x.size()
        x = x.reshape(x.size(0), self.num_features, -1)     # BxCxN -> BxCxNx1
        x = x.unsqueeze(-1)
        
        if self.training:
            self.iters.copy_(self.iters + 1)
            x = UN2dFunction.apply(x, self.weight, self.bias, self.running_var.view(1,self.num_features,1,1), self.ema_gz.view(1,self.num_features,1,1), self.eps, 
                                        self.momentum, self.buffer_x2, self.buffer_gz, self.iters, 
                                        self.fp_buffer_size, self.bp_buffer_size, self.warmup_iters, self.outlier_filtration, self.skip_counter, mask_input)
        else:
            B, C, H, W = x.size()
            var = self.running_var.view(1, C, 1, 1)
            x = x / (var + self.eps).sqrt()
            x = self.weight.view(1,C,1,1) * x + self.bias.view(1,C,1,1)

        
        x = x.reshape(input_shape)
        x = x.permute(0, 2, 1).contiguous()
        
        # Reshape it.
        if shaped_input:
            x = x.squeeze(0)

        return x
