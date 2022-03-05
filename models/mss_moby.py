# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Zhenda Xie
# --------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from diffdist import functional


def dist_collect(x):
    """ collect all tensor from all GPUs
    args:
        x: shape (mini_batch, ...)
    returns:
        shape (mini_batch * num_gpu, ...)
    """
    x = x.contiguous()
    out_list = [torch.zeros_like(x, device=x.device, dtype=x.dtype).contiguous()
                for _ in range(dist.get_world_size())]
    out_list = functional.all_gather(out_list, x)
    return torch.cat(out_list, dim=0).contiguous()


class MoBYMSS(nn.Module):

    def __init__(self, cfg, encoder):
        super().__init__()
        
        self.cfg = cfg
        self.encoder = encoder
        if self.cfg.MODEL.SWIN.NORM_BEFORE_MLP == 'bn':
            nn.SyncBatchNorm.convert_sync_batchnorm(self.encoder)

        self.criterion = nn.CrossEntropyLoss()
        pass

    @staticmethod
    def asymmetric_loss(p, z):
        return  - nn.functional.cosine_similarity(p, z.detach(), dim=-1).mean()

    def forward(self, im):
        z1, z2, p1, p2 = self.encoder(im)
        # loss = 0.5 * (self.asymmetric_loss(p1, z2) + self.asymmetric_loss(p2, z1))
        loss = 0.5 * (self.criterion(z1, p2) + self.criterion(z2, p1))
        return loss

    pass
