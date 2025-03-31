# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import math

def adjust_learning_rate(optimizer, epoch, **kwargs):
    """Decay the learning rate with half-cycle cosine after warmup"""
    if epoch < kwargs['warmup_epochs']:
        lr = kwargs['lr'] * epoch / kwargs['warmup_epochs'] 
    else:
        lr = kwargs['min_lr'] + (kwargs['lr'] - kwargs['min_lr']) * 0.5 * \
            (1. + math.cos(math.pi * (epoch - kwargs['warmup_epochs']) / (kwargs['epochs'] - kwargs['warmup_epochs'])))
    for param_group in optimizer.param_groups:
        if "lr_scale" in param_group:
            param_group["lr"] = lr * param_group["lr_scale"]
        else:
            param_group["lr"] = lr
    return lr
