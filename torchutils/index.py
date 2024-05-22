# -*- coding: utf-8 -*-
'''
torchutils.index

Getting nd-array indexes.
'''

import torch


def index(tensor: torch.Tensor) -> torch.Tensor:
    n_dim = tensor.ndim
    new_shape = [n_dim] + list(tensor.shape)
    ret = torch.Tensor(*new_shape)
    for idx in range(n_dim):
        _index(tensor.shape, idx, ret)

    return ret


def _index(the_shape: torch.Size, idx: int, ret: torch.Tensor):
    pre_idxes = the_shape[:idx]
    post_idxes = the_shape[(idx + 1):]
    pre_slices = [slice(each) for each in pre_idxes]
    post_slices = [slice(each) for each in post_idxes]
    for each_idx in range(the_shape[idx]):
        ret[idx, *pre_slices, each_idx, *post_slices] = each_idx

    return ret
