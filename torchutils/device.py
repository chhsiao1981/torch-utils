# -*- coding: utf-8 -*-
'''
torchutils.device

setup torch device.
'''

import torch

_DEVICE = None


def monkeypatch_device(device_str: str = 'cuda', is_set_default_device: bool = True):
    global _DEVICE

    _DEVICE = torch.device(device_str)

    if is_set_default_device:
        torch.set_default_device(_DEVICE)


def device():
    return _DEVICE
