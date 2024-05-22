# -*- coding: utf-8 -*-

# random
from .random import init_meta_rng
from .random import monkeypatch_meta_rng

from .random import monkeypatch_seed

from .random import reset_meta_rng

from .random import get_meta_rng_state
from .random import set_meta_rng_state

# device
from .device import monkeypatch_device
from .device import device
