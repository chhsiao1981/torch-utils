# -*- coding: utf-8 -*-

import unittest
import logging

import random
import numpy as np
import torch

from torchutils import random as torchutils_random


class TestRandom(unittest.TestCase):

    def setUp(self):
        torchutils_random._META_RNG = None

    def tearDown(self):
        torchutils_random._META_RNG = None

    def test_monkeypatch_meta_rng_random(self):
        torchutils_random.monkeypatch_meta_rng()
        nums0 = [random.random() for _0 in range(10)]

        torchutils_random.reset_meta_rng()
        torchutils_random.monkeypatch_meta_rng()
        nums1 = [random.random() for _0 in range(10)]

        assert nums0 == nums1

        torchutils_random.monkeypatch_meta_rng()
        nums2 = [random.random() for _0 in range(10)]

        logging.warning(f'nums1: {nums1} nums2: {nums2}')

        assert nums1 != nums2

    def test_monkeypatch_meta_rng_np(self):
        torchutils_random.monkeypatch_meta_rng()
        nums0 = np.stack([np.random.random() for _0 in range(10)])

        torchutils_random.reset_meta_rng()
        torchutils_random.monkeypatch_meta_rng()
        nums1 = np.stack([np.random.random() for _0 in range(10)])

        assert (nums0 == nums1).all()

        torchutils_random.monkeypatch_meta_rng()
        nums2 = np.stack([np.random.random() for _0 in range(10)])

        logging.warning(f'nums1: {nums1} nums2: {nums2}')

        assert (nums1 != nums2).any()

    def test_monkeypatch_meta_rng_torch(self):
        torchutils_random.monkeypatch_meta_rng()
        nums0 = torch.stack([torch.rand(3, 4) for _0 in range(10)])

        torchutils_random.reset_meta_rng()
        torchutils_random.monkeypatch_meta_rng()
        nums1 = torch.stack([torch.rand(3, 4) for _0 in range(10)])

        assert (nums0 == nums1).all()

        torchutils_random.monkeypatch_meta_rng()
        nums2 = torch.stack([torch.rand(3, 4) for _0 in range(10)])

        logging.warning(f'nums1: {nums1} nums2: {nums2}')

        assert (nums1 != nums2).any()

    def test_set_meta_state(self):
        # if we monkeypatch only 1 time, we cannot get repeated rand-numbers.
        torchutils_random.monkeypatch_meta_rng()
        nums0 = [random.random() for _0 in range(10)]

        state = torchutils_random.get_meta_state()

        nums1 = [random.random() for _0 in range(10)]

        assert nums0 != nums1

        # if we monkeypatch_seed without first setting meta-state, we cannot get repeated rand-numbers.
        torchutils_random.monkeypatch_seed()

        nums2 = [random.random() for _0 in range(10)]

        assert nums1 != nums2

        # we can get repeated rand-numbers if we set meta-state before monkeypatch_seed.
        torchutils_random.set_meta_state(state)
        torchutils_random.monkeypatch_seed()

        nums3 = [random.random() for _0 in range(10)]

        assert nums2 == nums3

    def test_set_meta_state_np(self):
        # if we monkeypatch only 1 time, we cannot get repeated rand-numbers.
        torchutils_random.monkeypatch_meta_rng()
        nums0 = np.stack([np.random.random() for _0 in range(10)])

        state = torchutils_random.get_meta_state()

        nums1 = np.stack([np.random.random() for _0 in range(10)])

        assert (nums0 != nums1).any()

        # if we monkeypatch_seed without first setting meta-state, we cannot get repeated rand-numbers.
        torchutils_random.monkeypatch_seed()

        nums2 = np.stack([random.random() for _0 in range(10)])

        assert (nums1 != nums2).any()

        # we can get repeated rand-numbers if we set meta-state before monkeypatch_seed.
        torchutils_random.set_meta_state(state)
        torchutils_random.monkeypatch_seed()

        nums3 = np.stack([random.random() for _0 in range(10)])

        assert (nums2 == nums3).all()

    def test_set_meta_state_torch(self):
        # if we monkeypatch only 1 time, we cannot get repeated rand-numbers.
        torchutils_random.monkeypatch_meta_rng()
        nums0 = torch.stack([torch.rand(3, 4) for _0 in range(10)])

        state = torchutils_random.get_meta_state()

        nums1 = torch.stack([torch.rand(3, 4) for _0 in range(10)])

        assert (nums0 != nums1).any()

        # if we monkeypatch_seed without first setting meta-state, we cannot get repeated rand-numbers.
        torchutils_random.monkeypatch_seed()

        nums2 = torch.stack([torch.rand(3, 4) for _0 in range(10)])

        assert (nums1 != nums2).any()

        # we can get repeated rand-numbers if we set meta-state before monkeypatch_seed.
        torchutils_random.set_meta_state(state)
        torchutils_random.monkeypatch_seed()

        nums3 = torch.stack([torch.rand(3, 4) for _0 in range(10)])

        assert (nums2 == nums3).all()
