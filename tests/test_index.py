# -*- coding: utf-8 -*-

import unittest
import logging

import torch

import torchutils


class TestIndex(unittest.TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_index(self):
        a = torch.Tensor(4, 3)
        ret = torchutils.index(a)
        assert list(ret.shape) == [2, 4, 3]

        expected = torch.Tensor(
            [
                # 0 0 0
                # 1 1 1
                # 2 2 2
                # 3 3 3
                [[0, 0, 0], [1, 1, 1], [2, 2, 2], [3, 3, 3]],
                # 0 1 2
                # 0 1 2
                # 0 1 2
                # 0 1 2
                [[0, 1, 2], [0, 1, 2], [0, 1, 2], [0, 1, 2]],
            ],
        )

        assert (ret == expected).all()

    def test_index2(self):
        a = torch.Tensor(2, 4, 5)
        ret = torchutils.index(a)
        assert list(ret.shape) == [3, 2, 4, 5]

        expected = torch.Tensor(
            [
                [
                    [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]],
                    [[1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1]],
                ],
                [
                    [[0, 0, 0, 0, 0], [1, 1, 1, 1, 1], [2, 2, 2, 2, 2], [3, 3, 3, 3, 3]],
                    [[0, 0, 0, 0, 0], [1, 1, 1, 1, 1], [2, 2, 2, 2, 2], [3, 3, 3, 3, 3]],
                ],
                [
                    [[0, 1, 2, 3, 4], [0, 1, 2, 3, 4], [0, 1, 2, 3, 4], [0, 1, 2, 3, 4]],
                    [[0, 1, 2, 3, 4], [0, 1, 2, 3, 4], [0, 1, 2, 3, 4], [0, 1, 2, 3, 4]],
                ],
            ],
        )

        assert (ret == expected).all()
