# -*- coding: utf-8 -*-

import unittest
import logging

import torchutils


class TestDevice(unittest.TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_monkeypatch_device(self):
        torchutils.monkeypatch_device('cpu')

        device = torchutils.device()

        assert device is not None
