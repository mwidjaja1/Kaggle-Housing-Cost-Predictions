#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 18 12:39:52 2017

@author: Matthew
"""

from argparse import ArgumentParser


def parse(inargs=None):
    """ Input arguments parser """
    parser = ArgumentParser("Kaggle Houses")
    parser.add_argument('train_csv', help='Path to the CSV training data file')
    parser.add_argument('--test_csv', default=None,
                        help='Path to the CSV test data file.')

    if inargs:
        args = parser.parse_args(inargs)
    else:
        args = parser.parse_args()
    return args
