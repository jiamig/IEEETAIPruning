#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import torch
import copy
import numpy as np


def FedAvg(w, pruning_rate):
    """

    """
    w_avg = copy.deepcopy(w[0])
    diff_matrix = {key: np.zeros_like(value) for key, value in w_avg.items()}


    for key in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[key] += w[i][key]
        w_avg[key] = torch.div(w_avg[key], len(w))


    for key in diff_matrix.keys():
        for i in range(len(w)):
            diff_matrix[key] += torch.abs(w[i][key] - w_avg[key])

    for key in diff_matrix.keys():
        sorted_diff, _ = torch.sort(diff_matrix[key].flatten(), descending=True)
        threshold = sorted_diff[int(pruning_rate * len(sorted_diff))]
        mask = diff_matrix[key] >= threshold
        for i in range(len(w)):
            w[i][key][mask] = 0


    w_avg = copy.deepcopy(w[0])
    for key in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[key] += w[i][key]
        w_avg[key] = torch.div(w_avg[key], len(w))

    return w_avg

