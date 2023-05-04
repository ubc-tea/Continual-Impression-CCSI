#!/usr/bin/env python
# coding=utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision
from torchvision import datasets, models, transforms
from torch.autograd import Variable
import numpy as np
import time
import os
import copy
import argparse
from PIL import Image
from scipy.spatial.distance import cdist
from sklearn.metrics import confusion_matrix
from utils_pytorch import *

def compute_confusion_matrix(tg_model,evalloader, device=None):
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    tg_model.eval()

    #evalset = torchvision.datasets.CIFAR100(root='./data', train=False,
    #                                   download=False, transform=transform_test)
    #evalset.test_data = input_data.astype('uint8')
    #evalset.test_labels = input_labels
    #evalloader = torch.CCSI_utils.data.DataLoader(evalset, batch_size=128,
    #    shuffle=False, num_workers=2)

    correct = 0
    total = 0
    num_classes = tg_model.fc.out_features
    all_targets = []
    all_predicted = []
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(eval_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            total += targets.size(0)
            all_targets.append(targets.cpu())

            outputs = tg_model(inputs)
            _, predicted = outputs.max(1)
            correct += predicted.eq(targets).sum().item()
            all_predicted.append(predicted.cpu())

            # print(sqd_icarl.shape, score_icarl.shape, predicted_icarl.shape, \
                  # sqd_ncm.shape, score_ncm.shape, predicted_ncm.shape)

    return confusion_matrix(np.concatenate(all_targets), np.concatenate(all_predicted))
