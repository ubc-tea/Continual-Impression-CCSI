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

def compute_accuracy(tg_model, tg_feature_model, class_means, evalloader, scale=None, print_info=True, device=None):
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    tg_model.eval()
    tg_feature_model.eval()

    #evalset = torchvision.datasets.CIFAR100(root='./data', train=False,
    #                                   download=False, transform=transform_test)
    #evalset.test_data = input_data.astype('uint8')
    #evalset.test_labels = input_labels
    #evalloader = torch.CCSI_utils.data.DataLoader(evalset, batch_size=128,
    #    shuffle=False, num_workers=2)

    correct = 0
    correct_icarl = 0
    correct_ncm = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(evalloader):
            inputs, targets = inputs.to(device), targets.to(device)
            total += targets.size(0)
            # print(inputs.shape)
            # print(batch_idx,targets)
            outputs = tg_model(inputs)
            # print(outputs.shape)
            outputs = F.softmax(outputs, dim=1)
            if scale is not None:
                assert(scale.shape[0] == 1)
                assert(outputs.shape[1] == scale.shape[1])
                outputs = outputs / scale.repeat(outputs.shape[0], 1).type(torch.FloatTensor).to(device)
            _, predicted = outputs.max(1)
            correct += predicted.eq(targets).sum().item()

            outputs_feature = np.squeeze(tg_feature_model(inputs))
            if class_means != None:
                # Compute score for iCaRL
                sqd_icarl = cdist(class_means[:,:,0].T, outputs_feature.cpu(), 'sqeuclidean')
                score_icarl = torch.from_numpy((-sqd_icarl).T).to(device)
                _, predicted_icarl = score_icarl.max(1)
                correct_icarl += predicted_icarl.eq(targets).sum().item()
                # Compute score for NCM
                sqd_ncm = cdist(class_means[:,:,1].T, outputs_feature.cpu(), 'sqeuclidean')
                score_ncm = torch.from_numpy((-sqd_ncm).T).to(device)
                _, predicted_ncm = score_ncm.max(1)
                correct_ncm += predicted_ncm.eq(targets).sum().item()
                # print(sqd_icarl.shape, score_icarl.shape, predicted_icarl.shape, \
                      # sqd_ncm.shape, score_ncm.shape, predicted_ncm.shape)
    if print_info:
        print("  top 1 accuracy CNN            :\t\t{:.2f} %".format(100.*correct/total))
        if class_means != None:
            print("  top 1 accuracy iCaRL          :\t\t{:.2f} %".format(100.*correct_icarl/total))
            print("  top 1 accuracy NCM            :\t\t{:.2f} %".format(100.*correct_ncm/total))

    cnn_acc = 100.*correct/total
    if class_means != None:
        icarl_acc = 100.*correct_icarl/total
        ncm_acc = 100.*correct_ncm/total
    else: 
        icarl_acc, ncm_acc = 0,0
    return [cnn_acc, icarl_acc, ncm_acc]



# def calculate_metrics(self, outputs, targets,device):
#     """Contains the main Task-Aware and Task-Agnostic metrics"""
#     pred = torch.zeros_like(targets.to(device))
#     # Task-Aware Multi-Head
#     for m in range(len(pred)):
#         pred[m] = outputs[0][m].argmax()
#     hits_taw = (pred == targets.to(device)).float()
#     # Task-Agnostic Multi-Head
#     # if self.multi_softmax:
#     #     outputs = [torch.nn.functional.log_softmax(output, dim=1) for output in outputs]
#     #     pred = torch.cat(outputs, dim=1).argmax(1)
#     # else:
#     #     pred = torch.cat(outputs, dim=1).argmax(1)
#     # hits_tag = (pred == targets.to(self.device)).float()
#     return hits_taw

# def compute_accuracy(tg_model, tg_feature_model, class_means, evalloader, scale=None, print_info=True, device=None):
#     print("I'm evaluating here")
#     if device is None:
#         device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#     counter = 0
#     i_s = True
#     with torch.no_grad():
#         total_loss, total_acc_taw, total_acc_tag, total_auc_taw, total_num = 0, 0, 0, 0, 0
#         tg_model.eval()
#         for images, targets in evalloader:
#             counter += 1
#             # Forward old model
#             if i_s:
#                 print('images.shape',images.shape)
#                 print(images[0][0])
#                 i_s = False
#             # Forward current model
#             outputs = tg_model(images.to(device), return_features=False)
#             # print(targets)
#             # during training, the usual accuracy is computed on the outputs
#             # import pdb; pdb.set_trace()
#             # print('self.exemplar_means')
#             # print(self.exemplar_means)
# #                 if not self.exemplar_means:

# #                     print("I'm calculating metrics")
#             hits_tag = calculate_metrics(outputs, targets,device)
#             # else:
#             #     print('Im classifying')
#             #     hits_taw, hits_tag = self.classify(t, feats, targets)
#             # Log
#             total_acc_tag += hits_tag.sum().item()
#             total_num += len(targets)
#         print("hereeeeeeeeeeee:",total_acc_tag / total_num)
#     return total_acc_tag / total_num

# def calculate_metrics(outputs, targets,device):
#     """Contains the main Task-Aware and Task-Agnostic metrics"""
#     print('outputs',outputs)
#     print('targets',targets)
#     pred = torch.zeros_like(targets.to(device))
#     # Task-Aware Multi-Head
    
#     # outputs = [torch.nn.functional.log_softmax(output, dim=1) for output in outputs]
#     # pred = outputs.argmax(1)
#     pred = torch.cat(outputs, dim=1).argmax(1)
#     hits_tag = (pred == targets.to(device)).float()
#     return hits_tag