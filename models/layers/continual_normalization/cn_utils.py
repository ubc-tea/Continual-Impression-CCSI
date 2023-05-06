from models.layers.continual_normalization.cn import *
import torch


def check_training(module, print_var=False):
    for attr_str in dir(module):
        target_attr = getattr(module, attr_str)
        if type(target_attr) == CN4 or type(target_attr) == CN8 or type(target_attr) == CN16:
            print(target_attr.training)
            if print_var:
                print("Am I saving these values:", target_attr.group_running_mean.shape)
    for name, icm in module.named_children():
        if type(icm) == CN4 or type(icm) == CN8 or type(icm) == CN16:
            print(icm.training)
        check_training(icm)
    return


def load_continual_variables(module, name, device, group_running_mean_list, group_running_var_list, b_size_list):
    for attr_str in dir(module):
        target_attr = getattr(module, attr_str)
        if type(target_attr) == CN4 or type(target_attr) == CN8 or type(target_attr) == CN16:
            group_running_mean, group_running_var, b_size = group_running_mean_list.pop(0), group_running_var_list.pop(
                0), b_size_list.pop(0)
            target_attr.load_group_vars(group_running_mean, group_running_var, b_size, device)

    for name, icm in module.named_children():
        group_running_mean, group_running_var, b_size = group_running_mean_list.pop(0), group_running_var_list.pop(
            0), b_size_list.pop(0)
        if type(icm) == CN4 or type(icm) == CN8 or type(icm) == CN16:
            # print('detected')
            icm.load_group_vars(group_running_mean, group_running_var, b_size, device)
            group_running_mean, group_running_var, b_size = group_running_mean_list.pop(0), group_running_var_list.pop(
                0), b_size_list.pop(0)
        load_continual_variables(icm, name, device, group_running_mean, group_running_var, b_size)
    return


def replace_bn(module, name, gn_size, layer_number=0):
    pre_bn = []
    for attr_str in dir(module):
        target_attr = getattr(module, attr_str)
        if type(target_attr) == torch.nn.BatchNorm2d:
            pre_bn.append(target_attr)
            if gn_size == 4:
                new_bn = CN4(target_attr, layer_number)
            elif gn_size == 8:
                new_bn = CN8(target_attr, layer_number)
            elif gn_size == 16:
                new_bn = CN16(target_attr, layer_number)
            layer_number += 1
            setattr(module, attr_str, new_bn)
    for name, icm in module.named_children():
        if type(icm) == torch.nn.BatchNorm2d:
            pre_bn.append(icm)
            if gn_size == 4:
                new_bn = CN4(icm, layer_number)
            elif gn_size == 8:
                new_bn = CN8(icm, layer_number)
            elif gn_size == 16:
                new_bn = CN16(icm, layer_number)
            layer_number += 1
            setattr(module, name, new_bn)
        b, layer_number = replace_bn(icm, name, gn_size, layer_number)
        pre_bn.append(b)
    return pre_bn, layer_number


def get_continual_variables(module, name):
    group_running_mean_list = []
    group_running_var_list = []
    b_size_list = []

    for attr_str in dir(module):
        target_attr = getattr(module, attr_str)
        if type(target_attr) == CN4 or type(target_attr) == CN8 or type(target_attr) == CN16:
            group_running_mean, group_running_var, b_size = target_attr.get_group_vars()
            group_running_mean_list.append(group_running_mean)
            group_running_var_list.append(group_running_var)
            b_size_list.append(b_size)

    for name, icm in module.named_children():
        if type(icm) == CN4 or type(icm) == CN8 or type(icm) == CN16:
            group_running_mean, group_running_var, b_size = icm.get_group_vars()
            group_running_mean_list.append(group_running_mean)
            group_running_var_list.append(group_running_var)
            b_size_list.append(b_size)
        group_running_mean, group_running_var, b_size = get_continual_variables(icm, name)
        group_running_mean_list.append(group_running_mean)
        group_running_var_list.append(group_running_var)
        b_size_list.append(b_size)
    return group_running_mean_list, group_running_var_list, b_size_list
