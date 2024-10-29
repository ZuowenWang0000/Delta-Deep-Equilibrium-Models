#  The functions are largely copied or based on the code from the THOPS project
#  https://github.com/Lyken17/pytorch-OpCounter/blob/master/thop


import torch
import numpy as np
import warnings

def l_prod(in_list):
    res = 1
    for _ in in_list:
        res *= _
    return res

def l_sum(in_list):
    res = 0
    for _ in in_list:
        res += _
    return res

def conv2d_flops(input_size: list, output_size: list, kernel_size: list, groups: int, bias: bool = False):
    in_c = input_size[1]
    g = groups
    num_ops =  l_prod(output_size) * (in_c // g) * l_prod(kernel_size[2:])
    if bias:
        num_ops += l_prod(output_size)
    
    return num_ops

def act_sparse_conv2d_theory_flops(input: list, output_size: list, kernel_size: list, groups: int, bias: bool = False):
    input_size = list(input.size())
    in_c = input_size[1]
    g = groups
    num_ops =  l_prod(output_size) * (in_c // g) * l_prod(kernel_size[2:])
    if bias:
        num_ops += l_prod(output_size)
    num_ops = num_ops * (input.count_nonzero() / input.numel())
    return num_ops.item()


def delta_diff_flops(input_size: list):
    return l_prod(input_size)


def conv2d_bias_flops(output_size: list):
    return l_prod(output_size)

def relu_flops(input_size: list):
    return l_prod(input_size)

def sigmoid_flops(input_size: list):
    return 3*l_prod(input_size) #we assuming calculating sigmoid is 3 FLOPs

def tanh_flops(input_size: list):
    return 10*l_prod(input_size) #we assuming calculating sigmoid is 10 FLOPs

def counter_matmul(input_size, output_size):
    input_size = np.array(input_size)
    output_size = np.array(output_size)
    return np.prod(input_size) * output_size[-1]