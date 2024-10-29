import torch
import torch.nn as nn
import torch.nn.functional as F

from gma import Aggregate
from utils.compute_flops import *

# the element-wise delta operation
def delta_op(mem, new, threshold):
    delta = new - mem
    return torch.where(torch.abs(delta) <= threshold, torch.zeros_like(delta), delta)

def delta_op_analyse(mem, new, threshold):
    delta = new - mem

    threshed_delta_mask = torch.abs(delta) <= threshold
    num_zeros = torch.sum(threshed_delta_mask).item()
    num_total_elements = delta.numel()

    return torch.where(threshed_delta_mask, torch.zeros_like(delta), delta),\
          num_zeros, num_total_elements

def analyse_flops(layer_list, input_list, output_list, groups_list, bias_true_list, total_flops, total_theo_min_flops):
    normal_flops = analyse_normal_conv_flops(layer_list, [inp.size() for inp in input_list], 
                        [out.size() for out in output_list], groups_list, bias_true_list)

    max_sparse_flops = analyse_max_act_sparse_flops(layer_list, input_list, [out.size() for out in output_list], groups_list, bias_true_list)

    return total_flops + normal_flops, total_theo_min_flops + max_sparse_flops


def analyse_normal_conv_flops(layer_list, input_list, output_size_list, groups_list, bias_true_list):
    assert len(layer_list) == len(input_list) == len(output_size_list) == len(groups_list) == len(bias_true_list)
    full_flops = 0
    for i in range(len(layer_list)):
        full_flops += conv2d_flops(input_list[i], output_size_list[i], layer_list[i].kernel_size, groups_list[i], bias_true_list[i])
    
    return full_flops


def analyse_max_act_sparse_flops(layer_list, input_list, output_list, groups_list, bias_true_list):
    assert len(layer_list) == len(input_list) == len(output_list) == len(groups_list) == len(bias_true_list)
    full_flops = 0
    for i in range(len(layer_list)):
        full_flops += act_sparse_conv2d_theory_flops(input_list[i], output_list[i], layer_list[i].kernel_size, groups_list[i], bias_true_list[i])
        # add delta diff operation FLOPs
        full_flops += delta_diff_flops(input_list[i].size())
    return full_flops

class FlowHead(nn.Module):
    def __init__(self, input_dim=128, hidden_dim=256):
        super(FlowHead, self).__init__()
        self.conv1 = nn.Conv2d(input_dim, hidden_dim, 3, padding=1)
        self.conv2 = nn.Conv2d(hidden_dim, 2, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        return self.conv2(self.relu(self.conv1(x)))

    def reset_delta_memory(self):
        return
    
# DeltaFlowHead is inherited from FlowHead, but with a delta threshold and overwrite the forward method
class DeltaFlowHead(FlowHead):
    def __init__(self, input_dim=128, hidden_dim=256, delta_thresh=0.0):
        super(DeltaFlowHead, self).__init__(input_dim, hidden_dim)
        self.delta_thresh = delta_thresh
        self.relu = nn.ReLU(inplace=False)  # otherwise the self.mem will be modified!
        self.reset_delta_memory()

        print(f"Initialized Delta FlowHead with delta threshold: {self.delta_thresh}")
    
    # WARNING: this function is very important, one needs to reset the delta memory
    # when needed such as: new sequence, diverging etc. it should be called in the forward pass
    # with corresponding criteria
    def reset_delta_memory(self):
        self.init_memory = True
        self.mem_x = None
        self.mem_conv1_x = None
    
    def forward(self, x):
        if self.init_memory:
            self.mem_x = x
            self.mem_conv1_x = self.conv1(x)
            self.init_memory = False
        else:
            thred_delta_x = delta_op(self.mem_x, x, self.delta_thresh)
            self.mem_conv1_x = self.conv1(thred_delta_x) + self.mem_conv1_x - self.conv1.bias.unsqueeze(0).unsqueeze(2).unsqueeze(3)
            self.mem_x = x
            
        x = self.relu(self.mem_conv1_x)
        # conv2 has only 2 output kernels, the computational saving is very limited, so don't use delta
        x = self.conv2(x)
        return x
    
    def forward_delta_analyse(self, x):
        total_flops, total_theo_min_flops, non_skippable_flops = 0, 0, 0
        total_zeros_cross_layers = 0
        total_activations_numel = 0
        if self.init_memory:
            self.mem_x = x
            self.mem_conv1_x = self.conv1(x)
            total_zeros_cross_layers += torch.sum(x == 0).item()
            #we also take the sparsity in the input into account
            total_activations_numel += x.numel()
            self.init_memory = False

            # also calculate the theoretical FLOPs
            total_flops, total_theo_min_flops = analyse_flops([self.conv1], [x], [self.mem_conv1_x], [self.conv1.groups], [True],
                                                              total_flops, total_theo_min_flops)

        else:
            thred_delta_x, zeros, numel = delta_op_analyse(self.mem_x, x, self.delta_thresh)
            total_zeros_cross_layers += zeros
            total_activations_numel += numel
            self.mem_conv1_x = self.conv1(thred_delta_x) + self.mem_conv1_x - self.conv1.bias.unsqueeze(0).unsqueeze(2).unsqueeze(3)
            self.mem_x = x

            # also calculate the theoretical FLOPs
            total_flops, total_theo_min_flops = analyse_flops([self.conv1], [thred_delta_x], [self.mem_conv1_x], [self.conv1.groups], [True],
                                                              total_flops, total_theo_min_flops)

            
        x = self.relu(self.mem_conv1_x)
        non_skippable_flops += relu_flops(list(self.mem_conv1_x.size()))
        # conv2 has only 2 output kernels, the computational saving is very limited, so don't use delta
        x_out = self.conv2(x)
        total_flops, total_theo_min_flops = analyse_flops([self.conv2], [x], [x_out], [self.conv2.groups], 
                                                          [True], total_flops, total_theo_min_flops)

        total_flops += non_skippable_flops
        total_theo_min_flops += non_skippable_flops
        
        return x_out, {"total_zeros_cross_layers":total_zeros_cross_layers, "total_activation_numel":total_activations_numel,
                       "total_flops":total_flops, "total_theo_min_flops":total_theo_min_flops}

        
class SepConvGRU(nn.Module):
    def __init__(self, hidden_dim=128, input_dim=192+128):
        super(SepConvGRU, self).__init__()
        self.convz1 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (1,5), padding=(0,2))
        self.convr1 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (1,5), padding=(0,2))
        self.convq1 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (1,5), padding=(0,2))

        self.convz2 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (5,1), padding=(2,0))
        self.convr2 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (5,1), padding=(2,0))
        self.convq2 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (5,1), padding=(2,0))

    def forward(self, h, x):
        # horizontal
        hx = torch.cat([h, x], dim=1)
        z = torch.sigmoid(self.convz1(hx))
        r = torch.sigmoid(self.convr1(hx))
        q = torch.tanh(self.convq1(torch.cat([r*h, x], dim=1)))        
        h = (1-z) * h + z * q

        # vertical
        hx = torch.cat([h, x], dim=1)
        z = torch.sigmoid(self.convz2(hx))
        r = torch.sigmoid(self.convr2(hx))
        q = torch.tanh(self.convq2(torch.cat([r*h, x], dim=1)))       
        h = (1-z) * h + z * q

        return h
    
    def reset_delta_memory(self):
        return

# inherited from MotionEncoder, but with a delta threshold and overwrite the forward method
class DeltaSepConvGRU(SepConvGRU):
    def __init__(self, hidden_dim, input_dim, delta_thresh=0.0):
        super(DeltaSepConvGRU, self).__init__(hidden_dim, input_dim)
        self.delta_thresh = delta_thresh
        self.reset_delta_memory()

        print(f"DeltaSepConvGRU with delta threshold: {self.delta_thresh}")

    # WARNING: this function is very important, one needs to reset the delta memory
    # when needed such as: new sequence, diverging etc. it should be called in the forward pass
    # with corresponding criteria
    def reset_delta_memory(self):
        self.init_memory = True
        self.mem_hx = None
        self.mem_rhx = None
        self.mem_hx2 = None
        self.mem_rhx2 = None
        self.mem_convz1_hx = None
        self.mem_convr1_hx = None
        self.mem_convz2_hx = None
        self.mem_convz1_hx = None


    def forward(self, h, x):
        # horizontal
        hx = torch.cat([h, x], dim=1)
        if self.init_memory:
            self.mem_hx = hx
            self.mem_convz1_hx = self.convz1(hx)
            self.mem_convr1_hx = self.convr1(hx)
        else:
            thred_delta_hx = delta_op(self.mem_hx, hx, self.delta_thresh)
            self.mem_convz1_hx = self.convz1(thred_delta_hx) + self.mem_convz1_hx - self.convz1.bias.unsqueeze(0).unsqueeze(2).unsqueeze(3)     
            thred_delta_hx = delta_op(self.mem_hx, hx, self.delta_thresh)
            self.mem_convr1_hx = self.convr1(thred_delta_hx) + self.mem_convr1_hx - self.convr1.bias.unsqueeze(0).unsqueeze(2).unsqueeze(3)
            self.mem_hx = hx

        z = torch.sigmoid(self.mem_convz1_hx)
        r = torch.sigmoid(self.mem_convr1_hx)

        if self.init_memory:
            self.mem_rhx = torch.cat([r*h, x], dim=1)
            self.mem_convq1_rhx = self.convq1(self.mem_rhx)
        else:
            rhx = torch.cat([r*h, x], dim=1)
            thred_delta_rhx = delta_op(self.mem_rhx, rhx, self.delta_thresh)
            self.mem_convq1_rhx = self.convq1(thred_delta_rhx) + self.mem_convq1_rhx - self.convq1.bias.unsqueeze(0).unsqueeze(2).unsqueeze(3)
            self.mem_rhx = rhx

        q = torch.tanh(self.mem_convq1_rhx)        
        h = (1-z) * h + z * q

        # vertical
        hx = torch.cat([h, x], dim=1)
        if self.init_memory:
            self.mem_hx2 = hx
            self.mem_convz2_hx = self.convz2(hx)
            self.mem_convr2_hx = self.convr2(hx)
        else:
            thred_delta_hx2 = delta_op(self.mem_hx2, hx, self.delta_thresh)
            self.mem_convz2_hx = self.convz2(thred_delta_hx2) + self.mem_convz2_hx - self.convz2.bias.unsqueeze(0).unsqueeze(2).unsqueeze(3)
            self.mem_convr2_hx = self.convr2(thred_delta_hx2) + self.mem_convr2_hx - self.convr2.bias.unsqueeze(0).unsqueeze(2).unsqueeze(3)
            self.mem_hx2 = hx

        z = torch.sigmoid(self.mem_convz2_hx)
        r = torch.sigmoid(self.mem_convr2_hx)

        if self.init_memory:
            self.mem_rhx2 = torch.cat([r*h, x], dim=1)
            self.mem_convq2_rhx = self.convq2(self.mem_rhx2)
            self.init_memory = False
        else:
            rhx2 = torch.cat([r*h, x], dim=1)
            thred_delta_rhx = delta_op(self.mem_rhx2, rhx2, self.delta_thresh)
            self.mem_convq2_rhx = self.convq2(thred_delta_rhx) + self.mem_convq2_rhx - self.convq2.bias.unsqueeze(0).unsqueeze(2).unsqueeze(3)
            self.mem_rhx2 = rhx2

        q = torch.tanh(self.mem_convq2_rhx)       
        h = (1-z) * h + z * q

        return h

    def forward_delta_analyse(self, h, x):
        total_zeros_cross_layers = 0
        total_activations_numel = 0
        total_flops, total_theo_min_flops, non_skippable_flops = 0, 0, 0
        # horizontal
        hx = torch.cat([h, x], dim=1)
        if self.init_memory:
            self.mem_hx = hx
            self.mem_convz1_hx = self.convz1(hx)
            total_zeros_cross_layers += torch.sum(hx == 0).item()
            total_activations_numel += hx.numel()
            self.mem_convr1_hx = self.convr1(hx)
            # here the same activation map hx got counted twice. but it's ok
            # just need to write clearly in the paper that the number is not simply the number of activations
            # but the number of activations balabala which are involved in the conv operation
            total_zeros_cross_layers += torch.sum(hx == 0).item()
            total_activations_numel += hx.numel()

            # also calculate the theoretical FLOPs
            total_flops, total_theo_min_flops = analyse_flops([self.convz1, self.convr1], [hx, hx],
                            [self.mem_convz1_hx, self.mem_convr1_hx],
                            [self.convz1.groups, self.convr1.groups], [True, True], total_flops, total_theo_min_flops)
        else:
            thred_delta_hx, zeros, numel = delta_op_analyse(self.mem_hx, hx, self.delta_thresh)
            total_zeros_cross_layers += zeros
            total_activations_numel += numel
            self.mem_convz1_hx = self.convz1(thred_delta_hx) + self.mem_convz1_hx - self.convz1.bias.unsqueeze(0).unsqueeze(2).unsqueeze(3)     
            
            thred_delta_hx, zeros, numel = delta_op_analyse(self.mem_hx, hx, self.delta_thresh)
            total_zeros_cross_layers += zeros
            total_activations_numel += numel
            self.mem_convr1_hx = self.convr1(thred_delta_hx) + self.mem_convr1_hx - self.convr1.bias.unsqueeze(0).unsqueeze(2).unsqueeze(3)
            self.mem_hx = hx

            # also calculate the theoretical FLOPs
            total_flops, total_theo_min_flops = analyse_flops([self.convz1, self.convr1], [thred_delta_hx, thred_delta_hx],
                            [self.mem_convz1_hx, self.mem_convr1_hx],
                            [self.convz1.groups, self.convr1.groups], [True, True], total_flops, total_theo_min_flops)

        z = torch.sigmoid(self.mem_convz1_hx)
        r = torch.sigmoid(self.mem_convr1_hx)

        non_skippable_flops += sigmoid_flops(list(self.mem_convz1_hx.size()))
        non_skippable_flops += sigmoid_flops(list(self.mem_convr1_hx.size()))

        if self.init_memory:
            self.mem_rhx = torch.cat([r*h, x], dim=1)
            self.mem_convq1_rhx = self.convq1(self.mem_rhx)
            total_zeros_cross_layers += torch.sum(self.mem_rhx == 0).item()
            total_activations_numel += self.mem_rhx.numel()

            total_flops, total_theo_min_flops = analyse_flops([self.convq1], [self.mem_rhx],
                            [self.mem_convq1_rhx], [self.convq1.groups], [True], total_flops, total_theo_min_flops)
        else:
            rhx = torch.cat([r*h, x], dim=1)
            thred_delta_rhx, zeros, numel = delta_op_analyse(self.mem_rhx, rhx, self.delta_thresh)
            total_zeros_cross_layers += zeros
            total_activations_numel += numel
            self.mem_convq1_rhx = self.convq1(thred_delta_rhx) + self.mem_convq1_rhx - self.convq1.bias.unsqueeze(0).unsqueeze(2).unsqueeze(3)
            self.mem_rhx = rhx

            total_flops, total_theo_min_flops = analyse_flops([self.convq1], [thred_delta_rhx],
                            [self.mem_convq1_rhx], [self.convq1.groups], [True], total_flops, total_theo_min_flops)

        q = torch.tanh(self.mem_convq1_rhx)    
        tanh_flops_count = tanh_flops(list(self.mem_convq1_rhx.size()))
        non_skippable_flops += tanh_flops_count

        h = (1-z) * h + z * q
        non_skippable_flops += l_prod(list(h.size()))*2

        # vertical
        hx = torch.cat([h, x], dim=1)
        if self.init_memory:
            self.mem_hx2 = hx
            self.mem_convz2_hx = self.convz2(hx)
            self.mem_convr2_hx = self.convr2(hx)
            total_zeros_cross_layers += 2*torch.sum(hx == 0).item()
            total_activations_numel += 2*hx.numel()

            # also calculate the theoretical FLOPs
            total_flops, total_theo_min_flops = analyse_flops([self.convz2, self.convr2], [hx, hx],
                            [self.mem_convz2_hx, self.mem_convr2_hx],
                            [self.convz2.groups, self.convr2.groups], [True, True], total_flops, total_theo_min_flops)
        else:
            thred_delta_hx2, zeros, numel = delta_op_analyse(self.mem_hx2, hx, self.delta_thresh)
            self.mem_convz2_hx = self.convz2(thred_delta_hx2) + self.mem_convz2_hx - self.convz2.bias.unsqueeze(0).unsqueeze(2).unsqueeze(3)
            self.mem_convr2_hx = self.convr2(thred_delta_hx2) + self.mem_convr2_hx - self.convr2.bias.unsqueeze(0).unsqueeze(2).unsqueeze(3)
            total_zeros_cross_layers += 2*zeros
            total_activations_numel += 2*numel
            self.mem_hx2 = hx

            # also calculate the theoretical FLOPs
            total_flops, total_theo_min_flops = analyse_flops([self.convz2, self.convr2], [thred_delta_hx2, thred_delta_hx2],
                            [self.mem_convz2_hx, self.mem_convr2_hx],
                            [self.convz2.groups, self.convr2.groups], [True, True], total_flops, total_theo_min_flops)

        z = torch.sigmoid(self.mem_convz2_hx)
        non_skippable_flops += sigmoid_flops(list(self.mem_convz2_hx.size()))

        r = torch.sigmoid(self.mem_convr2_hx)
        non_skippable_flops += sigmoid_flops(list(self.mem_convr2_hx.size()))

        if self.init_memory:
            self.mem_rhx2 = torch.cat([r*h, x], dim=1)
            self.mem_convq2_rhx = self.convq2(self.mem_rhx2)
            total_zeros_cross_layers += torch.sum(self.mem_rhx2 == 0).item()
            total_activations_numel += self.mem_rhx2.numel()
            self.init_memory = False

            total_flops, total_theo_min_flops = analyse_flops([self.convq2], [self.mem_rhx2],
                            [self.mem_convq2_rhx], [self.convq2.groups], [True], total_flops, total_theo_min_flops)
        else:
            rhx2 = torch.cat([r*h, x], dim=1)
            thred_delta_rhx, zeros, numel = delta_op_analyse(self.mem_rhx2, rhx2, self.delta_thresh)
            total_zeros_cross_layers += zeros
            total_activations_numel += numel
            self.mem_convq2_rhx = self.convq2(thred_delta_rhx) + self.mem_convq2_rhx - self.convq2.bias.unsqueeze(0).unsqueeze(2).unsqueeze(3)
            self.mem_rhx2 = rhx2

            total_flops, total_theo_min_flops = analyse_flops([self.convq2], [thred_delta_rhx],
                            [self.mem_convq2_rhx], [self.convq2.groups], [True], total_flops, total_theo_min_flops)

        q = torch.tanh(self.mem_convq2_rhx)   
        tanh_flops_count = tanh_flops(list(self.mem_convq2_rhx.size()))
        # print(f"tanh_flops in GFlops: {tanh_flops_count/1e9:.4f}")
        non_skippable_flops += tanh_flops_count

        h = (1-z) * h + z * q
        non_skippable_flops += l_prod(list(h.size()))*2

        total_flops += non_skippable_flops
        total_theo_min_flops += non_skippable_flops
        return h, {"total_zeros_cross_layers":total_zeros_cross_layers, "total_activation_numel":total_activations_numel,
                       "total_flops":total_flops, "total_theo_min_flops":total_theo_min_flops}

class MotionEncoder(nn.Module):
    def __init__(self, args):
        super(MotionEncoder, self).__init__()
        
        if args.large:
            c_dim_1 = 256 + 128
            c_dim_2 = 192 + 96

            f_dim_1 = 128 + 64
            f_dim_2 = 64 + 32

            cat_dim = 128 + 64
        elif args.huge:
            c_dim_1 = 256 + 256
            c_dim_2 = 192 + 192

            f_dim_1 = 128 + 128
            f_dim_2 = 64 + 64

            cat_dim = 128 + 128
        elif args.gigantic:
            c_dim_1 = 256 + 384
            c_dim_2 = 192 + 288

            f_dim_1 = 128 + 192
            f_dim_2 = 64 + 96

            cat_dim = 128 + 192
        elif args.tiny:
            c_dim_1 = 64
            c_dim_2 = 48

            f_dim_1 = 32
            f_dim_2 = 16

            cat_dim = 32
        else:
            c_dim_1 = 256
            c_dim_2 = 192

            f_dim_1 = 128
            f_dim_2 = 64

            cat_dim = 128

        cor_planes = args.corr_levels * (2*args.corr_radius + 1)**2
        self.convc1 = nn.Conv2d(cor_planes, c_dim_1, 1, padding=0)
        self.convc2 = nn.Conv2d(c_dim_1, c_dim_2, 3, padding=1)
        self.convf1 = nn.Conv2d(2, f_dim_1, 7, padding=3)
        self.convf2 = nn.Conv2d(f_dim_1, f_dim_2, 3, padding=1)
        self.conv = nn.Conv2d(c_dim_2+f_dim_2, cat_dim-2, 3, padding=1)

    def forward(self, flow, corr):
        # print(f"in original MotionEncoder forward pass")
        cor = F.relu(self.convc1(corr))
        cor = F.relu(self.convc2(cor))
        flo = F.relu(self.convf1(flow))
        flo = F.relu(self.convf2(flo))

        cor_flo = torch.cat([cor, flo], dim=1)
        out = F.relu(self.conv(cor_flo))
        return torch.cat([out, flow], dim=1)
    

    def reset_delta_memory(self):
        return 

class DeltaMotionEncoder(MotionEncoder):
    def __init__(self, args, delta_thresh=0.0):
        super(DeltaMotionEncoder, self).__init__(args)
        self.delta_thresh = delta_thresh
        self.reset_delta_memory()

    def reset_delta_memory(self):
        # needed such as: new sequence. it should be called in the forward pass
        self.init_memory = True
        self.mem_corr = None
        self.mem_flow = None
        self.mem_cor1 = None
        self.mem_flo1 = None    
        self.mem_cor_flo = None
        self.mem_conv_cor_flo = None
        self.mem_convc1_corr = None
        self.mem_convc2_cor1 = None
        self.mem_convf1_flow = None
        self.mem_convf2_flo1 = None

    def forward(self, flow, corr):
        if self.init_memory:  # first iteration of the forward pass
            self.mem_corr = corr
            self.mem_flow = flow
            self.mem_convc1_corr = self.convc1(corr)
            self.mem_convf1_flow = self.convf1(flow)
        else:
            thed_delta_corr = delta_op(self.mem_corr, corr, self.delta_thresh)
            thed_delta_flow = delta_op(self.mem_flow, flow, self.delta_thresh)
            
            self.mem_convc1_corr = self.convc1(thed_delta_corr) + self.mem_convc1_corr - self.convc1.bias.unsqueeze(0).unsqueeze(2).unsqueeze(3)
            self.mem_convf1_flow = self.convf1(thed_delta_flow) + self.mem_convf1_flow - self.convf1.bias.unsqueeze(0).unsqueeze(2).unsqueeze(3)    
            # the handling of input is different from the original delta RNN work
            self.mem_corr = corr
            self.mem_flow = flow

        cor1 = F.relu(self.mem_convc1_corr, inplace=False)
        flo1 = F.relu(self.mem_convf1_flow, inplace=False)

        if self.init_memory:
            self.mem_cor1 = cor1
            self.mem_flo1 = flo1
            self.mem_convc2_cor1 = self.convc2(cor1)
            self.mem_convf2_flo1 = self.convf2(flo1)
        else:
            thred_delta_cor1 = delta_op(self.mem_cor1, cor1, self.delta_thresh)
            self.mem_convc2_cor1 = self.convc2(thred_delta_cor1) + self.mem_convc2_cor1 - self.convc2.bias.unsqueeze(0).unsqueeze(2).unsqueeze(3)   
            thred_delta_flo1 = delta_op(self.mem_flo1, flo1, self.delta_thresh)
            self.mem_convf2_flo1 = self.convf2(thred_delta_flo1) + self.mem_convf2_flo1 - self.convf2.bias.unsqueeze(0).unsqueeze(2).unsqueeze(3)   

            self.mem_cor1 = cor1
            self.mem_flo1 = flo1

        cor = F.relu(self.mem_convc2_cor1, inplace=False)
        flo = F.relu(self.mem_convf2_flo1, inplace=False)

        cor_flo = torch.cat([cor, flo], dim=1)

        if self.init_memory:
            self.mem_cor_flo = cor_flo
            self.mem_conv_cor_flo = self.conv(cor_flo)
            self.init_memory = False
        else:
            thred_delta_cor_flo = delta_op(self.mem_cor_flo, cor_flo, self.delta_thresh)
            self.mem_conv_cor_flo = self.conv(thred_delta_cor_flo) + self.mem_conv_cor_flo - self.conv.bias.unsqueeze(0).unsqueeze(2).unsqueeze(3)  
            self.mem_cor_flo = cor_flo

        out = F.relu(self.mem_conv_cor_flo, inplace=False)
        return torch.cat([out, flow], dim=1)


    def forward_delta_analyse(self, flow, corr):
        total_zeros_cross_layers = 0
        total_activations_numel = 0
        total_flops, total_theo_min_flops, non_skippable_flops = 0, 0, 0

        if self.init_memory:  # first iteration of the forward pass
            self.mem_corr = corr
            self.mem_flow = flow
            self.mem_convc1_corr = self.convc1(corr)
            self.mem_convf1_flow = self.convf1(flow)
            # also need to count in the zeros in inputs
            total_zeros_cross_layers += torch.sum(corr == 0).item()
            total_activations_numel += corr.numel()
            total_zeros_cross_layers += torch.sum(flow == 0).item()
            total_activations_numel += flow.numel()
            # also calculate the theoretical FLOPs

            total_flops, total_theo_min_flops = analyse_flops([self.convc1, self.convf1], [corr, flow], 
                            [self.mem_convc1_corr, self.mem_convf1_flow], 
                            [self.convc1.groups, self.convf1.groups], [True, True], total_flops, total_theo_min_flops)

        else:
            thed_delta_corr, zeros, numel = delta_op_analyse(self.mem_corr, corr, self.delta_thresh)
            total_zeros_cross_layers += zeros
            total_activations_numel += numel
            thed_delta_flow, zeros, numel = delta_op_analyse(self.mem_flow, flow, self.delta_thresh)
            total_zeros_cross_layers += zeros
            total_activations_numel += numel

            self.mem_convc1_corr = self.convc1(thed_delta_corr) + self.mem_convc1_corr - self.convc1.bias.unsqueeze(0).unsqueeze(2).unsqueeze(3)
            self.mem_convf1_flow = self.convf1(thed_delta_flow) + self.mem_convf1_flow - self.convf1.bias.unsqueeze(0).unsqueeze(2).unsqueeze(3)    
            # the handling of input is different from the original delta RNN work
            self.mem_corr = corr
            self.mem_flow = flow

            # also calculate the theoretical FLOPs
            total_flops, total_theo_min_flops = analyse_flops([self.convc1, self.convf1], [thed_delta_corr, thed_delta_flow],
                            [self.mem_convc1_corr, self.mem_convf1_flow],
                            [self.convc1.groups, self.convf1.groups], [True, True], total_flops, total_theo_min_flops)

        cor1 = F.relu(self.mem_convc1_corr, inplace=False)
        flo1 = F.relu(self.mem_convf1_flow, inplace=False)

        non_skippable_flops += relu_flops(list(self.mem_convc1_corr.size()))
        non_skippable_flops += relu_flops(list(self.mem_convf1_flow.size()))

        if self.init_memory:
            self.mem_cor1 = cor1
            self.mem_flo1 = flo1
            self.mem_convc2_cor1 = self.convc2(cor1)
            self.mem_convf2_flo1 = self.convf2(flo1)

            # also need to count in the zeros in inputs
            total_zeros_cross_layers += torch.sum(cor1 == 0).item()
            total_activations_numel += cor1.numel()
            total_zeros_cross_layers += torch.sum(flo1 == 0).item()
            total_activations_numel += flo1.numel()

            # also calculate the theoretical FLOPs
            total_flops, total_theo_min_flops = analyse_flops([self.convc2, self.convf2], [cor1, flo1],
                            [self.mem_convc2_cor1, self.mem_convf2_flo1],
                            [self.convc2.groups, self.convf2.groups], [True, True], total_flops, total_theo_min_flops)
            
        else:
            thred_delta_cor1, zeros, numel = delta_op_analyse(self.mem_cor1, cor1, self.delta_thresh)
            total_zeros_cross_layers += zeros
            total_activations_numel += numel            
            self.mem_convc2_cor1 = self.convc2(thred_delta_cor1) + self.mem_convc2_cor1 - self.convc2.bias.unsqueeze(0).unsqueeze(2).unsqueeze(3)   

            thred_delta_flo1, zeros, numel = delta_op_analyse(self.mem_flo1, flo1, self.delta_thresh)
            total_zeros_cross_layers += zeros
            total_activations_numel += numel
            self.mem_convf2_flo1 = self.convf2(thred_delta_flo1) + self.mem_convf2_flo1 - self.convf2.bias.unsqueeze(0).unsqueeze(2).unsqueeze(3)   

            self.mem_cor1 = cor1
            self.mem_flo1 = flo1

            # also calculate the theoretical FLOPs
            total_flops, total_theo_min_flops = analyse_flops([self.convc2, self.convf2], [thred_delta_cor1, thred_delta_flo1],
                            [self.mem_convc2_cor1, self.mem_convf2_flo1],
                            [self.convc2.groups, self.convf2.groups], [True, True], total_flops, total_theo_min_flops)

        cor = F.relu(self.mem_convc2_cor1, inplace=False)
        flo = F.relu(self.mem_convf2_flo1, inplace=False)

        non_skippable_flops += relu_flops(list(self.mem_convc2_cor1.size()))
        non_skippable_flops += relu_flops(list(self.mem_convf2_flo1.size()))

        cor_flo = torch.cat([cor, flo], dim=1)

        if self.init_memory:
            self.mem_cor_flo = cor_flo
            self.mem_conv_cor_flo = self.conv(cor_flo)
            total_zeros_cross_layers += torch.sum(cor_flo == 0).item()
            total_activations_numel += cor_flo.numel()
            self.init_memory = False

            # also calculate the theoretical FLOPs
            total_flops, total_theo_min_flops = analyse_flops([self.conv], [cor_flo], 
                                         [self.mem_conv_cor_flo], [self.conv.groups], [True], 
                                         total_flops, total_theo_min_flops)


        else:
            thred_delta_cor_flo, zeros, numel = delta_op_analyse(self.mem_cor_flo, cor_flo, self.delta_thresh)
            total_zeros_cross_layers += zeros
            total_activations_numel += numel
            self.mem_conv_cor_flo = self.conv(thred_delta_cor_flo) + self.mem_conv_cor_flo - self.conv.bias.unsqueeze(0).unsqueeze(2).unsqueeze(3)  
            self.mem_cor_flo = cor_flo

            # also calculate the theoretical FLOPs
            total_flops, total_theo_min_flops = analyse_flops([self.conv], [thred_delta_cor_flo],
                                            [self.mem_conv_cor_flo], [self.conv.groups], [True],
                                            total_flops, total_theo_min_flops)

        out = F.relu(self.mem_conv_cor_flo, inplace=False)
        non_skippable_flops += relu_flops(list(self.mem_conv_cor_flo.size()))

        total_flops += non_skippable_flops
        total_theo_min_flops += non_skippable_flops
        return torch.cat([out, flow], dim=1), {"total_zeros_cross_layers":total_zeros_cross_layers, "total_activation_numel":total_activations_numel,
                       "total_flops":total_flops, "total_theo_min_flops":total_theo_min_flops}
    

class UpdateBlock(nn.Module):
    def __init__(self, args, hidden_dim=128, input_dim=128):
        super(UpdateBlock, self).__init__()
        self.args = args
    
        if args.tiny:
            cat_dim = 32
        elif args.large:
            cat_dim = 128 + 64
        elif args.huge:
            cat_dim = 128 + 128
        elif args.gigantic:
            cat_dim = 128 + 192
        else:
            cat_dim = 128
        
        if args.old_version:
            flow_head_dim = min(256, 2*cat_dim)
        else:
            flow_head_dim = 2*cat_dim

        
        if args.gma:
            self.gma = Aggregate(dim=cat_dim, dim_head=cat_dim, heads=1)

            gru_in_dim = 2 * cat_dim + hidden_dim
        else:
            self.gma = None

            gru_in_dim = cat_dim + hidden_dim
        
        if args.delta:
            self.delta_thresh = self.args.delta_threshold
            self.encoder = DeltaMotionEncoder(args, self.delta_thresh)
            self.gru = DeltaSepConvGRU(hidden_dim=hidden_dim, input_dim=gru_in_dim, delta_thresh=self.delta_thresh)
            self.flow_head = DeltaFlowHead(hidden_dim, hidden_dim=flow_head_dim, delta_thresh=self.delta_thresh)
        else:
            self.encoder = MotionEncoder(args)
            self.gru = SepConvGRU(hidden_dim=hidden_dim, input_dim=gru_in_dim)
            self.flow_head = FlowHead(hidden_dim, hidden_dim=flow_head_dim)
    
    def reset_memories(self):
        self.encoder.reset_delta_memory()
        self.gru.reset_delta_memory()
        self.flow_head.reset_delta_memory()

    def forward(self, net, inp, corr, flow, attn=None, upsample=True):
        motion_features = self.encoder(flow, corr)
        
        if self.gma:
            motion_features_global = self.gma(attn, motion_features)
            inp = torch.cat([inp, motion_features, motion_features_global], dim=1)
        else:
            inp = torch.cat([inp, motion_features], dim=1)
        
        # inp are the input injections, gru is the recurrent part 
        net = self.gru(net, inp)
        delta_flow = self.flow_head(net)

        return net, delta_flow
    
    def forward_delta_analyse(self, net, inp, corr, flow, attn=None, upsample=True):
        motion_features, stat_delta_encoder = self.encoder.forward_delta_analyse(flow, corr)
        
        if self.gma:
            motion_features_global = self.gma(attn, motion_features)
            inp = torch.cat([inp, motion_features, motion_features_global], dim=1)
        else:
            inp = torch.cat([inp, motion_features], dim=1)
        
        # inp are the input injections, gru is the recurrent part 
        net, stat_delta_gru = self.gru.forward_delta_analyse(net, inp)
        delta_flow, stat_delta_flowhead = self.flow_head.forward_delta_analyse(net)

        delta_stat = {'iter':self.delta_analyse_iter_counter, \
                      'encoder':stat_delta_encoder, \
                      'gru':stat_delta_gru, \
                      'flowhead':stat_delta_flowhead}
        self.encoder_sparsity_dicts.append(delta_stat)
        self.delta_analyse_iter_counter += 1
        return net, delta_flow
    
    def get_delta_sparsity_dicts(self):
        return self.encoder_sparsity_dicts
