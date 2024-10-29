import numpy as np
import matplotlib.pyplot as plt
import os
import sys


def calculate_overall_sparsity(delta_stats):
    encoder_sp, gru_sp, flowhead_sp = [], [], []
    
    for sample in delta_stats:
        val_id_0_stats = delta_stats[sample]
        num_deq_iters = len(val_id_0_stats)
        # the sparsity data are stored as (num_zero_entries, total_entries)
        for i in range(num_deq_iters):
            encoder_sp.append(val_id_0_stats[i]['encoder'])
            gru_sp.append(val_id_0_stats[i]['gru'])
            flowhead_sp.append(val_id_0_stats[i]['flowhead'])

    encoder_numel = sum([entry['total_activation_numel'] for entry in encoder_sp])
    encoder_zeros = sum([entry['total_zeros_cross_layers'] for entry in encoder_sp])
    gru_numel = sum([entry['total_activation_numel'] for entry in gru_sp])
    gru_zeros = sum([entry['total_zeros_cross_layers'] for entry in gru_sp])
    flowhead_numel = sum([entry['total_activation_numel'] for entry in flowhead_sp])
    flowhead_zeros = sum([entry['total_zeros_cross_layers'] for entry in flowhead_sp])

    numels = encoder_numel + gru_numel + flowhead_numel
    zeros = encoder_zeros + gru_zeros + flowhead_zeros

    return zeros/numels

def calculate_overall_flops(delta_stats):
    encoder_flops, gru_flops, flowhead_flops = [], [], []
    
    num_sample = 0
    for sample in delta_stats:
        num_sample += 1
        val_id_0_stats = delta_stats[sample]
        num_deq_iters = len(val_id_0_stats)

        # the sparsity data are stored as (num_zero_entries, total_entries)
        for i in range(num_deq_iters):
            encoder_flops.append(val_id_0_stats[i]['encoder'])
            gru_flops.append(val_id_0_stats[i]['gru'])
            flowhead_flops.append(val_id_0_stats[i]['flowhead'])

    encoder_norm_flops = sum([entry['total_flops'] for entry in encoder_flops])
    encoder_theo_min_flops = sum([entry['total_theo_min_flops'] for entry in encoder_flops])
    gru_norm_flops = sum([entry['total_flops'] for entry in gru_flops])
    gru_theo_min_flops = sum([entry['total_theo_min_flops'] for entry in gru_flops])
    flowhead_norm_flops = sum([entry['total_flops'] for entry in flowhead_flops])
    flowhead_theo_min_flops = sum([entry['total_theo_min_flops'] for entry in flowhead_flops])

    total_norm_flops = encoder_norm_flops + gru_norm_flops + flowhead_norm_flops
    total_theo_min_flops = encoder_theo_min_flops + gru_theo_min_flops + flowhead_theo_min_flops

    avg_norm_flops_per_sample = total_norm_flops / num_sample
    avg_theo_min_flops_per_sample = total_theo_min_flops / num_sample
    return avg_norm_flops_per_sample, avg_theo_min_flops_per_sample


if __name__ == '__main__':

    # Load the data from the .npy file
    if len(sys.argv) != 2:
        print("Usage: python calc_activation_sparsity_flops.py <result_folder>")
        sys.exit(1)

    result_folder = sys.argv[1]

    for result_file_name in ['sintel_val_results.npy', 'kitti_val_results.npy']:
        print(f"Evaluating {result_file_name}")
        data = np.load(os.path.join(result_folder, result_file_name), allow_pickle=True)

        # convert data as dictionary
        data = data.item() # type: ignore

        if 'sintel' in result_file_name:
            print(f"sintel-clean: {data['clean']}, sintel-final: {data['final']}")
        elif 'kitti' in result_file_name:
            print(f"kitti-epe: {data['kitti-epe']}, kitti-f1: {data['kitti-f1']}")
        else:
            raise ValueError(f"Unknown dataset: {result_file_name}")


        delta_stats = data['delta_stat']

        avg_sparsity = calculate_overall_sparsity(delta_stats)
        print(f"Average Delta Sparsity: {avg_sparsity}")
        avg_norm_flops_per_sample, avg_theo_min_flops_per_sample = calculate_overall_flops(delta_stats)
        print(f"Average Theoretical Minimum FLOPs per pair (with Delta) in Giga: {avg_theo_min_flops_per_sample/1e9}")
        print("=====================================================")