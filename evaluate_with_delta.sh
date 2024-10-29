#!/bin/bash

F_SOLVER='km_solver'

F_STOP_MODE='rel'
F_ALPHA=0.9
EVAL_F_THRES=60
CHECKPOINT='deq-flow-H-things-test-3x'
DELTA_THRESHOLD=0.005

python -u main.py --eval --delta --delta_analyse --delta_threshold $DELTA_THRESHOLD --name deq-flow-H-all-grad \
    --validation kitti sintel --restore_ckpt ./checkpoints/v2/$CHECKPOINT.pth --gpus 0 \
    --wnorm --eval_f_thres $EVAL_F_THRES --f_solver $F_SOLVER --f_alpha $F_ALPHA --f_stop_mode $F_STOP_MODE \
    --huge --fixed_point_reuse --warm_start \
    --output_path ./result_out 

python calc_activation_sparsity_flops.py ./result_out