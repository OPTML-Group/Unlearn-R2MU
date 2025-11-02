#!/bin/bash




CUDA_VISIBLE_DEVICES=0,1,2,3,4 torchrun --nproc_per_node=5 evaluate_claude_save.py --mode Reason_think --datasets wmdp --model_choice RMU_unlearn_test_11_2_2025 --wmdp_subject wmdp-bio --batch_size 4 --max_samples 10


