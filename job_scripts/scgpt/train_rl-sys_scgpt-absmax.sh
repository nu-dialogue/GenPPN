#!/bin/bash -x
#PJM -L rscgrp=cx-small
#PJM -L node=4
#PJM -L elapse=12:00:00
#PJM -j
#PJM -S
#PJM -o train_rl-sys_scgpt-absmax.log

module load gcc/8.4.0
module load cuda/12.0.1
module load cudnn/8.8.0
module load openmpi_cuda/4.1.5
module load nccl/2.17.1

. .venv/bin/activate

export CUDA_VISIBLE_DEVICES="0,1,2,3"

RUN_DPATH="outputs/rl/sys_scgpt-absmax"

mpirun -n 16 -machinefile $PJM_O_NODEINF -display-devel-map -map-by ppr:2:socket \
    python run_rl_alpaca_lora.py \
        --dist_type mpi \
        --run_dpath $RUN_DPATH \
        --system_nlg_name scgpt_nlg \
        --base_model_name ohashi56225/alpaca-7b \
        --wandb_project_name genppn-emnlp2023 \
        --do_train \
        --total_iterations 200 \
        --batch_size_per_device 32 \
        --mini_batch_size_per_device 2 \
        --gradient_accumulation_steps 2 \
        --num_epochs 4 \
        --learning_rate 1e-5 \
        --reward_factors da_contribution_absmax \
        --adaptive_dac \
        --dac_dialogue_dpath_to_init "outputs/dialogue_data/sys_scgpt/dialogues" \
        --dac_num_dialogues_to_init 1000
