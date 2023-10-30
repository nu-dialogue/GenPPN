#!/bin/bash -x
#PJM -L rscgrp=cx-small
#PJM -L node=4
#PJM -L elapse=2:00:00
#PJM -j
#PJM -S
#PJM -o test_rl-sys_template-mean-150.log

module load gcc/8.4.0
module load cuda/12.0.1
module load cudnn/8.8.0
module load openmpi_cuda/4.1.5
module load nccl/2.17.1

. .venv/bin/activate

export CUDA_VISIBLE_DEVICES="0,1,2,3"

ADAPTER_DPATH="outputs/rl/sys_template-mean/checkpoints/iteration-150"

mpirun -n 16 -machinefile $PJM_O_NODEINF -display-devel-map -map-by ppr:2:socket \
    python run_rl_alpaca_lora.py \
        --dist_type mpi \
        --run_dpath $ADAPTER_DPATH \
        --system_nlg_name template_nlg \
        --base_model_name ohashi56225/alpaca-7b \
        --adapter_checkpoint_path $ADAPTER_DPATH \
        --do_test
