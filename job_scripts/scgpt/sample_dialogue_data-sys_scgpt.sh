#!/bin/bash -x
#PJM -L rscgrp=cx-small
#PJM -L node=2
#PJM -L elapse=2:00:00
#PJM -j
#PJM -S
#PJM -o sample_dialogue_data-sys_scgpt.log

module load gcc/8.4.0
module load cuda/12.0.1
module load cudnn/8.8.0
module load openmpi_cuda/4.1.5
module load nccl/2.17.1

. .venv/bin/activate

export CUDA_VISIBLE_DEVICES="0,1,2,3"

RUN_DPATH="outputs/dialogue_data/sys_scgpt"

mpirun -n 8 -machinefile $PJM_O_NODEINF -display-devel-map -map-by ppr:2:socket \
    python run_baseline_system.py \
        --dist_type mpi \
        --run_dpath $RUN_DPATH \
        --system_nlg_name scgpt_nlg \
        --turns_per_process 2000
