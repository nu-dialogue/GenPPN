import os
import math
from glob import glob
import json
import numpy as np
import warnings
import pandas as pd
from typing import Tuple, Any, Optional, List
from dataclasses import dataclass, field
from logging import getLogger
import torch.distributed as dist
import torch

from transformers import (
    HfArgumentParser,
    set_seed
)

from utils import (
    set_logger,
    sample_dialogues,
    DistributedEnvType,
    set_default_env,
    set_mpi_env,
    TEST_GOAL_SEEDS
)

from arguments import (
    GeneralArguments,
    DialogueSamplingArguments,
    save_args
)

from system import SystemAgent
from user_simulator  import UserAgent

logger = getLogger(__name__)
set_logger(logger)

@dataclass
class RunningArguments:
    dist_type: str = field(
        default=DistributedEnvType.DEFAULT.value,
        metadata={
            "help": "distributed environment type"
        }
    )
    turns_per_process: Optional[int] = field(
        default=None,
        metadata={
            "help": "number of turns to be sampled per process"
        }
    )
    do_test: bool = field(
        default=False,
        metadata={
            "help": (
                "whether to run as test. If True, goal seeds are set to: np.arange(1024)"
            )
        }
    )

    def __post_init__(self):
        if not self.do_test:
            assert self.turns_per_process is not None, (
                "turns_per_process must be specified when do_test=False"
            )

def main():
    parser = HfArgumentParser((GeneralArguments, DialogueSamplingArguments, RunningArguments))
    general_args, ds_args, running_args = parser.parse_args_into_dataclasses()

    # Setup distributed training
    if running_args.dist_type == DistributedEnvType.DEFAULT.value:
        world_size, world_rank, local_rank = set_default_env()
    elif running_args.dist_type == DistributedEnvType.MPI.value:
        world_size, world_rank, local_rank = set_mpi_env()
    else:
        raise ValueError(f"Unknown distributed environment type: {running_args.dist_type}")
    
    running_args.ddp = world_size != 1
    if running_args.ddp:
        logger.info(f"Using DistributedDataParallel: world_size: {world_size}, world_rank: {world_rank}, local_rank: {local_rank}")
        running_args.world_size = world_size
    else:
        logger.info("Not using DistributedDataParallel")

    # Initialize distributed training
    if running_args.ddp:
        dist.init_process_group(backend="nccl",
                                world_size=running_args.world_size,
                                rank=world_rank)
    
    # Save arguments
    if world_rank == 0:
        args = {"general_args": general_args, "dialogue_sampling_args": ds_args, "running_args": running_args}
        save_args(args, os.path.join(general_args.run_dpath, "args.json"))

    # Load system and user agents
    sys_agent = SystemAgent(nlu_name=ds_args.system_nlu_name,
                            nlu_config_file=ds_args.system_nlu_config_file,
                            dst_name=ds_args.system_dst_name,
                            policy_name=ds_args.system_policy_name,
                            nlg_name=ds_args.system_nlg_name)
    
    user_agent = UserAgent(nlu_name=ds_args.user_nlu_name,
                           nlu_config_file=ds_args.user_nlu_config_file,
                           policy_name=ds_args.user_policy_name,
                           nlg_name=ds_args.user_nlg_name,
                           max_turn=ds_args.max_turns_per_dialogue,
                           max_initiative=ds_args.user_max_initiative)
    
    # Create directory to save sampled dialogues
    dialogues_dpath = os.path.join(general_args.run_dpath, "dialogues")
    if world_rank == 0:
        if not os.path.exists(dialogues_dpath):
            os.makedirs(dialogues_dpath)

    if running_args.ddp:
        dist.barrier()
    
    # Set different seed on each process for different dialogue sampling
    set_seed(general_args.random_seed+world_rank)
    
    # 1. Sample dialogues
    # Make goal seeds for each process
    if running_args.do_test:
        examples_per_process = None
        goal_seeds = np.reshape(TEST_GOAL_SEEDS, [world_size, -1])[world_rank].tolist()
    else:
        examples_per_process = running_args.turns_per_process
        goal_seeds = None
    logger.info(f"Sampling dialogues for testing: world_rank: {world_rank}, goal_seeds: {goal_seeds}")
    result = sample_dialogues(iteration_id=0, sys_agent=sys_agent, user_agent=user_agent,
                              log_dpath=dialogues_dpath, examples_per_process=examples_per_process,
                              max_turns_per_dialogue=ds_args.max_turns_per_dialogue, goal_seeds=goal_seeds,
                              process_id=world_rank, validate_training_examples=False, drop_success_dialogue=False)
    
    # 2. Gather results
    if running_args.ddp:
        dist.barrier()

    if world_rank == 0:
        sampled_dialogues = []
        for dialogue_fpath in glob(os.path.join(dialogues_dpath, "*.json")):
            sampled_dialogues.append(json.load(open(dialogue_fpath, "r")))
        df = pd.DataFrame(sampled_dialogues)
        mean_scores = df.mean(numeric_only=True)
        log_summary = {
            "sampled_dialogues": len(df),
            "scores": mean_scores.to_dict(),
        }
        logger.info(f"Summary:\n{log_summary}")
        json.dump(log_summary, open(os.path.join(general_args.run_dpath, "eval_summary.json"), "w"), indent=4)

if __name__ == "__main__":
    # Run the main function
    main()