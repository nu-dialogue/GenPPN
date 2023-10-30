from typing import List, Dict, Tuple, Optional, Union
import pandas as pd
import json
import os
from logging import getLogger
import numpy as np
from glob import glob
from tqdm import tqdm
import torch.distributed as dist
from utils.evaluator import calc_da_accuracy
from utils.log import set_logger

logger = getLogger(__name__)
set_logger(logger)

class DialogueActContributionModel:
    def __init__(self, process_id: int, dpath: str, dialogue_dpath_to_init: Optional[str], num_dialogues_to_init: Optional[int],
                 adaptive: bool, include_value: bool = False) -> None:
        self.process_id = process_id
        self.dpath = dpath
        if self.process_id == 0:
            os.makedirs(dpath, exist_ok=True)
        
        self.adaptive = adaptive
        if not self.adaptive:
            assert dialogue_dpath_to_init is not None, \
                "dialogue_dpath_to_init must be specified if adaptive is False"
            
        self.include_value = include_value
        self.unknown_da = "unknown"

        self.global_da_counts = self._initialize_da_counts(set_unknown=True)
        if dialogue_dpath_to_init:
            assert num_dialogues_to_init is not None, \
                "dialogues_to_initialize must be specified if dialogue_dpath_to_init_da_counts is specified"
            logger.info(f"Initializing dialogue act counts from {dialogue_dpath_to_init}")
            
            for dialogue_fpath in tqdm(glob(os.path.join(dialogue_dpath_to_init, "*.json"))[:num_dialogues_to_init]):
                dialogue_log = json.load(open(dialogue_fpath))
                da_counts = self.count_da(dialogue_log)
                self.global_da_counts = self.global_da_counts.add(da_counts, fill_value=0)
            if self.process_id == 0:
                fpath = os.path.join(self.dpath, f"global_da_counts-initial.feather")
                logger.info(f"Saving global dialogue act counts to {fpath}")
                self.global_da_counts.reset_index().to_feather(fpath)

    def _initialize_da_counts(self, set_unknown=False) -> pd.DataFrame:
        da_counts = pd.DataFrame(columns=["flat_da",
                                          "recognized/success",
                                          "recognized/fail",
                                          "misrecognized/success",
                                          "misrecognized/fail"],
                                ).set_index("flat_da")
        if set_unknown:
            da_counts.loc[self.unknown_da] = [1, 1, 1, 1]
        return da_counts
    
    def update(self, iteration_id: int, world_size: int, dialogues:List[List[Dict[str, any]]]) -> None:
        """
        Update the local dialogue act counts with the given dialogue log.
        """
        assert self.adaptive, "This method can only be called when adaptive is True"

        # DA counts of this process
        logger.info(f"[Process {self.process_id}] Counting local dialogue acts")
        local_da_counts = self._initialize_da_counts()
        for dialogue_log in dialogues:
            da_counts = self.count_da(dialogue_log)
            local_da_counts = local_da_counts.add(da_counts, fill_value=0)
        
        # Save local da counts
        fpath = os.path.join(self.dpath, f"local_da_counts-{iteration_id}-{self.process_id}.feather")
        logger.info(f"[Process {self.process_id}] Saving local dialogue act counts to {fpath}")
        local_da_counts.reset_index().to_feather(fpath)
        del local_da_counts

        # Gather local da counts from all processes
        if world_size > 1:
            dist.barrier()
        logger.info(f"[Process {self.process_id}] Loading dialogue act counts from all processes")
        for process_id in range(world_size):
            fpath = os.path.join(self.dpath, f"local_da_counts-{iteration_id}-{process_id}.feather")
            local_da_counts = pd.read_feather(fpath).set_index("flat_da")
            self.global_da_counts = self.global_da_counts.add(local_da_counts, fill_value=0)

    def _flatten_da(self, da: List[Tuple[str, str, str, str]]) -> List[str]:
        flat_da = []
        for intent, domain, slot, value in da:
            if self.include_value:
                flat_da.append(f"{intent}-{domain}-{slot}-{value}")
            else:
                flat_da.append(f"{intent}-{domain}-{slot}")
        return flat_da

    def count_da(self, dialogue_log: List[Dict[str, any]]) -> None:
        """
        Count dialogue acts in a dialogue log
        """
        task_success = dialogue_log["task_success"]

        recognized_key = "recognized/" + ("success" if task_success else "fail")
        misrecognized_key = "misrecognized/" + ("success" if task_success else "fail")

        turn_evals_of_da_accuracy = dialogue_log["turn_evals_of_da_accuracy"]

        recognized_da = []
        misrecognized_da = []
        for turn_eval in turn_evals_of_da_accuracy[:-1]:
            recognized_da += self._flatten_da(turn_eval["system_da"]["tp_da"])
            misrecognized_da += self._flatten_da(turn_eval["system_da"]["fn_da"])

        recognized_da = pd.Series(recognized_da, dtype=str).value_counts()
        recognized_da.name = recognized_key
        misrecognized_da = pd.Series(misrecognized_da, dtype=str).value_counts()
        misrecognized_da.name = misrecognized_key

        da_counts = pd.concat([recognized_da, misrecognized_da], axis=1).fillna(0)
        da_counts.index.name = "flat_da"

        # Fill zero misrecognized da if it was also recognized
        da_counts.loc[da_counts[recognized_key] > 0, misrecognized_key] = 0
        
        return da_counts

    def get_contribution(self, da: List[Tuple[str, str, str, str]]) -> np.ndarray:
        """
        Get the contribution of the given dialogue act.
        """
        flat_da = []
        for flat_da_ in self._flatten_da(da):
            if flat_da_ in self.global_da_counts.index:
                flat_da.append(flat_da_)
            else:
                logger.warning(f"DA {flat_da_} not found in global DA counts")
                flat_da.append(self.unknown_da)
        
        da_counts = self.global_da_counts.loc[flat_da]
        contribution = da_counts[["recognized/success", "misrecognized/fail"]].sum(1) / da_counts.sum(1)

        return contribution.to_numpy(dtype=float)
    
    def get_data(self) -> pd.DataFrame:
        """
        Get the matrix of dialogue act counts.
        """
        global_da_counts = self.global_da_counts.copy()
        global_da_counts["contribution"] = global_da_counts[["recognized/success", "misrecognized/fail"]].sum(1) / global_da_counts.sum(1)
        return global_da_counts