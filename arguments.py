import json
from copy import deepcopy
import os
from typing import Tuple, Any, Optional, List
from dataclasses import dataclass, field
import torch

class CustomJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        # if not isinstance(obj, (int, float, str, list, dict, tuple, bool, type(None))):
        #     breakpoint()
        if isinstance(obj, torch.device):
            return {'_type': 'datetime', 'value': str(obj)}
        return super().default(obj)

def save_args(args_dict, fpath):
    os.makedirs(os.path.dirname(fpath), exist_ok=True)
    with open(fpath, "w") as f:
        json.dump({key: args.__dict__ for key, args in args_dict.items()},
                  f, indent=4, cls=CustomJSONEncoder)

@dataclass
class GeneralArguments:
    run_dpath: str = field(
        metadata={
            "help": "Run directory path. Save all the outputs in this directory"
        }
    )
    random_seed: int = field(
        default=42,
        metadata={
            "help": "random seed"
        }
    )

    def __post_init__(self):
        self.run_id = os.path.basename(self.run_dpath)

        # if os.path.exists(self.run_dpath):
        #     raise ValueError("Run directory already exists: {}".format(self.run_dpath))
        os.makedirs(self.run_dpath, exist_ok=True)

@dataclass
class DialogueSamplingArguments:
    system_nlu_name: str = field(
        default="bert_nlu",
        metadata={
            "choices": ["bert_nlu", "bert_nlu_ctx"],
            "help": "name of NLU module"
        }
    )
    system_nlu_config_file: str = field(
        default="full-usr.json",
        metadata={
            "help": "NLU config file for system"
        }
    )
    system_dst_name: str = field(
        default="rule_dst",
        metadata={
            "choices": ["rule_dst"],
            "help": "name of DST module"
        }
    )
    system_policy_name: str = field(
        default="rule_policy",
        metadata={
            "choices": ["rule_policy", "mle_policy"],
            "help": "name of Policy module"
        }
    )
    system_nlg_name: str = field(
        default="sclstm_nlg",
        metadata={
            "choices": ["template_nlg", "retrieval_nlg", "sclstm_nlg", "scgpt_nlg", "gpt2rl_nlg"],
            "help": "name of NLG module"
        }
    )

    user_nlu_name: str = field(
        default="bert_nlu",
        metadata={
            "choices": ["bert_nlu", "bert_nlu_ctx"],
            "help": "name of NLU module"
        }
    )
    user_nlu_config_file: str = field(
        default="full-sys.json",
        metadata={
            "help": "NLU config file for user"
        }
    )
    user_policy_name: str = field(
        default="rule_policy",
        metadata={
            "choices": ["rule_policy"],
            "help": "name of Policy module"
        }
    )
    user_nlg_name: str = field(
        default="template_nlg",
        metadata={
            "choices": ["template_nlg", "retrieval_nlg"],
            "help": "name of NLG module"
        }
    )
    user_max_initiative: int = field(
        default=4,
        metadata={
            "help": "maximum number of slots user can mention in a turn"
        }
    )
    max_turns_per_dialogue: int = field(
        default=20,
        metadata={
            "help": "maximum number of timesteps per episode"
        }
    )
