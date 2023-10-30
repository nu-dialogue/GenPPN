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
from tqdm import tqdm
import gc
import torch
import torch.distributed as dist
from torch.optim import Adam

# monkey patch for wandb logging
import wandb
def maybe_compress_history(obj: Any) -> Tuple[Any, bool]:
    """
    Automatically cast to histogram if size is 32 or more (i.e., `>=32`)
    while more than 32 (i.e., `>32`) is cast to histogram in the original implementation.
    """
    # 32 or smaller np obj has been converted to list by `wandb.util.json_friendly()`,
    # so we need to convert it back to np array
    if isinstance(obj, list):
        obj = np.array(obj)

    if np and isinstance(obj, np.ndarray) and obj.size >= 32:
        return wandb.Histogram(obj, num_bins=32).to_json(), True
    else:
        return obj, False
wandb.util.maybe_compress_history = maybe_compress_history


from transformers import (
    # AutoTokenizer,
    LlamaTokenizer,
    AutoModelForCausalLM,
    HfArgumentParser,
    get_constant_schedule_with_warmup,
    set_seed
)

from peft import (
    PeftModel,
    LoraConfig,
    get_peft_model,
    prepare_model_for_int8_training,
)

from trl import (
    AutoModelForCausalLMWithValueHead
)

from ppn import (
    PPNNLG,
    FixedPPOTrainer,
    FixedPPOConfig
)

from utils import (
    set_logger,
    RewardFunction,
    sample_dialogues,
    RewardFactor,
    DialogueActContributionModel,
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
class PPNArguments:
    """
    Arguments for PPN.
    """
    base_model_name: str = field(
        metadata={
            "help": (
                "The checkpoint for base model of LoRA"
            )
        }
    )

    adapter_checkpoint_path: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The checkpoint path of trained LoRA adapter weights"
            )
        }
    )

    # Text generation parameters

    max_context_turns: int = field(
        default=1,
        metadata={"help": "Maximum number of tokens for the dialogue context"}
    )
    use_system_da: bool = field(
        default=True,
        metadata={"help": "Whether to input system dialogue act"}
    )
    input_no_response: bool = field(
        default=False,
        metadata={"help": "Whether not to input response to PPN"}
    )
    use_no_rephrasing_keyword: bool = field(
        default=False,
        metadata={"help": "Whether to use repeat token for PPN repeating the last system utterance."}
    )

    max_generation_tokens: int = field(
        default=128,
        metadata={"help": "Maximum number of new tokens to be generated"}
    )

    num_beams: int = field(
        default=1,
        metadata={"help": "Number of beams for beam search"}
    )

    temperature: float = field(
        default=1.0,
        metadata={"help": "Temperature for generation"}
    )

    top_k: int = field(
        default=0,
        metadata={"help": "Top-k sampling for generation"}
    )

    top_p: float = field(
        default=1.0,
        metadata={"help": "Top-p sampling for generation"}
    )

    do_sample: bool = field(
        default=True,
        metadata={"help": "Whether to sample or not"}
    )

    def __post_init__(self):
        if self.input_no_response:
            assert self.use_system_da, "If input_no_response is True, use_system_da must be True."

@dataclass
class PPOTrainingArguments:
    """
    Arguments for training PPO.
    """
    
    wandb_project_name: Optional[str] = field(
        default=None,
        metadata={"help": "Project id for wandb tracking."}
    )
    wandb_group_name: Optional[str] = field(
        default=None,
        metadata={"help": "Group id for wandb tracking."}
    )

    load_in_8bit: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to load base model in 8bit"
            )
        }
    )
    torch_dtype: str = field(
        default="float16",
        metadata={
            "help": (
                "Torch dtype for training"
            ),
            "choices": ["float32", "float16", "bfloat16"]
        }
    )
    disallow_tf32: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to not allow tf32"
            )
        }
    )

    dist_type: str = field(
        default=DistributedEnvType.DEFAULT.value,
        metadata={"help": "Distributed environment type",
                  "choices": [e.value for e in DistributedEnvType]}
    )

    do_train: bool = field(
        default=False,
        metadata={"help": "Whether to run training."}
    )
    do_test: bool = field(
        default=False,
        metadata={"help": "Whether to run predictions on the test set."}
    )

    total_iterations: int = field(
        default=100,
        metadata={"help": "Number of total iterations."}
    )

    batch_size_per_device: int = field(
        default=32,
        metadata={"help": "Number of examples per iteration."}
    )
    mini_batch_size_per_device: int = field(
        default=2,
        metadata={"help": "Number of forward passes per mini batch."}
    )
    num_epochs: int = field(
        default=4,
        metadata={"help": "Number of optimization epochs for each batch."}
    )
    gradient_accumulation_steps: int = field(
        default=1,
        metadata={"help": "Nmber of gradient accumulation steps"}
    )
    
    learning_rate: float = field(
        default=5e-6,
        metadata={"help": "Learning rate for the optimizer."}
    )
    num_warmup_steps: int = field(
        default=0,
        metadata={"help": "Number of warmup steps for the optimizer."}
    )
    gamma: float = field(
        default=1.0,
        metadata={"help": "Discount factor of GAE."}
    )
    lam: float = field(
        default=0.95,
        metadata={"help": "Lambda factor of GAE."}
    )
    cliprange: float = field(
        default=0.2,
        metadata={"help": "Cliprange for PPO."}
    )
    cliprange_value: float = field(
        default=0.2,
        metadata={"help": "Cliprange for value function."}
    )
    vf_coef: float = field(
        default=0.1,
        metadata={"help": "Coefficient for value function loss."}
    )
    adap_kl_ctrl: bool = field(
        default=False,
        metadata={"help": "Whether to use adaptive KL control."}
    )
    init_kl_coef: float = field(
        default=0.01,
        metadata={"help": "Initial KL coefficient for adaptive KL control."}
    )
    target_kl: float = field(
        default=6.0,
        metadata={"help": "Target KL value for adaptive KL control. Set to None to disable adaptive KL control."}
    )
    horizon: int = field(
        default=10000,
        metadata={"help": "Horizon for adaptive KL control."}
    )
    ent_coef: float = field(
        default=0.0,
        metadata={"help": "Coefficient for entropy loss."}
    )

    drop_success_dialogue: bool = field(
        default=False,
        metadata={"help": "Whether to exclude success dialogues in training examples."}
    )

    save_iterations: int = field(
        default=10,
        metadata={"help": "Number of iterations to save checkpoints."}
    )

    def __post_init__(self):
        if self.do_train and self.do_test:
            raise NotImplementedError(
                "Cannot train and predict at the same runing."
                "After training is finished, run test."
            )
        
        if self.do_train and not self.wandb_project_name:
            raise ValueError("wandb_project_name must be specified for training.")
        
        # Set the total number of steps
        self.steps = self.total_iterations * self.batch_size_per_device

@dataclass
class RewardArguments:
    reward_factors: List[str] = field(
        default_factory=lambda: [RewardFactor.DA_CONTRIBUTION_ABSMAX.value],
        metadata={"help": (
            "Factors to be included in the reward computation. \n"
            f"Possible factors are: [{[e.value for e in RewardFactor]}]\n"
            "Factors can be combined with space, e.g., 'inform_f1 belief_accuracy'."
        )},
    )
    adaptive_dac: bool = field(
        default=True,
        metadata={"help": "Whether to use adaptive DA contribution."}
    )

    dac_dialogue_dpath_to_init: str = field(
        default=None,
        metadata={"help": "Path to the dialogue file to initialize DA contribution."}
    )
    dac_num_dialogues_to_init: int = field(
        default=1000,
        metadata={"help": "Number of dialogues to initialize DA contribution."}
    )

    def __post_init__(self):
        self.use_dac = any(["da_contribution" in factor for factor in self.reward_factors])
    

def load_adapter_weights(base_model, adapter_name, is_trainable=False, merge=False):
    logger.info(f"Loading pretrained adapter weights from {adapter_name} ...")
    peft_model = PeftModel.from_pretrained(model=base_model,
                                           model_id=adapter_name,
                                           # torch_dtype=torch_dtype,
                                           is_trainable=is_trainable)
    if merge:
        logger.info(f"Merge adapter weights into the model...")
        base_model = peft_model.merge_and_unload()
        return base_model
    else:
        return peft_model
    
def get_parameters_summarization(model: PeftModel):
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        num_params = param.numel()
        # if using DS Zero 3 and the weights are initialized empty
        if num_params == 0 and hasattr(param, "ds_numel"):
            num_params = param.ds_numel

        all_param += num_params
        if param.requires_grad:
            trainable_params += num_params
    return {"trainable params": trainable_params, "all params": all_param, "trainable%": 100 * trainable_params / all_param}

def check_args_consistency(args1: dict, args2: dict, arg_names: list = []):
    if not arg_names:
        arg_names = args1.keys()
    wrong_arg_names = []
    for arg_name in arg_names:
        if args1[arg_name] != args2[arg_name]:
            wrong_arg_names.append(arg_name)
    if wrong_arg_names:
        raise ValueError(f"Arguments {wrong_arg_names} are inconsistent:\n{args1}\n{args2}")

def train(general_args, ds_args, ppn_args, training_args, reward_args, world_rank, local_rank):
    # Save arguments
    if world_rank == 0:
        args = {"general_args": general_args, "dialogue_sampling_args": ds_args,
                "ppn_args": ppn_args, "training_args": training_args, "reward_args": reward_args}
        save_args(args, os.path.join(general_args.run_dpath, "args.json"))
    
    # Set random seed for model initialization
    set_seed(general_args.random_seed)

    # Build PPN
    device_map = {"": local_rank} if training_args.ddp else "auto"
    torch_dtype = {"float32": torch.float32,
                   "float16": torch.float16,
                   "bfloat16": torch.bfloat16}[training_args.torch_dtype]
    logger.info("Loading PPN model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        ppn_args.base_model_name,
        load_in_8bit=training_args.load_in_8bit,
        torch_dtype=torch_dtype,
        device_map=device_map,
    )
    
    if training_args.load_in_8bit:
        # Cast norm layers' parameters to float32 for stable training
        params_casted_to_float32 = []
        for name, param in base_model.named_parameters():
            if param.dtype in [torch.float16, torch.bfloat16]:
                param.data = param.data.to(torch.float32)
                params_casted_to_float32.append(name)
        if params_casted_to_float32:
            logger.info(f"Parameters casted to float32: {params_casted_to_float32}")
        
    lora_config = LoraConfig(
        bias="none",
        lora_alpha=16,
        lora_dropout=0.05,
        r=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        task_type="CAUSAL_LM"
    )
    peft_model = get_peft_model(model=base_model, peft_config=lora_config)

    policy_model = AutoModelForCausalLMWithValueHead.from_pretrained(
        pretrained_model_name_or_path=peft_model,
        device_map=device_map,
        # v_head_init_strategy="normal"
    )
    policy_model.eval()
    if world_rank == 0:
        for name, params in policy_model.named_parameters():
            print(params.requires_grad, params.dtype, params.data.dtype, name)

    logger.info(f"Policy model: {get_parameters_summarization(policy_model)}")

    tokenizer = LlamaTokenizer.from_pretrained(ppn_args.base_model_name)

    ppn_nlg = PPNNLG(
        pretrained_name=ppn_args.base_model_name,
        model=policy_model,
        tokenizer=tokenizer,
        max_context_turns=ppn_args.max_context_turns,
        use_system_da=ppn_args.use_system_da,
        input_no_response=ppn_args.input_no_response,
        use_no_rephrasing_keyword=ppn_args.use_no_rephrasing_keyword,
        max_generation_tokens=ppn_args.max_generation_tokens,
        temperature=ppn_args.temperature,
        top_k=ppn_args.top_k,
        top_p=ppn_args.top_p,
        do_sample=ppn_args.do_sample,
    )

    # Set ppo trainer config
    ppo_config = FixedPPOConfig(
        model_name=ppn_args.base_model_name,
        steps=training_args.steps,
        batch_size=training_args.batch_size_per_device,
        mini_batch_size=training_args.mini_batch_size_per_device,
        gradient_accumulation_steps=training_args.gradient_accumulation_steps,
        ppo_epochs=training_args.num_epochs,
        learning_rate=training_args.learning_rate,
        adap_kl_ctrl=False,
        init_kl_coef=training_args.init_kl_coef,
        target=training_args.target_kl,
        horizon=training_args.horizon,
        gamma=training_args.gamma,
        lam=training_args.lam,
        cliprange=training_args.cliprange,
        cliprange_value=training_args.cliprange_value,
        vf_coef=training_args.vf_coef,
        ent_coef=training_args.ent_coef,

        log_with="wandb",
        accelerator_kwargs={ # Kwargs for huggingface accelerator
            "project_dir": general_args.run_dpath,
            # "mixed_precision": "fp16"
        },
        tracker_project_name=training_args.wandb_project_name, # Project name for wandb
        tracker_kwargs={ # Kwargs for initialzation of wandb tracker
            "wandb": {
                # "project": training_args.wandb_project_name, # Not needed since it is already passed above as tracker_project_name
                "group": training_args.wandb_group_name,
                "name": general_args.run_id,
            },
        },
        # optimize_cuda_cache=True
    )
    # Set custom arguments to record in wandb
    ppo_config.general_args = vars(general_args)
    ppo_config.dialogue_sampling_args = vars(ds_args)
    ppo_config.ppn_args = vars(ppn_args)
    ppo_config.training_args = vars(training_args)
    ppo_config.reward_args = vars(reward_args)

    # Build the optimizer and the scheduler
    optimizer = Adam(
        filter(lambda p: p.requires_grad, policy_model.parameters()), lr=training_args.learning_rate
    )
    lr_scheduler = get_constant_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=training_args.num_warmup_steps
    )

    # Build PPO trainer, passing the model, the reference model, the tokenizer
    # Note: torch.distributed (dist) of Accerelator is initialized in the PPOTrainer
    ppo_trainer = FixedPPOTrainer(config=ppo_config,
                                  model=policy_model,
                                  ref_model=None,
                                  tokenizer=tokenizer,
                                  optimizer=optimizer,
                                  lr_scheduler=lr_scheduler)

    # Load system and user agents
    sys_agent = SystemAgent(nlu_name=ds_args.system_nlu_name,
                            nlu_config_file=ds_args.system_nlu_config_file,
                            dst_name=ds_args.system_dst_name,
                            policy_name=ds_args.system_policy_name,
                            nlg_name=ds_args.system_nlg_name,
                            ppn_nlg=ppn_nlg)
    user_agent = UserAgent(nlu_name=ds_args.user_nlu_name,
                           nlu_config_file=ds_args.user_nlu_config_file,
                           policy_name=ds_args.user_policy_name,
                           nlg_name=ds_args.user_nlg_name,
                           max_turn=ds_args.max_turns_per_dialogue,
                           max_initiative=ds_args.user_max_initiative)
    
    # Make reward function
    reward_function = RewardFunction(reward_factors=reward_args.reward_factors)

    # Set up DA contribution model
    if reward_args.use_dac:
        dac_model = DialogueActContributionModel(process_id=world_rank,
                                                 dpath=os.path.join(general_args.run_dpath, "da_contribution"),
                                                 dialogue_dpath_to_init=reward_args.dac_dialogue_dpath_to_init,
                                                 num_dialogues_to_init=reward_args.dac_num_dialogues_to_init,
                                                 adaptive=reward_args.adaptive_dac,
                                                 include_value=False)
    else:
        dac_model = None

    # Create directory to save sampled dialogues
    dialogues_dpath = os.path.join(general_args.run_dpath, "dialogues")
    if ppo_trainer.accelerator.is_main_process:
        os.makedirs(dialogues_dpath, exist_ok=True)

    if training_args.ddp:
        dist.barrier()

    # Set different seed on each process for different dialogue sampling
    set_seed(general_args.random_seed+world_rank)
    
    # PPO training
    if world_rank == 0:
        hyperparams = pd.Series({
            "Total iterations": training_args.total_iterations,
            "Num. processes": training_args.world_size,
            "Batch size (per process)": training_args.batch_size_per_device,
            "Batch size (total)": training_args.world_size*training_args.batch_size_per_device,
            "Mini batch size (per process)": training_args.mini_batch_size_per_device,
            "Gradient accumulation steps": training_args.gradient_accumulation_steps,
            "Mini batch size (total)": training_args.world_size*training_args.mini_batch_size_per_device*training_args.gradient_accumulation_steps,
        })
        logger.info(f"Start PPO training:\n{hyperparams.to_string()}")
    for iteration_id in tqdm(range(training_args.total_iterations), desc="PPO Training", ncols=100):

        # 1. Sample dialogues
        if ppo_trainer.accelerator.is_main_process:
            logger.info(f"Sampling dialogues (iteration_id={iteration_id})")
        result = sample_dialogues(iteration_id=iteration_id, sys_agent=sys_agent, user_agent=user_agent,
                                  log_dpath=dialogues_dpath, max_turns_per_dialogue=ds_args.max_turns_per_dialogue,
                                  process_id=world_rank, examples_per_process=training_args.batch_size_per_device,
                                  validate_training_examples=True, drop_success_dialogue=training_args.drop_success_dialogue)

        # 2. Update DA Contribution Model
        if reward_args.use_dac and reward_args.adaptive_dac:
            if ppo_trainer.accelerator.is_main_process:
                logger.info("Updating DA contribution model")
            dac_model.update(iteration_id=iteration_id,
                             world_size=training_args.world_size,
                             dialogues=result.log)

        # 3. Make trajectory of PPN from the sampled dialogues
        if ppo_trainer.accelerator.is_main_process:
            logger.info("Gathering the trajectory from the sampled dialogues")
        dialogue_task_stats = []
        reward_factor_stats = []
        rewards = []
        prompt_tensors = []
        response_tensors = []
        table_data = []
        for dialogue in result.log:
            # 3-1. Gather dialogue task metrics
            dialogue_task_stats.append({
                key: dialogue[key] for key in ["task_complete", "task_success", "book_rate", "inform_f1",
                                               "inform_precision", "inform_recall", "goal_match_rate", "turn",
                                               "user_da_f1", "system_da_f1"]
            })
            
            # 3-2. Compute reward and gather the trajectory
            ppn_log = dialogue["ppn_log"][:-1] # Remove the last turn since it is not used for training
            reward_dicts = reward_function.compute_reward(dialogue, dac_model=dac_model)
            assert len(ppn_log) == len(reward_dicts), f"len(ppn_log)={len(ppn_log)} != len(reward_dicts)={len(reward_dicts)}"
            for i in range(len(reward_dicts)):
                is_training_example = reward_dicts[i].pop("is_training_example")
                if not is_training_example:
                    continue
                reward_factor_stats.append(reward_dicts[i])
                rewards.append(torch.tensor(reward_dicts[i]["reward"], dtype=torch.float32))
                prompt_tensors.append(ppn_log[i]["ppn_nlg"]["prompt_ids"])
                response_tensors.append(ppn_log[i]["ppn_nlg"]["response_ids"])
                table_data.append({
                    "prompt": ppn_log[i]["ppn_nlg"]["prompt_text"],
                    "response": ppn_log[i]["ppn_nlg"]["response_text"],
                    **reward_dicts[i]
                })
        reward_factor_stats = reward_factor_stats[:training_args.batch_size_per_device]
        rewards = rewards[:training_args.batch_size_per_device]
        prompt_tensors = prompt_tensors[:training_args.batch_size_per_device]
        response_tensors = response_tensors[:training_args.batch_size_per_device]
        table_data = table_data[:training_args.batch_size_per_device]
        
        # 4. Update the PPN model
        if ppo_trainer.accelerator.is_main_process:
            logger.info(f"Updating the PPN model (iteration_id={iteration_id})")
        stats = ppo_trainer.step(queries=prompt_tensors, responses=response_tensors, scores=rewards)

        # 5. Log stats
        dialogue_task_stats = ppo_trainer.gather_stats({
            # Note: Don't use numeric_only=True since book_rate is missed if it is NaN
            f"dialogue_task/{key}": torch.tensor(value, dtype=torch.float32, device=ppo_trainer.accelerator.device) 
            for key, value in pd.DataFrame(dialogue_task_stats).mean().to_dict().items()
        })
        reward_factor_stats = ppo_trainer.gather_stats({
            # Note: Use numeric_only=True to ignore some non-numeric values e.g., "dac" list
            # but this may miss some NaN columns and cause all_reduce mismatch 
            f"reward_factor/{key}": torch.tensor(value, dtype=torch.float32, device=ppo_trainer.accelerator.device) 
            for key, value in pd.DataFrame(reward_factor_stats).mean(numeric_only=True).to_dict().items()
        })
        stats.update({**dialogue_task_stats, **reward_factor_stats})
        
        wandb_table_data = {"game_log": wandb.Table(dataframe=pd.DataFrame(table_data))}
        if reward_args.use_dac:
            wandb_table_data["da_contribution"] = wandb.Table(dataframe=dac_model.get_data().reset_index())
        ppo_trainer.log_stats(stats=stats, wandb_table_data=wandb_table_data, rewards=rewards)

        # 6. Save the PPN model
        if ppo_trainer.accelerator.is_main_process and iteration_id > 0 and iteration_id % training_args.save_iterations == 0:
            logger.info(f"Saving the PPN model (iteration_id={iteration_id})")
            ppo_trainer.save_pretrained(os.path.join(general_args.run_dpath, f"checkpoints/iteration-{iteration_id}"))
        
        gc.collect(); torch.cuda.empty_cache()

    # Save the final PPN model
    if ppo_trainer.accelerator.is_main_process:
        logger.info(f"Saving the final PPN model")
        ppo_trainer.save_pretrained(os.path.join(general_args.run_dpath, "checkpoints/iteration-final"))

def test(general_args, ds_args, ppn_args, training_args, world_rank, local_rank):
    # Initialize distributed training
    if training_args.ddp:
        dist.init_process_group(backend="nccl",
                                world_size=training_args.world_size,
                                rank=world_rank)
        
    # Save arguments
    if world_rank == 0:
        args = {"general_args": general_args, "dialogue_sampling_args": ds_args,
                "ppn_args": ppn_args, "training_args": training_args}
        save_args(args, os.path.join(general_args.run_dpath, "test_args.json"))
    
    # Set random seed for model initialization
    set_seed(general_args.random_seed)

    # Build PPN
    device_map = {"": local_rank} if training_args.ddp else "auto"
    torch_dtype = {"float32": torch.float32,
                   "float16": torch.float16,
                   "bfloat16": torch.bfloat16}[training_args.torch_dtype]
    logger.info("Loading PPN model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        ppn_args.base_model_name,
        load_in_8bit=training_args.load_in_8bit,
        torch_dtype=torch_dtype,
        device_map=device_map,
    )
    
    if training_args.load_in_8bit or torch_dtype == torch.float32:
        # Cast norm layers' parameters to float32 for stable training
        params_casted_to_float32 = []
        for name, param in base_model.named_parameters():
            if param.dtype in [torch.float16, torch.bfloat16]:
                param.data = param.data.to(torch.float32)
                params_casted_to_float32.append(name)
        if params_casted_to_float32:
            logger.info(f"Parameters casted to float32: {params_casted_to_float32}")

    if ppn_args.adapter_checkpoint_path:
        # Sanity check
        adapter_training_args_path = os.path.join(os.path.dirname(os.path.dirname(ppn_args.adapter_checkpoint_path)), "args.json")
        adapter_training_args = json.load(open(adapter_training_args_path))
        check_args_consistency(adapter_training_args["dialogue_sampling_args"], vars(ds_args))
        check_args_consistency(adapter_training_args["ppn_args"], vars(ppn_args),
                               arg_names=["base_model_name", "max_context_turns", "use_system_da", "input_no_response"])
        check_args_consistency(adapter_training_args["training_args"], vars(training_args),
                               arg_names=["load_in_8bit", "torch_dtype"])
        # Load adapter weights
        peft_model = load_adapter_weights(base_model=base_model,
                                          adapter_name=ppn_args.adapter_checkpoint_path,
                                          is_trainable=False, merge=False)
    else:
        logger.info(f"No adapter is used.")
        peft_model = base_model
    
    if world_rank == 0:
        for name, params in peft_model.named_parameters():
            print(params.requires_grad, params.dtype, params.data.dtype, name)

    tokenizer = LlamaTokenizer.from_pretrained(ppn_args.base_model_name)

    ppn_nlg = PPNNLG(
        pretrained_name=ppn_args.base_model_name,
        model=peft_model,
        tokenizer=tokenizer,
        max_context_turns=ppn_args.max_context_turns,
        use_system_da=ppn_args.use_system_da,
        input_no_response=ppn_args.input_no_response,
        use_no_rephrasing_keyword=ppn_args.use_no_rephrasing_keyword,
        max_generation_tokens=ppn_args.max_generation_tokens,
        temperature=ppn_args.temperature,
        top_k=ppn_args.top_k,
        top_p=ppn_args.top_p,
        do_sample=ppn_args.do_sample,
    )

    # Load system and user agents
    sys_agent = SystemAgent(nlu_name=ds_args.system_nlu_name,
                            nlu_config_file=ds_args.system_nlu_config_file,
                            dst_name=ds_args.system_dst_name,
                            policy_name=ds_args.system_policy_name,
                            nlg_name=ds_args.system_nlg_name,
                            ppn_nlg=ppn_nlg)
    user_agent = UserAgent(nlu_name=ds_args.user_nlu_name,
                           nlu_config_file=ds_args.user_nlu_config_file,
                           policy_name=ds_args.user_policy_name,
                           nlg_name=ds_args.user_nlg_name,
                           max_turn=ds_args.max_turns_per_dialogue,
                           max_initiative=ds_args.user_max_initiative)

    # Create directory to save sampled dialogues
    dialogues_dpath = os.path.join(general_args.run_dpath, "test_dialogues")
    if world_rank == 0:
        if not os.path.exists(dialogues_dpath):
            os.makedirs(dialogues_dpath)

    if training_args.ddp:
        dist.barrier()

    # 1. Sample dialogues
    # Make goal seeds for each process
    goal_seeds =  np.reshape(TEST_GOAL_SEEDS, [training_args.world_size, -1])[world_rank].tolist()
    logger.info(f"Sampling dialogues for testing: world_rank: {world_rank}, goal_seeds: {goal_seeds}")
    result = sample_dialogues(iteration_id=0, sys_agent=sys_agent, user_agent=user_agent,
                              log_dpath=dialogues_dpath, max_turns_per_dialogue=ds_args.max_turns_per_dialogue,
                              process_id=world_rank, goal_seeds=goal_seeds,
                              validate_training_examples=False, drop_success_dialogue=False)

    # 2. Gather results
    if training_args.ddp:
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
        json.dump(log_summary, open(os.path.join(general_args.run_dpath, "test_summary.json"), "w"), indent=4)

def main():
    parser = HfArgumentParser((GeneralArguments, DialogueSamplingArguments, PPNArguments, PPOTrainingArguments, RewardArguments))
    general_args, ds_args, ppn_args, training_args, reward_args = parser.parse_args_into_dataclasses()

    # Set tf32
    if training_args.disallow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False

    # Setup distributed training
    if training_args.dist_type == DistributedEnvType.DEFAULT.value:
        world_size, world_rank, local_rank = set_default_env()
    elif training_args.dist_type == DistributedEnvType.MPI.value:
        world_size, world_rank, local_rank = set_mpi_env()
    else:
        raise ValueError(f"Unknown distributed environment type: {training_args.dist_type}")
    training_args.ddp = world_size != 1
    if training_args.ddp:
        logger.info(f"Using DistributedDataParallel: world_size: {world_size}, world_rank: {world_rank}, local_rank: {local_rank}")
        training_args.world_size = world_size
    else:
        logger.info("Not using DistributedDataParallel")

    # Do train or test
    if training_args.do_train:
        train(general_args=general_args, ds_args=ds_args, ppn_args=ppn_args, training_args=training_args, reward_args=reward_args,
              world_rank=world_rank, local_rank=local_rank)
    elif training_args.do_test:
        test(general_args=general_args, ds_args=ds_args, ppn_args=ppn_args, training_args=training_args,
             world_rank=world_rank, local_rank=local_rank)
    else:
        raise ValueError("You need to specify either do_train or do_test")
    
if __name__ == "__main__":
    main()
