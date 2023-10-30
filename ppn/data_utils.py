import os
import json
import random
import pandas as pd
import datasets
import numpy as np
from typing import List, Dict, Tuple, Any
from glob import glob
from tqdm import tqdm

from utils import AgentType, PromptSymbol

def load_df_from_dialogue_data(log_dpath):
    """Build a dataframe from the dialogue directory."""
    dicts = []
    for dialogue_fpath in tqdm(glob(os.path.join(log_dpath, "*.json")), desc="Loading dialogue data"):
        with open(dialogue_fpath, "r") as f:
            dialogue = json.load(f)
        
        history = []
        for turn in dialogue["system_dialog"]:
            history += [
                {"agent": AgentType.USER, "text": turn["user_utterance"]},
                {"agent": AgentType.SYSTEM, "text": turn["system_response"]}
            ]
        episode_id = dialogue["episode_id"] if "episode_id" in dialogue else dialogue["dialogue_id"]
        dicts.append({
            "dialogue_id": f'{dialogue["iteration_id"]}-{dialogue["process_id"]}-{episode_id}',
            "iteration_id": dialogue["iteration_id"],
            "process_id": dialogue["process_id"],
            "episode_id": episode_id,
            "task_complete": dialogue["task_complete"],
            "task_success": dialogue["task_success"],
            "book_rate": dialogue["book_rate"],
            "inform_f1": dialogue["inform_f1"],
            "inform_precision": dialogue["inform_precision"],
            "inform_recall": dialogue["inform_recall"],
            "reward": dialogue["common_reward_history"],
            "history": history,
        })
            
    return pd.DataFrame(dicts).set_index("dialogue_id")

def format_text_for_ppn(text: str, empty_text="<|empty|>"):
    if text:
        text = text.strip()
    else:
        text = empty_text
    return text

def format_text_from_ppn(text: str, empty_text="<|empty|>"):
    text = text.replace(empty_text, "")
    return text

def format_da_for_ppn(dialogue_act: List[Tuple[str, str, str, str]]):
    """
    Format the dialogue act for PPN.
    Exemple:
    da = [["Inform", "Hotel", "Choice", "2"],
          ["Recommend", "Hotel", "Name", "bridge guest house"],
          ["Recommend", "Hotel", "Phone", "01223247942"]]
    da_text = "Inform(Hotel-Choice=2), Recommend(Hotel-Name=bridge guest house, Hotel-Phone=01223247942)"
    """
    da_dict = {}
    for intent, domain, slot, value in dialogue_act:
        intent_domain = f"{intent}-{domain}"
        if not intent_domain in da_dict:
            da_dict[intent_domain] = {}
        da_dict[intent_domain][slot] = value
    da_text = []
    for intent_domain, slots in da_dict.items():
        slots = [f"{slot}={value}" for slot, value in slots.items()]
        da_text.append(f"{intent_domain}({'; '.join(slots)})")
    da_text = ", ".join(da_text)
    return da_text

def make_history_dicts_from_logs(previous_logs: List[Dict[str, Any]], current_log: Dict[str, Any], max_context_turns):
    """
    Make a history dictionary from the logs.
    Exemple:
    previous_logs = [
        {"turn_id": 0, "user_utterance": ..., "nlu": {"user_action": [...]}, "policy": {"system_action": [...]}, "nlg": {"system_response": ...}},
        ...
    ]
    current_log = {"turn_id": 1, "user_utterance": ..., "nlu": {"user_action": [...]}, "policy": {"system_action": [...]}, "nlg": {"system_response": ...}}

    history_dicts = [
        {"agent": "user", "text": ...},
        {"agent": "system", "da": ..., "text": ...},
        ...
        {"agent": "system", "da": ..., "text": ...}
    ]
    """
    
    history_dicts = []
    if max_context_turns > 0:
        for turn in previous_logs[-max_context_turns:]:
            history_dicts += [
                {"agent": AgentType.USER.value, "text": turn["user_utterance"]},
                {"agent": AgentType.SYSTEM.value, "da": turn["policy"]["system_action"], "text": turn["system_response"]}
            ]
        history_dicts.append(
            {"agent": AgentType.USER.value, "text": current_log["user_utterance"]}
        )

    history_dicts.append(
        {"agent": AgentType.SYSTEM.value, "da": current_log["policy"]["system_action"], "text": current_log["nlg"]["system_response"]},
    )
    return history_dicts

def make_prompt_text_from_history_dicts(pretrained_name, history_dicts, use_system_da=False, use_no_rephrasing_keyword=False):
    """
    Make a prompt text for PPN from history_dicts.
    """
    # Make context text
    context_text = []
    for turn in history_dicts:
        if turn["agent"] == AgentType.USER.value:
            text = format_text_for_ppn(turn["text"])
            context_text.append(PromptSymbol.USER.value + " " + text)
                
        elif turn["agent"] == AgentType.SYSTEM.value:
            if use_system_da:
                da = format_da_for_ppn(turn["da"])
                context_text.append(PromptSymbol.SYSTEM_DA.value + " " + da)
            text = format_text_for_ppn(turn["text"])
            context_text.append(PromptSymbol.SYSTEM.value + " " + text)

        else:
            raise ValueError(f"{turn['agent']} is not agent type.")
    context_text = PromptSymbol.SEPERATOR.value.join(context_text)

    # Make prompt text
    if pretrained_name in ["tloen/alpaca-lora-7b", "chavinlo/alpaca-native", "ohashi56225/alpaca-7b"]:
        prompt_text = (
            "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n"
            "\n"
            "### Instruction:\n"
            "Begin by reading a conversation between a customer and a chatbot about travel information.\n"
            "```\n"
            f"{context_text}\n"
            "```\n"
            "Your task is to rephrase the last chatbot's utterance so the customer can understand. "
        )
        if use_system_da:
            prompt_text += (
                f"Ensure to include the content of `{PromptSymbol.SYSTEM_DA.value}` in the utterance. "
            )
        if use_no_rephrasing_keyword:
            prompt_text += (
                f"If no rephrasing is necessary, just respond with `{PromptSymbol.NO_REPHRASING.value}`.\n"
            )
        else:
            prompt_text += (
                "If no rephrasing is necessary, repeat the original utterance.\n"
            )
        prompt_text += (
            "\n"
            "### Response:\n"
            f"{PromptSymbol.SYSTEM.value}"
        )
        
    else:
        raise ValueError(f"{pretrained_name} is not supported.")
    
    return prompt_text

def make_no_response_prompt_text_from_history_dicts(pretrained_name, history_dicts):
    # Make context text
    context_text = []
    for turn in history_dicts[:-1]:
        if turn["agent"] == AgentType.USER.value:
            text = format_text_for_ppn(turn["text"])
            context_text.append(PromptSymbol.USER.value + " " + text)
                
        elif turn["agent"] == AgentType.SYSTEM.value:
            da = format_da_for_ppn(turn["da"])
            context_text.append(PromptSymbol.SYSTEM_DA.value + " " + da)
            text = format_text_for_ppn(turn["text"])
            context_text.append(PromptSymbol.SYSTEM.value + " " + text)

        else:
            raise ValueError(f"{turn['agent']} is not agent type.")
    
    last_turn = history_dicts[-1]
    assert last_turn["agent"] == AgentType.SYSTEM.value
    da = format_da_for_ppn(last_turn["da"])
    context_text.append(PromptSymbol.SYSTEM_DA.value + " " + da)
    context_text = PromptSymbol.SEPERATOR.value.join(context_text)

    # Make prompt text
    if pretrained_name in ["tloen/alpaca-lora-7b", "chavinlo/alpaca-native", "ohashi56225/alpaca-7b"]:
        prompt_text = (
            "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n"
            "\n"
            "### Instruction:\n"
            "Begin by reading a conversation between a customer and a chatbot about travel information.\n"
            "```\n"
            f"{context_text}\n"
            "```\n"
            f"Your task is to generate the last chatbot's utterance so the customer can understand the content of `{PromptSymbol.SYSTEM_DA.value}`."
            "\n"
            "### Response:\n"
            f"{PromptSymbol.SYSTEM.value}"
        )
        
    else:
        raise ValueError(f"{pretrained_name} is not supported.")
    
    return prompt_text
    

def build_bc_dataset_from_dialogue_data(dialogue_data_log_dpath, pretrained_name, max_context_turns,
                                        use_system_da, use_no_rephrasing_keyword, system_da_f1_threshold=None,
                                        train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, random_seed=42):
    """
    Build a dataset for the Behavior Cloning.
    """
    assert train_ratio + val_ratio + test_ratio == 1.0, f"{train_ratio} + {val_ratio} + {test_ratio} != 1.0"

    if not os.path.isdir(dialogue_data_log_dpath):
        raise FileNotFoundError(dialogue_data_log_dpath)
    
    # 1. Load the dialogue_data as dataframes
    dicts = []
    for dialogue_fpath in tqdm(glob(os.path.join(dialogue_data_log_dpath, "*.json")), desc="Loading dialogue data"):
        with open(dialogue_fpath, "r") as f:
            dialogue = json.load(f)
        turns = dialogue["system_dialog"][:-1] # Last system response is not observed by the user
        turn_evals_of_da_accuracy = dialogue["turn_evals_of_da_accuracy"][:-1] # Last system response is not observed by the user
        for i in range(len(turns)):
            previous_logs = turns[:i]
            current_log = turns[i]
            da_accuracy = turn_evals_of_da_accuracy[i]

            # Skip empty system response
            if not current_log["nlg"]["system_response"].strip():
                continue

            history_dicts = make_history_dicts_from_logs(
                previous_logs=previous_logs, current_log=current_log, max_context_turns=max_context_turns
            )
            prompt_text = make_prompt_text_from_history_dicts(
                pretrained_name=pretrained_name, history_dicts=history_dicts, use_system_da=use_system_da,
                use_no_rephrasing_keyword=use_no_rephrasing_keyword
            )
            target_text = format_text_for_ppn(current_log["nlg"]["system_response"])
            dicts.append({
                "dialogue_id": f"{dialogue['iteration_id']}-{dialogue['process_id']}-{dialogue['episode_id']}",
                "turn_id": i,
                "prompt_text": prompt_text,
                "target_text": target_text,
                "system_da_f1": da_accuracy["system_da_f1"],
            })
    df = pd.DataFrame(dicts)

    # 2. Filter the dataframe
    # 2-1. system_da_f1_threshold
    if system_da_f1_threshold is not None:
        df = df[df["system_da_f1"] >= system_da_f1_threshold]

    # 3. Split the dataframe into train and valid (do not split by dialogue_id)
    train_size = int(len(df) * train_ratio)
    val_size = int(len(df) * val_ratio)
    train_df, val_df, test_df = np.split(df.sample(frac=1, random_state=random_seed), 
                                         [train_size, train_size+val_size])
    
    # 4. Convert the dataframes into HuggingFace's DatasetDict
    raw_datasets = datasets.DatasetDict({
        "train": datasets.Dataset.from_pandas(train_df),
        "validation": datasets.Dataset.from_pandas(val_df),
        "test": datasets.Dataset.from_pandas(test_df)
    })
    return raw_datasets


def _split_df_by_dialogue(df, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2, seed=42):
    """
    We split the dataset by dialogue_id to avoid data leakage.
    """
    assert sum([train_ratio, val_ratio, test_ratio]) == 1.0, \
        "The sum of train_ratio, val_ratio, and test_ratio must be 1.0."
    
    dialogue_ids = df["dialogue_id"].unique()
    dialogue_ids = np.random.RandomState(seed=seed).permutation(dialogue_ids)
    train_ids, val_ids, test_ids = np.split(
        dialogue_ids, [int(train_ratio*len(dialogue_ids)), int((train_ratio+val_ratio)*len(dialogue_ids))]
    )
    train_df = df[df["dialogue_id"].isin(train_ids)]
    val_df = df[df["dialogue_id"].isin(val_ids)]
    test_df = df[df["dialogue_id"].isin(test_ids)]
    return train_df, val_df, test_df


def get_context_and_responses(history, context_turns, context_start_idx, use_last_system_response, use_last_user_response):
    turn_ids = list(range(len(history)))

    # Get the indices of the context
    context_end_idx = context_start_idx+context_turns
    if use_last_system_response:
        if context_end_idx % 2 != 1:
            context_start_idx += 1
            context_end_idx += 1
    elif use_last_user_response:
        if context_end_idx % 2 != 0:
            context_start_idx += 1
            context_end_idx += 1
            
    # Get the indices of the last responses
    response_start_idx = context_end_idx
    response_end_idx = response_start_idx + 1
    if use_last_system_response and use_last_user_response:
        response_end_idx += 1

    if response_end_idx > len(turn_ids):
        return None
    
    context_indices = turn_ids[context_start_idx:context_end_idx]
    last_responses_indices = turn_ids[response_start_idx:response_end_idx]

    result = {
        "context_indices": context_indices,
        "context": [history[idx] for idx in context_indices],
        "last_responses_indices": last_responses_indices,
        "last_responses": [history[idx] for idx in last_responses_indices]
    }
    return result

def build_rm_datasets_from_dialogue_data(dialogue_data_log_dpath,
                                         use_last_system_response, use_last_user_response,
                                         fixed_context_turns=3, min_context_turns=2, max_context_turns=4,
                                         high_value_text_column="high_value_text", low_value_text_column="low_value_text",
                                         train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, seed=42):
    """Build a dataset for reward model (rm) from the dataframe."""

    # 1. Load the dialogue_data as dataframe
    if not os.path.isdir(dialogue_data_log_dpath):
        raise FileNotFoundError(dialogue_data_log_dpath)
    df = load_df_from_dialogue_data(dialogue_data_log_dpath)
    

    # 2. Select turns for the context and the last utterances.
    def get_context_turns():
        if fixed_context_turns is not None:
            return fixed_context_turns
        else:
            return random.randint(min_context_turns, max_context_turns)

    selected_turns = []
    for dialogue_id, dialogue_series in df.iterrows():
        context_start_idx = random.randrange(0, get_context_turns())
        while True:
            context_and_responses = get_context_and_responses(history=dialogue_series.history,
                                                              context_turns=get_context_turns(),
                                                              context_start_idx=context_start_idx,
                                                              use_last_system_response=use_last_system_response,
                                                              use_last_user_response=use_last_user_response)
            if not context_and_responses:
                break

            selected_turns.append({
                "dialogue_id": dialogue_series.name,
                "task_success": dialogue_series.task_success,
                **context_and_responses
            })
            
            context_start_idx = context_and_responses["last_responses_indices"][-1] + 1

    df = pd.DataFrame(selected_turns)


    # 3. Build the high value and low value text
    # 3-1. Define the function to build the text
    def make_high_value_text_funct(row):
        if row["task_success"]:
            # We assume that if the task is successful, the following utterances are high value
            text = make_history_text(history=row["context"] + row["last_responses"],
                                     user_utterance_prefix=user_utterance_prefix, system_response_prefix=system_response_prefix)
        else:
            # We assume that if the task is failed, the context is high value compared to the following utterances
            text = make_history_text(history=row["context"],
                                     user_utterance_prefix=user_utterance_prefix, system_response_prefix=system_response_prefix)
        return text
    
    def make_low_value_text_funct(row):
        if row["task_success"]:
            # We assume that if the task is successful, the context is low value compared to the following utterances
            text = make_history_text(history=row["context"],
                                     user_utterance_prefix=user_utterance_prefix, system_response_prefix=system_response_prefix)
        else:
            # We assume that if the task is failed, the system response is low value
            text = make_history_text(history=row["context"] + row["last_responses"],
                                     user_utterance_prefix=user_utterance_prefix, system_response_prefix=system_response_prefix)
        return text

    # 3-2. Apply the function
    df[high_value_text_column] = df.apply(make_high_value_text_funct, axis=1)
    df[low_value_text_column] = df.apply(make_low_value_text_funct, axis=1)

    # 4. Split the dataset
    train_df, val_df, test_df = _split_df_by_dialogue(df, train_ratio=train_ratio,
                                                      val_ratio=val_ratio, test_ratio=test_ratio, seed=seed)
    
    raw_datasets = datasets.DatasetDict({
        "train": datasets.Dataset.from_pandas(train_df),
        "validation": datasets.Dataset.from_pandas(val_df),
        "test": datasets.Dataset.from_pandas(test_df)
    })

    return raw_datasets