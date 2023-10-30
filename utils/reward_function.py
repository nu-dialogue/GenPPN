import numpy as np
import pandas as pd
from typing import List, Dict, Any
# from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk import edit_distance

from utils import RewardFactor
from utils.evaluator import flatten_da
from utils.da_contribution import DialogueActContributionModel

def _get_score_diff(curr_score, next_score):
    if next_score is None:
        return 0
    if curr_score is None:
        return next_score
    return next_score - curr_score

def compute_inform_f1(dialogue_log: Dict[str, Dict[str, Any]]):
    rewards = []
    turn_evals = dialogue_log["turn_evals_from_user"]

    for i in range(len(turn_evals[:-1])): # Not include the last turn
        curr_score = turn_evals[i]["inform_f1"]
        next_score = turn_evals[i+1]["inform_f1"]
        rewards.append(_get_score_diff(curr_score=curr_score, next_score=next_score))
    return rewards

def compute_book_rate(dialogue_log: Dict[str, Dict[str, Any]]):
    rewards = []
    turn_evals = dialogue_log["turn_evals_from_user"]

    for i in range(len(turn_evals[:-1])): # Not include the last turn
        curr_score = turn_evals[i]["book_rate"]
        next_score = turn_evals[i+1]["book_rate"]
        rewards.append(_get_score_diff(curr_score=curr_score, next_score=next_score))
    return rewards

def compute_belief_accurcay(dialogue_log: Dict[str, Dict[str, Any]]):
    rewards = []
    turn_evals = dialogue_log["turn_evals_from_system"]

    for i in range(len(turn_evals[:-1])): # Not include the last turn
        curr_score = turn_evals[i]["belief_state"]["accuracy"]
        next_score = turn_evals[i+1]["belief_state"]["accuracy"]
        rewards.append(_get_score_diff(curr_score=curr_score, next_score=next_score))
    return rewards

def compute_fail_penalty(dialogue_log, penalty_value):
    turn_evals = dialogue_log["turn_evals_from_user"]
    if not dialogue_log["task_success"]:
        penalties = [penalty_value] * len(turn_evals[:-1]) # Not include the last turn
    else:
        penalties = [0.0] * len(turn_evals[:-1]) # Not include the last turn
    return penalties

# def compute_bleu_penalty(dialogue_log, smoothing_function_name, coef_value):
#     if smoothing_function_name is None:
#         smooth_func = None
#     else:
#         smooth_func = getattr(SmoothingFunction(), smoothing_function_name)
    
#     penalties = []
#     *system_turns, _ = dialogue_log["system_dialog"] # User don't observe the final system's response.
    
#     for sys_turn in system_turns:
#         ori_tokens = sys_turn["nlg"]["system_response"].split()
#         gen_tokens = sys_turn["ppn_nlg"]["system_response"].split()
#         bleu_score = sentence_bleu(references=[ori_tokens], hypothesis=gen_tokens, smoothing_function=smooth_func)
#         penalties.append(-bleu_score * coef_value)
#     return penalties

def compute_system_da_f1(dialogue_log: Dict[str, Dict[str, Any]]):
    reward_data = {
        "system_da_f1": [],
    }
    *system_turns, _ = dialogue_log["system_dialog"] # User don't observe the final system's response.
    *turn_evals_of_da_accuracy, _ = dialogue_log["turn_evals_of_da_accuracy"]

    assert len(system_turns) == len(turn_evals_of_da_accuracy), \
        "The number of system turns and turn evals for da should be the same."

    for turn_eval in turn_evals_of_da_accuracy:
        reward_data["system_da_f1"].append(turn_eval["system_da"]["f1"])
    return reward_data

def compute_task_success(dialogue_log: Dict[str, Dict[str, Any]]):
    reward_data = {
        "task_success": [],
    }
    for i in range(len(dialogue_log["system_dialog"][:-1])):
        reward_data["task_success"].append(dialogue_log["task_success"])
    return reward_data

def compute_inform_book_match(dialogue_log: Dict[str, Dict[str, Any]], agg_func="mean"):
    reward_data = {
        "inform_book_match": [],
    }
    scores = np.array([dialogue_log["inform_recall"], # This can be None
                       dialogue_log["book_rate"], # This can be None
                       dialogue_log["goal_match_rate"]],
                       dtype=float)
    if agg_func == "mean":
        total_score = np.nanmean(scores).item()
    elif agg_func == "sum":
        total_score = np.nansum(scores).item()
    elif agg_func == "product":
        total_score = np.nanprod(scores).item()
    else:
        raise ValueError(f"Invalid aggregation function: {agg_func}")

    for i in range(len(dialogue_log["system_dialog"][:-1])):
        reward_data["inform_book_match"].append(total_score)
    return reward_data

def compute_user_da_change(dialogue_log: Dict[str, Dict[str, Any]]):
    reward_data = {
        "user_da_change": [],
    }
    system_turns = dialogue_log["system_dialog"]
    assert len(system_turns) == len(dialogue_log["user_dialog"]), "The number of system turns and user turns should be the same."

    for i in range(len(system_turns[:-1])): # User don't observe the final system's response.
        curr_turn = system_turns[i]
        next_turn = system_turns[i+1]
        curr_user_da = set(flatten_da(curr_turn["nlu"]["user_action"]))
        next_user_da = set(flatten_da(next_turn["nlu"]["user_action"]))
        if any([curr_user_da, next_user_da]):
            da_overlap_rate = len(curr_user_da & next_user_da) / len(curr_user_da | next_user_da)
        else:
            da_overlap_rate = 1.0 # Both user actions are empty.
        reward_data["user_da_change"].append(1.0 - da_overlap_rate)
    return reward_data

def compute_reqt_goal_accuracy(dialogue_log: Dict[str, Dict[str, Any]], base_acc):
    reward_data = {
        "reqt_goal_accuracy": [],
    }
    *system_turns, _ = dialogue_log["system_dialog"] # User don't observe the final system's response.
    *turn_evals_from_reqt_goal, _ = dialogue_log["turn_evals_from_reqt_goal"] # User don't observe the final system's response.
    assert len(system_turns) == len(turn_evals_from_reqt_goal), \
        "The number of system turns and turn evals from reqt goal should be the same."
    
    for turn_eval in turn_evals_from_reqt_goal:
        acc = turn_eval["accuracy"]
        if acc is None:
            acc = base_acc
        reward_data["reqt_goal_accuracy"].append(acc)
    return reward_data

def compute_distance_penalty(dialogue_log: Dict[str, Dict[str, Any]], penalty_scale: float):
    penalty_data = {
        "edit_distance": [],
        "distance_penalty": [],
    }
    *system_turns, _ = dialogue_log["system_dialog"] # User don't observe the final system's response.

    for sys_turn in system_turns:
        ori_tokens = sys_turn["nlg"]["system_response"].split()
        gen_tokens = sys_turn["ppn_nlg"]["system_response"].split()
        norm_distance = edit_distance(ori_tokens, gen_tokens) / max(len(ori_tokens), len(gen_tokens))
        penalty = norm_distance + (1-norm_distance)*penalty_scale
        penalty_data["edit_distance"].append(norm_distance)
        penalty_data["distance_penalty"].append(penalty)
    return penalty_data

def compute_da(dialogue_log: Dict[str, Dict[str, Any]], agg_func="mean"):
    reward_data = {
        "da": [],
        f"da_{agg_func}": [],
    }
    turn_evals_of_da_accuracy = dialogue_log["turn_evals_of_da_accuracy"][:-1] # User don't observe the final system's response.
    for turn_eval in turn_evals_of_da_accuracy:
        tp_da = turn_eval["system_da"]["tp_da"]
        fn_da = turn_eval["system_da"]["fn_da"]
        da_scores = [1]*len(tp_da) + [-1]*len(fn_da)
        if agg_func == "mean":
            total_score = np.mean(da_scores).item()
        else:
            raise ValueError(f"Invalid aggregation function: {agg_func}")
        reward_data["da"].append(da_scores)
        reward_data[f"da_{agg_func}"].append(total_score)
    return reward_data

def compute_da_contribution(dialogue_log: Dict[str, Dict[str, Any]], dac_model: DialogueActContributionModel, agg_func):
    reward_data = {
        "da_contribution": [],
        f"da_contribution_{agg_func}": [],
    }
    turn_evals_of_da_accuracy = dialogue_log["turn_evals_of_da_accuracy"][:-1] # User don't observe the final system's response.
    for turn_eval in turn_evals_of_da_accuracy:
        tp_da = turn_eval["system_da"]["tp_da"]
        fn_da = turn_eval["system_da"]["fn_da"]
        tp_da_contribution = dac_model.get_contribution(tp_da)
        fn_da_contribution = dac_model.get_contribution(fn_da)
        da_scores = np.r_[
            tp_da_contribution, # Positive score for true positive DAs.
            fn_da_contribution * -1 # Negative score for false negative DAs.
        ]
        if agg_func == "mean":
            total_score = np.nanmean(da_scores).item()
        elif agg_func == "absmax":
            total_score = da_scores[np.abs(da_scores).argmax()].item()
        else:
            raise ValueError(f"Invalid aggregation function: {agg_func}")
        reward_data["da_contribution"].append(da_scores.round(3).tolist())
        reward_data[f"da_contribution_{agg_func}"].append(total_score)
    return reward_data

class RewardFunction:
    def __init__(self, reward_factors):
        self.reward_factors = []
        self.penalty_factors = []
        self.distance_penalty_scale = 1.0
        self.reqt_goal_base_acc = 1.0
        for factor in reward_factors:
            if "_with_distance_penalty" in factor:
                main_factor, distance_penalty_scale = factor.split("_with_distance_penalty")
                factor = main_factor + "_with_distance_penalty"
                self.distance_penalty_scale = float(distance_penalty_scale)

            if "_with_reqt_goal_accuracy" in factor:
                main_factor, base_acc = factor.split("_with_reqt_goal_accuracy")
                factor = main_factor + "_with_reqt_goal_accuracy"
                self.reqt_goal_base_acc = float(base_acc)

            if factor not in [e.value for e in RewardFactor]:
                raise ValueError(f"Invalid reward factor: {factor}")
            
            self.reward_factors.append(factor)


    def compute_reward(self, dialogue_log, dac_model: DialogueActContributionModel):
        rewards = {
            "is_training_example": dialogue_log["is_training_example"][:-1]
        }

        # Inform F1
        if RewardFactor.INFORM_F1.value in self.reward_factors:
            raise NotImplementedError
            rewards[RewardFactor.INFORM_F1.value] = compute_inform_f1(dialogue_log)

        # Book Rate
        if RewardFactor.BOOK_RATE.value in self.reward_factors:
            raise NotImplementedError
            rewards[RewardFactor.BOOK_RATE.value] = compute_book_rate(dialogue_log)
        
        # DA F1
        if RewardFactor.SYSTEM_DA_F1.value in self.reward_factors:
            rewards.update(compute_system_da_f1(dialogue_log))
        
        # DA F1 with distance penalty
        if RewardFactor.SYSTEM_DA_F1_WITH_DISTANCE_PENALTY.value in self.reward_factors:
            raise NotImplementedError
            da_accuracy = compute_da_accuracy(dialogue_log)
            penalties = compute_distance_penalty(dialogue_log, penalty_scale=self.distance_penalty_scale)
            assert len(da_accuracy) == len(penalties), "The number of DA accuracy and penalties should be the same."
            rewards[RewardFactor.DA_ACCURACY.value] = da_accuracy # For debugging
            rewards[RewardFactor.DA_ACCURACY_WITH_DISTANCE_PENALTY.value] = [r*p for r, p in zip(da_accuracy, penalties)]

        # Belief accuracy
        if RewardFactor.BELIEF_ACCURACY.value in self.reward_factors:
            raise NotImplementedError
            rewards[RewardFactor.BELIEF_ACCURACY.value] = compute_belief_accurcay(dialogue_log)

        # User DA change
        if RewardFactor.USER_DA_CHANGE.value in self.reward_factors:
            rewards.update(compute_user_da_change(dialogue_log))

        if RewardFactor.USER_DA_CHANGE_WITH_DISTANCE_PENALTY.value in self.reward_factors:
            rewards.update({
                **compute_user_da_change(dialogue_log),
                **compute_distance_penalty(dialogue_log, penalty_scale=self.distance_penalty_scale)
            })
            assert len(rewards["user_da_change"]) == len(rewards["distance_penalty"]), \
                "The number of user DA change and distance penalty should be the same."
            
            rewards[RewardFactor.USER_DA_CHANGE_WITH_DISTANCE_PENALTY.value] = [
                r*p for r, p in zip(rewards["user_da_change"], rewards["distance_penalty"])
            ]

        # Task Success
        if RewardFactor.TASK_SUCCESS.value in self.reward_factors:
            rewards.update(compute_task_success(dialogue_log))

        # Task Success with distance penalty
        if RewardFactor.TASK_SUCCESS_WITH_DISTANCE_PENALTY.value in self.reward_factors:
            rewards.update({
                **compute_task_success(dialogue_log),
                **compute_distance_penalty(dialogue_log, penalty_scale=self.distance_penalty_scale)
            })
            assert len(rewards["task_success"]) == len(rewards["distance_penalty"]), \
                "The number of task success and distance penalty should be the same."
            
            rewards[RewardFactor.TASK_SUCCESS_WITH_DISTANCE_PENALTY.value] = [
                r*p for r, p in zip(rewards["task_success"], rewards["distance_penalty"])
            ]

        # Inform Recall, Book Rate and Goal Match Rate
        if RewardFactor.INFORM_BOOK_MATCH.value in self.reward_factors:
            rewards.update(compute_inform_book_match(dialogue_log))

        # Inform Recall, Book Rate and Goal Match Rate with distance penalty
        if RewardFactor.INFORM_BOOK_MATCH_WITH_DISTANCE_PENALTY.value in self.reward_factors:
            rewards.update({
                **compute_inform_book_match(dialogue_log),
                **compute_distance_penalty(dialogue_log, penalty_scale=self.distance_penalty_scale)
            })
            assert len(rewards["inform_book_match"]) == len(rewards["distance_penalty"]), \
                "The number of inform-book-match and distance penalty should be the same."
            
            rewards[RewardFactor.INFORM_BOOK_MATCH_WITH_DISTANCE_PENALTY.value] = [
                r*p for r, p in zip(rewards["inform_book_match"], rewards["distance_penalty"])
            ]
        
        # Inform Recall, Book Rate and Goal Match Rate with User DA change
        if RewardFactor.INFORM_BOOK_MATCH_WITH_USER_DA_CHANGE.value in self.reward_factors:
            rewards.update({
                **compute_inform_book_match(dialogue_log),
                **compute_user_da_change(dialogue_log)
            })
            assert len(rewards["inform_book_match"]) == len(rewards["user_da_change"]), \
                "The number of inform-book-match and user DA change should be the same."
            
            rewards[RewardFactor.INFORM_BOOK_MATCH_WITH_USER_DA_CHANGE.value] = [
                r*c for r, c in zip(rewards["inform_book_match"], rewards["user_da_change"])
            ]

        # Inform Recall, Book Rate and Goal Match Rate with System DA F1
        if RewardFactor.INFORM_BOOK_MATCH_WITH_SYSTEM_DA_F1.value in self.reward_factors:
            rewards.update({
                **compute_inform_book_match(dialogue_log),
                **compute_system_da_f1(dialogue_log)
            })
            assert len(rewards["inform_book_match"]) == len(rewards["system_da_f1"]), \
                "The number of inform-book-match and system DA F1 should be the same."
            
            rewards[RewardFactor.INFORM_BOOK_MATCH_WITH_SYSTEM_DA_F1.value] = [
                r*f for r, f in zip(rewards["inform_book_match"], rewards["system_da_f1"])
            ]

        # Inform Recall, Book Rate and Goal Match Rate with Reqt Goal Acc.
        if RewardFactor.INFORM_BOOK_MATCH_WITH_REQT_GOAL_ACCURACY.value in self.reward_factors:
            rewards.update({
                **compute_inform_book_match(dialogue_log),
                **compute_reqt_goal_accuracy(dialogue_log, base_acc=self.reqt_goal_base_acc)
            })
            assert len(rewards["inform_book_match"]) == len(rewards["reqt_goal_accuracy"]), \
                "The number of inform-book-match and reqt goal should be the same."
            
            rewards[RewardFactor.INFORM_BOOK_MATCH_WITH_REQT_GOAL_ACCURACY.value] = [
                r*g for r, g in zip(rewards["inform_book_match"], rewards["reqt_goal_accuracy"])
            ]

        # DA
        if RewardFactor.DA_MEAN.value in self.reward_factors:
            rewards.update(compute_da(dialogue_log, agg_func="mean"))

        # DA contribution mean
        if RewardFactor.DA_CONTRIBUTION_MEAN.value in self.reward_factors:
            rewards.update(compute_da_contribution(dialogue_log, dac_model=dac_model, agg_func="mean"))
        
        # DA contribution abs max
        if RewardFactor.DA_CONTRIBUTION_ABSMAX.value in self.reward_factors:
            rewards.update(compute_da_contribution(dialogue_log, dac_model=dac_model, agg_func="absmax"))

        # Task Fail penalty
        if RewardFactor.FAIL_PENALTY.value in self.penalty_factors:
            raise NotImplementedError
            def check_to_penalize(row):
                if not row["is_training_example"]: # Exclude non-training examples
                    return False
                if (row[self.reward_factors] != 0).any(): # Exclude examples that have non-zero reward
                    return False
                return True
            examples_to_penalize = pd.DataFrame(rewards).apply(check_to_penalize, axis=1).to_list()
            fail_penalty = compute_fail_penalty(dialogue_log, penalty_value=-0.2)
            rewards[RewardFactor.FAIL_PENALTY.value] = [penalty if penalize else 0.0 for penalty, penalize in zip(fail_penalty, examples_to_penalize)]
        
        # Compute total reward
        rewards_df = pd.DataFrame(rewards)
        def total_reward(row):
            if row["is_training_example"]:
                return row[self.reward_factors].sum()
            else:
                return None
        rewards_df["reward"] = rewards_df.apply(total_reward, axis=1)
        return rewards_df.to_dict(orient="records")