from copy import deepcopy
import os
import json
import math
import pandas as pd
from logging import getLogger
from dataclasses import dataclass
from transformers import set_seed

from utils.log import set_logger
from utils.evaluator import Evaluator

logger = getLogger(__name__)
set_logger(logger)


def exclude_dst_history(log):
    """Delete system.dst["state"]["history"]"""
    log = deepcopy(log)
    for turn_id in range(len(log)):
        try:
            if "history" in log[turn_id]["dst"]["dialogue_state"]:
                del log[turn_id]["dst"]["dialogue_state"]["history"]
        except KeyError:
            pass
    return log

class Session:
    """
    Manage the interaction between the system and the simulator
    Reference to convlab2.dialog_agent.BiSession
    """
    def __init__(self, goal_seed, sys_agent, user_agent, evaluator, iteration_id, process_id, episode_id, log_dpath, drop_success_dialogue):
        self.goal_seed = goal_seed
        self.sys_agent = sys_agent
        self.user_agent = user_agent
        self.evaluator = evaluator
        self.iteration_id = iteration_id
        self.process_id = process_id
        self.episode_id = episode_id
        self.log_fpath = os.path.join(log_dpath, f"{iteration_id}-{process_id}-{episode_id}.json")
        self.drop_success_dialogue = drop_success_dialogue

    def init_session(self):
        self.sys_agent.init_session()
        self.user_agent.init_session()
        goal = self.user_agent.policy.get_goal()
        self.evaluator.add_goal(goal)

        self.final_goal = goal
        self.initial_goal = deepcopy(goal)

        self.turn = 0
        self.turn_evals_from_user = []
        self.turn_evals_from_system = []

    def step(self, last_system_response):
        user_response = self.user_agent.response(last_system_response)

        self.evaluator.add_sys_da(self.user_agent.get_in_da())
        self.evaluator.add_usr_da(self.user_agent.get_out_da())

        turn_eval_from_user = self.evaluator.evaluate_from_user(self.user_agent)
        self.turn_evals_from_user.append(turn_eval_from_user)

        if hasattr(self.sys_agent, 'dst'):
            self.sys_agent.dst.state['terminated'] = turn_eval_from_user["session_over"]
        
        system_response = self.sys_agent.response(user_response) # Final turn is processed by the system side for episode termination
        
        turn_eval_from_system = self.evaluator.evaluate_from_system(self.sys_agent)
        self.turn_evals_from_system.append(turn_eval_from_system)

        self.turn += 1

        return user_response, turn_eval_from_user["session_over"], system_response

    def save_log(self):
        task_complete = self.user_agent.policy.policy.goal.task_complete()
        prec, rec, f1 = self.evaluator.inform_F1()
        task_success = self.evaluator.task_success()
        book_rate = self.evaluator.book_rate()
        goal_match_rate = self.evaluator.final_goal_analyze()
        is_training_example = self.sys_agent.is_training_example
        if self.drop_success_dialogue:
            is_training_example = [bool(not task_success and mask) for mask in is_training_example]

        turn_evals_of_da_accuracy = self.evaluator.evaluate_da_accuracy(user_agent=self.user_agent, sys_agent=self.sys_agent)
        user_da_f1, system_da_f1 = pd.json_normalize(turn_evals_of_da_accuracy).mean(numeric_only=True)[["user_da.f1", "system_da.f1"]]

        turn_evals_from_reqt_goal = self.evaluator.evaluate_from_reqt_goal(self.sys_agent)

        log_data = {
            "goal_seed": self.goal_seed,
            "iteration_id": self.iteration_id,
            "process_id": self.process_id,
            "episode_id": self.episode_id,
            "initial_goal": self.initial_goal,
            "final_goal": self.final_goal,
            "task_complete": task_complete,
            "task_success": task_success,
            "book_rate": book_rate,
            "inform_f1": f1,
            "inform_precision": prec,
            "inform_recall": rec,
            "goal_match_rate": goal_match_rate,
            "user_da_f1": user_da_f1,
            "system_da_f1": system_da_f1,
            "turn": self.turn,
            "turn_evals_from_user": self.turn_evals_from_user,
            "turn_evals_from_system": self.turn_evals_from_system,
            "turn_evals_of_da_accuracy": turn_evals_of_da_accuracy,
            "turn_evals_from_reqt_goal": turn_evals_from_reqt_goal,
            "user_dialog": self.user_agent.log,
            "system_dialog": exclude_dst_history(self.sys_agent.log),
            "is_training_example":is_training_example,
        }

        json.dump(log_data, open(self.log_fpath, "w"), indent=4)
        log_data["ppn_log"] = self.sys_agent.ppn_log
        return log_data

@dataclass
class SampledResult:
    """
    Dataclass for storing the sampled results
    """
    process_id: int
    log: list
    sampled_turns: int
    sampled_training_examples: int

def sample_dialogues(iteration_id, sys_agent, user_agent, log_dpath, max_turns_per_dialogue,
                     process_id, goal_seeds=None, examples_per_process=None,
                     validate_training_examples=False, drop_success_dialogue=False):
    assert goal_seeds is not None or examples_per_process is not None, \
        "Either examples_per_process or goal_seeds should be provided"
    assert goal_seeds is None or examples_per_process is None, \
        "Only one of examples_per_process and goal_seeds should be provided"
    
    def is_sampling_finished(sampled_dialogues, sampled_turns, sampled_training_examples):
        if goal_seeds is not None:
            return sampled_dialogues < len(goal_seeds)
        elif validate_training_examples:
            return sampled_training_examples < examples_per_process
        else:
            return sampled_turns < examples_per_process
    
    logger.info(f"Process {process_id} starts")

    sampled_dialogues = 0
    sampled_turns = 0
    sampled_training_examples = 0

    # Sample dialogues
    log = []
    while is_sampling_finished(sampled_dialogues, sampled_turns, sampled_training_examples):

        if goal_seeds is not None:
            goal_seed = goal_seeds[sampled_dialogues]
            set_seed(goal_seed)
        else:
            goal_seed = None

        evaluator = Evaluator(max_turn=max_turns_per_dialogue,
                              reward_type="task_success")
        session = Session(goal_seed=goal_seed,
                          sys_agent=sys_agent,
                          user_agent=user_agent,
                          evaluator=evaluator,
                          iteration_id=iteration_id,
                          process_id=process_id,
                          episode_id=sampled_dialogues,
                          log_dpath=log_dpath,
                          drop_success_dialogue=drop_success_dialogue)
        session.init_session()

        system_response = ""
        for t in range(max_turns_per_dialogue+1):
            user_response, session_over, system_response = session.step(system_response)
            if session_over:
                break

        # Get and record dialogue history and log
        log_ = session.save_log()
        log.append(log_)

        # Increment sampled data
        sampled_dialogues += 1
        sampled_turns += t # t == len(session.log_["system_dialog"]) - 1
        sampled_training_examples += sum(log_["is_training_example"])

    result = SampledResult(process_id=process_id, log=log, sampled_turns=sampled_turns,
                           sampled_training_examples=sampled_training_examples)

    logger.info(f"Process {process_id} exits")
    return result