import torch
import numpy as np
from typing import Union, Tuple, List, Dict
from logging import getLogger
from ppn.data_utils import make_history_text, get_context_and_responses
from utils import const, set_logger

logger = getLogger(__name__)
set_logger(logger)

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class RewardFunction:
    def __init__(self, model, tokenizer, context_turns, use_last_system_response, use_last_user_response) -> None:
        self.tokenizer = tokenizer
        self.model = model.to(DEVICE)

        self.context_turns = context_turns
        self.use_last_system_response = use_last_system_response
        self.use_last_user_response = use_last_user_response

    def get_logits(self, input_texts: List[str]) -> torch.Tensor:
        """
        Compute reward for a single turn.
        """
        self.tokenizer.truncation_side = "left"
        model_inputs = self.tokenizer(input_texts, add_special_tokens=True, return_tensors="pt",
                                      truncation=True, padding=True)
        self.tokenizer.truncation_side = "right"
        with torch.no_grad():
            outputs = self.model(**model_inputs.to(DEVICE))
        logits = outputs.logits.squeeze(-1).cpu() # (batch_size, 1) => (batch_size)

        return logits

    def compute_dialogue_rewards(self, history: List[Dict[str, str]],
                                 user_utterance_prefix: str = "User:", system_response_prefix: str = "System:") -> torch.Tensor:
        """
        Compute reward for a whole dialogue.
        """

        # Compute the rewards for each response
        input_texts = []
        evaluated_system_response_indices = []
        context_start_idx = 0
        while True:
            context_and_responses = get_context_and_responses(history=history,
                                                              context_turns=self.context_turns,
                                                              context_start_idx=context_start_idx,
                                                              use_last_system_response=self.use_last_system_response,
                                                              use_last_user_response=self.use_last_user_response)
            if context_and_responses is None:
                break

            input_texts.append(
                make_history_text(history=context_and_responses["context"] + context_and_responses["last_responses"],
                                  user_utterance_prefix=user_utterance_prefix, system_response_prefix=system_response_prefix)
            )
            context_start_idx = context_and_responses["context_indices"][0] + 1

            if self.use_last_system_response:
                evaluated_system_response_indices.append(
                    context_and_responses["last_responses_indices"][0]
                )
            elif self.use_last_user_response:
                evaluated_system_response_indices.append(
                    context_and_responses["context_indices"][-1]
                )

        if not input_texts:
            logger.warning("No input texts for computing rewards. Return None.")
            rewards = None
        else:
            rewards = self.get_logits(input_texts)

        return evaluated_system_response_indices, rewards