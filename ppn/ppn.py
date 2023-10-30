import torch
from logging import getLogger
from typing import List, Dict, Tuple, Any
from ppn.data_utils import (
    make_history_dicts_from_logs,
    make_prompt_text_from_history_dicts,
    make_no_response_prompt_text_from_history_dicts
)
from transformers import StoppingCriteria, StoppingCriteriaList
from utils import AgentType, PromptSymbol, set_logger

logger = getLogger(__name__)
set_logger(logger)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class StopPhraseCriteria(StoppingCriteria):
    def __init__(self, phrase_ids, min_length=0) -> None:
        self.phrase_ids = phrase_ids
        self.min_length = min_length
        assert self.phrase_ids.ndim == 1

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        if input_ids.shape[-1] <= self.min_length or input_ids.shape[-1] < self.phrase_ids.shape[0]:
            return False
        return (input_ids[:, -self.phrase_ids.shape[0]:].cpu() == self.phrase_ids.cpu()).all()

class PPNNLG:
    def __init__(self, pretrained_name, model, tokenizer, max_context_turns, use_system_da, input_no_response,
                 use_no_rephrasing_keyword, max_generation_tokens, temperature, top_k, top_p, do_sample) -> None:
        self.pretrained_name = pretrained_name
        self.model = model
        self.tokenizer = tokenizer

        self.max_context_turns = max_context_turns
        self.use_system_da = use_system_da
        self.input_no_response = input_no_response
        self.use_no_rephrasing_keyword = use_no_rephrasing_keyword
        self.max_length = 512 # Avoid OOM error in PPO training

        self.sep_text = PromptSymbol.SEPERATOR.value
        self.sep_ids = self.tokenizer.encode(self.sep_text,
                                             add_special_tokens=False,
                                             return_tensors="pt")[0, 1:] # Remove intial whitespace `_`
        
        self.no_rephrasing_keyword = PromptSymbol.NO_REPHRASING.value
        self.no_rephrasing_keyword_ids = self.tokenizer.encode(self.no_rephrasing_keyword,
                                                                add_special_tokens=False,
                                                                return_tensors="pt")[0, 1:] # Remove intial whitespace `_`
        
        # self.user_prefix_ids = self.tokenizer.encode(PromptSymbol.SEPERATOR.value + PromptSymbol.USER.value,
        #                                              add_special_tokens=False, return_tensors="pt")[0, 1:] # Remove intial whitespace `_`
        # self.system_prefix_ids = self.tokenizer.encode(PromptSymbol.SEPERATOR.value + PromptSymbol.SYSTEM.value,
        #                                                add_special_tokens=False, return_tensors="pt")[0, 1:] # Remove intial whitespace `_`
        
        self.generation_kwargs = {
            "max_new_tokens": max_generation_tokens,
            "num_beams": 1,
            "temperature": temperature,
            "top_k": top_k,
            "top_p": top_p,
            "do_sample": do_sample,
            "stopping_criteria": StoppingCriteriaList([
                StopPhraseCriteria(self.sep_ids), # Stop when the seperator is generated
                StopPhraseCriteria(self.no_rephrasing_keyword_ids), # Stop when the no rephrasing keyword is generated
                # StopPhraseCriteria(self.user_prefix_ids), # Stop when the user prefix is generated
                # StopPhraseCriteria(self.system_prefix_ids), # Stop when the system prefix is generated
            ]),
            "eos_token_id": self.tokenizer.eos_token_id,
            "pad_token_id": self.tokenizer.pad_token_id,
        }

    def cutoff_phrase_ids(self, response_ids: torch.Tensor, phrase_ids: torch.Tensor) -> torch.Tensor:
        if (response_ids.shape[0] >= phrase_ids.shape[0]) and \
            (response_ids[-phrase_ids.shape[0]:].cpu() == phrase_ids.cpu()).all():
            response_ids = response_ids[:-phrase_ids.shape[0]]
        return response_ids

    def cutoff_phrase(self, response_text: str, phrase: str) -> str:
        if response_text.endswith(phrase):
            response_text = response_text[:-len(phrase)]
        return response_text

    def postprocess(self, previous_logs: List[Dict[str, Any]], current_log: Dict[str, Any], return_ppn_log: bool = True) -> Tuple[str, Dict[str, torch.Tensor]]:
        # Make context text
        history_dicts = make_history_dicts_from_logs(previous_logs=previous_logs,
                                                    current_log=current_log,
                                                    max_context_turns=self.max_context_turns)
        if not self.input_no_response:
            prompt_text = make_prompt_text_from_history_dicts(pretrained_name=self.pretrained_name,
                                                              history_dicts=history_dicts,
                                                              use_system_da=self.use_system_da,
                                                              use_no_rephrasing_keyword=self.use_no_rephrasing_keyword)
        else:
            prompt_text = make_no_response_prompt_text_from_history_dicts(pretrained_name=self.pretrained_name,
                                                                          history_dicts=history_dicts)

        prompt_ids = self.tokenizer(prompt_text, return_tensors="pt").input_ids[0]
        
        # Generate system response
        output_ids = self.model.generate(input_ids=prompt_ids.unsqueeze(0).to(DEVICE),
                                         **self.generation_kwargs)
        response_ids = output_ids[0, prompt_ids.shape[0]:]
        # response_ids = self.cutoff_phrase_ids(response_ids, self.sep_ids)
        # response_ids = self.cutoff_phrase_ids(response_ids, self.user_prefix_ids)
        # response_ids = self.cutoff_phrase_ids(response_ids, self.system_prefix_ids)

        response_text = self.tokenizer.decode(response_ids, skip_special_tokens=True)
        response_text = self.cutoff_phrase(response_text, self.sep_text) # Remove the seperator at the end

        if self.use_no_rephrasing_keyword and PromptSymbol.NO_REPHRASING.value in response_text:
            ret = (current_log["nlg"]["system_response"],) # Use the original system response
        else:
            ret = (response_text,) # Use the generated PPN response
        
        if return_ppn_log:
            is_training_example = True
            if prompt_ids.shape[0] >= self.max_length: # If the prompt is too long, exclude it from training
                is_training_example = False
            elif response_ids.shape[0] <= 1: # If the response is too short, exclude it from training
                is_training_example = False
            else:
                if prompt_ids.shape[0] + response_ids.shape[0] > self.max_length:
                    response_ids = response_ids[:self.max_length - prompt_ids.shape[0]]
                
            ppn_log = {
                "prompt_ids": prompt_ids, "response_ids": response_ids,
                "prompt_text": prompt_text, "response_text": response_text,
                "is_training_example": is_training_example 
            }
            ret += (ppn_log,)
        return ret
        