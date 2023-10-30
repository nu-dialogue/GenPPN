import os
import json
import torch
import random
import string
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from convlab2.nlg import NLG
from system.nlg.gpt2rl.utils import make_act_sequence, make_resp_sequence

class GPT2RLNLG(NLG):
    def __init__(self, model_name_or_path, device) -> None:
        super().__init__()
        self.device = device
        self.gpt2 = GPT2LMHeadModel.from_pretrained(model_name_or_path).to(device)
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name_or_path)

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.lm_task_type = "act_resp"
        self.act_bos_token = "[ACT]"
        self.resp_bos_token = "[RSP]"

    def _make_input_seq(self, system_action):
        act_seq = make_act_sequence(act_bos_token=self.act_bos_token,
                                    action=system_action)
        resp_seq = make_resp_sequence(resp_bos_token=self.resp_bos_token)
        if self.lm_task_type == "act_resp":
            input_seq = act_seq + resp_seq
        else:
            raise ValueError
        return input_seq

    def generate(self, action, ret_w_eos_token=False) -> str:
        # Ref patch
        original_ref_num = None
        made_ref_num = None
        for act in action:
            if act[2] == "Ref":
                original_ref_num = act[3]
                made_ref_num = "".join(random.choices(string.ascii_uppercase + string.digits, k=8))
                act[3] = made_ref_num

        input_seq = self._make_input_seq(system_action=action)
        input_ids = self.tokenizer.encode(input_seq, return_tensors="pt")
        batch_size, query_len = input_ids.size()
        outputs = self.gpt2.generate(
            input_ids=input_ids.to(self.device),
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            max_length=128, num_beams=1, top_k=0,
            top_p=1.0, do_sample=True, temperature=0.8)
        gen_ids = outputs[0, query_len:]
        gen_txt = self.tokenizer.decode(gen_ids)
        if not ret_w_eos_token:
            gen_text = gen_txt.replace(self.tokenizer.eos_token, "")

        # Ref patch
        if original_ref_num is not None:
            gen_text = gen_text.replace(made_ref_num, original_ref_num)
        return gen_text

    def batch_generate(self, batch, ret_w_eos_token=True, **kwargs):
        """
        batch = [
            {"system_action": List[List[str, str, str, str], ...], "system_response": str},
            ...
        ]
        """
        input_txt_list = []
        for b in batch:
            input_seq = self._make_input_seq(system_action=b["system_action"])
            input_txt_list.append(input_seq)
        
        # batch generation
        # https://github.com/huggingface/transformers/pull/7552#issue-497255933
        self.tokenizer.padding_side = "left"
        model_inputs = self.tokenizer(input_txt_list, padding='longest',
                                      return_tensors="pt", return_length=True)
        self.tokenizer.padding_side = "right"
        act_gen_ids = self.gpt2.generate(
            input_ids=model_inputs["input_ids"].to(self.device),
            attention_mask=model_inputs["attention_mask"].to(self.device),
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
            max_length=kwargs["max_length"],
            # min_length=0,
            top_k=0, top_p=1.0, num_beams=1, 
            do_sample=kwargs["do_sample"], temperature=kwargs["temperature"])

        batch_size, query_max_len = model_inputs["input_ids"].size()
        gen_ids = act_gen_ids[:, query_max_len:]
        eos_offset = int(self.tokenizer.eos_token_id == self.tokenizer.pad_token_id)
        gen_batch = []
        for i in range(batch_size):
            query_ids = model_inputs["input_ids"][i, -model_inputs["length"][i]:]

            gen_pad_len = gen_ids[i].eq(self.tokenizer.pad_token_id).sum()
            if gen_pad_len <= eos_offset: # eos_tokenとpad_tokenが同じときは1つ以下
                pad_start_idx = None
            else:
                pad_start_idx = -gen_pad_len + eos_offset
            response_ids = gen_ids[i, :pad_start_idx]
            response_txt = self.tokenizer.decode(response_ids)
            if not ret_w_eos_token:
                response_txt = response_txt.replace(self.tokenizer.eos_token, "")
            
            gen_batch.append({
                "query_ids": query_ids,
                "response_ids": response_ids,
                "response_txt": response_txt
            })
        return gen_batch
