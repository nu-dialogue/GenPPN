from os.path import join, dirname, abspath, isdir
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from convlab2.nlg.scgpt.utils import tuple2seq
from convlab2.nlg import NLG

from system.nlg.utils import remove_ws_before_punctuation

class SCGPTNLG(NLG):
    def __init__(self, model_name_or_path, device) -> None:
        self.device = device
        self.model = GPT2LMHeadModel.from_pretrained(model_name_or_path).to(device)
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name_or_path)

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def generate(self, action, ret_w_eos_token=False):
        raw_text = tuple2seq(action)
        raw_text += " &"
        input_ids = self.tokenizer.encode(raw_text, return_tensors="pt",
                                          add_special_tokens=False)
        batch_size, input_len = input_ids.size()
        outputs = self.model.generate(
            input_ids=input_ids.to(self.device),
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            max_length=128, num_beams=1, top_k=0,
            top_p=1.0, do_sample=True, temperature=0.8 # We found that sampling with temp=0.8 is much better than greedy search
        )
        gen_ids = outputs[0, input_len:]
        text = self.tokenizer.decode(gen_ids, clean_up_tokenization_spaces=True)
        text = text.split('& ')[-1]
        if not ret_w_eos_token:
            text = text[: text.find(self.tokenizer.eos_token) if self.tokenizer.eos_token else None]
        text = remove_ws_before_punctuation(text)
        return text