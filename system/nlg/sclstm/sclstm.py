import re
import torch
from convlab2.nlg.sclstm.multiwoz import SCLSTM
from system.nlg.utils import remove_ws_before_punctuation

class WrappedSCLSTM(SCLSTM):
    def __init__(self, device) -> None:
        torch.cuda.set_device(device)
        super().__init__(model_file="https://huggingface.co/ConvLab/ConvLab-2_models/resolve/main/nlg_sclstm_multiwoz.zip",
                         use_cuda=True)
        self.unk_token = "UNK_token"
    
    def generate(self, action):
        self.args["beam_size"] = 1 # greedy decoding
        gen_txt = super().generate(action)
        gen_txt = gen_txt.replace(self.unk_token, "")
        gen_txt = remove_ws_before_punctuation(gen_txt)
        return gen_txt