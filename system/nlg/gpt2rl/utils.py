import os
import re

def make_act_sequence(act_bos_token, action):
    act_seq = " " + act_bos_token + ", ".join([f"{i}-{d}+{s}*{v}" for i,d,s,v in action])
    return act_seq

def make_resp_sequence(resp_bos_token, txt=None, eos_token=None):
    resp_seq = " " + resp_bos_token
    if txt is not None:
        assert eos_token is not None
        resp_seq += txt.strip() + eos_token
    return resp_seq

def split_act_sequence(seq, act_bos_token, resp_bos_token):
    act_seq, resp_seq = seq.replace(act_bos_token, "").split(resp_bos_token, 1)
    action = [re.split(r'[-\+\*]', a_seq) for a_seq in act_seq.split(", ")]
    return action, resp_seq