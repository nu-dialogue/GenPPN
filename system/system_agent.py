from copy import deepcopy
from convlab2.dialog_agent.agent import PipelineAgent
from utils import get_device

def build_nlu(nlu_name, nlu_config_file):
    if not nlu_name:
        return None
    elif nlu_name == "bert_nlu":
        from system.nlu.bert.bert_nlu import WrappedBERTNLU
        return WrappedBERTNLU(config_file=nlu_config_file)
    elif nlu_name == "bert_nlu_ctx":
        from system.nlu.bert.bert_nlu import WrappedBERTNLU
        return WrappedBERTNLU(config_file=nlu_config_file)
    else:
        raise NotImplementedError(nlu_name)

def build_dst(dst_name):
    if dst_name == "rule_dst":
        from system.dst.rule.rule_dst import WrappedRuleDST
        return WrappedRuleDST()
    else:
        raise NotImplementedError(dst_name)

def build_policy(policy_name):
    if policy_name == "rule_policy":
        from system.policy.rule.rule_policy import WrappedRulePolicy
        return WrappedRulePolicy()
    elif policy_name == "mle_policy":
        from system.policy.mle.mle_policy import WrappedMLEPolicy
        return WrappedMLEPolicy()
    else:
        raise NotImplementedError(policy_name)

def build_nlg(nlg_name):
    if not nlg_name:
        return None
    elif nlg_name == "template_nlg":
        from system.nlg.template.template_nlg import WrappedTemplateNLG
        return WrappedTemplateNLG(mode="manual")
    elif nlg_name == "retrieval_nlg":
        from system.nlg.template.template_nlg import WrappedTemplateNLG
        return WrappedTemplateNLG(mode="auto_manual")
    elif nlg_name == "sclstm_nlg":
        from system.nlg.sclstm.sclstm import WrappedSCLSTM
        return WrappedSCLSTM(device=get_device())
    elif nlg_name == "scgpt_nlg":
        from system.nlg.scgpt.scgpt import SCGPTNLG
        return SCGPTNLG(model_name_or_path="ohashi56225/scgpt-multiwoz-sys", device=get_device())
    elif nlg_name == "gpt2rl_nlg":
        from system.nlg.gpt2rl.gpt2rl import GPT2RLNLG
        return GPT2RLNLG(model_name_or_path="ohashi56225/antor-full_bert_nlu-no_noise", device=get_device())
    else:
        raise NotImplementedError(nlg_name)

class SystemAgent(PipelineAgent):
    def __init__(self, nlu_name, nlu_config_file, dst_name, policy_name, nlg_name, ppn_nlg=None) -> None:
        nlu = build_nlu(nlu_name, nlu_config_file)
        dst = build_dst(dst_name)
        policy = build_policy(policy_name)
        nlg = build_nlg(nlg_name)
        super().__init__(nlu=nlu, dst=dst, policy=policy, nlg=nlg, name="sys")

        self.ppn_nlg = ppn_nlg
        self.ppns = []
        if ppn_nlg is not None:
            self.ppns.append("ppn_nlg")
        
    def init_session(self):
        super().init_session()

        self.turn_count = 0
        self.log = []
        self.ppn_log = []
        self.is_training_example = []

    def response(self, observation):
        log_ = {
            "turn_id": self.turn_count,
            "user_utterance": observation,
        }
        ppn_log_ = {
            "turn_id": self.turn_count,
        }

        # Update history
        if self.dst is not None:
            self.dst.state['history'].append([self.opponent_name, observation]) # [['sys', sys_utt], ['user', user_utt],...]
        self.history.append([self.opponent_name, observation])

        # NLU
        if self.nlu is not None:
            self.input_action = self.nlu.predict(observation, context=[x[1] for x in self.history[:-1]])
            log_["nlu"] = {"user_action": deepcopy(self.input_action)} # get rid of reference problem
        else:
            self.input_action = observation
        self.input_action = deepcopy(self.input_action) # get rid of reference problem

        # DST
        if self.dst is not None:
            self.dst.state['user_action'] = self.input_action
            state = self.dst.update(self.input_action)
            log_["dst"] = {"dialogue_state": deepcopy(state)}
        else:
            state = self.input_action
        state = deepcopy(state) # get rid of reference problem

        # Policy
        self.output_action = deepcopy(self.policy.predict(state)) # get rid of reference problem
        log_["policy"] = {"system_action": deepcopy(self.output_action)}
        
        # NLG
        if self.nlg is not None:
            system_response = self.nlg.generate(self.output_action)
            log_["nlg"] = {"system_response": system_response}
        else:
            system_response = self.output_action

        # PPN_NLG
        if self.ppn_nlg is not None:
            system_response, ppn_nlg_log_ = self.ppn_nlg.postprocess(previous_logs=self.log,
                                                                     current_log=log_)
            log_["ppn_nlg"] = {"system_response": system_response}
            ppn_log_["ppn_nlg"] = ppn_nlg_log_
        log_["system_response"] = system_response
        
        # Update log, dst's history, and history
        self.log.append(log_)
        if self.dst is not None:
            self.dst.state['history'].append([self.name, system_response])
            self.dst.state['system_action'] = self.output_action
        self.history.append([self.name, system_response])


        # Save ppn's log
        self.ppn_log.append(ppn_log_)

        # Check if this is a training example
        if not self.dst.state["terminated"]:
            is_training_example = all([ppn_log_[ppn_name]["is_training_example"] for ppn_name in self.ppns])
        else:
            is_training_example = False # We don't use ppns' last action
        self.is_training_example.append(is_training_example)

        # Increment turn count
        self.turn_count += 1
        
        return system_response