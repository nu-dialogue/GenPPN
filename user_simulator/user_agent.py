from convlab2.dialog_agent import PipelineAgent

def build_nlu(nlu_name, nlu_config_file):
    if nlu_name == "bert_nlu":
        from user_simulator.nlu.bert.bert_nlu import WrappedBERTNLU
        return WrappedBERTNLU(config_file=nlu_config_file)
    elif nlu_name == "bert_nlu_ctx":
        from user_simulator.nlu.bert.bert_nlu import WrappedBERTNLU
        return WrappedBERTNLU(config_file=nlu_config_file)
    else:
        raise NotImplementedError(nlu_name)

def build_policy(policy_name, max_turn, max_initiative):
    if policy_name == "rule_policy":
        from user_simulator.policy.agenda_policy import AgendaPolicy
        return AgendaPolicy(max_turn=max_turn, max_initiative=max_initiative)
    else:
        raise NotImplementedError(policy_name)

def build_nlg(nlg_name):
    if not nlg_name:
        return None
    elif nlg_name == "template_nlg":
        from user_simulator.nlg.template_nlg import WrappedTemplateNLG
        return WrappedTemplateNLG(mode="manual")
    elif nlg_name == "retrieval_nlg":
        from user_simulator.nlg.template_nlg import WrappedTemplateNLG
        return WrappedTemplateNLG(mode="auto_manual")

class UserAgent(PipelineAgent):
    def __init__(self, nlu_name, nlu_config_file, policy_name, nlg_name, max_turn, max_initiative):
        self.name = 'user'
        self.opponent_name = 'sys'

        self.nlu = build_nlu(nlu_name, nlu_config_file)
        self.dst = None
        self.policy = build_policy(policy_name, max_turn, max_initiative)
        self.nlg = build_nlg(nlg_name)

        self.turn_count = 0
        self.history = []
        self.log = []

    def init_session(self, ini_goal=None):
        self.turn_count = 0
        self.history = []
        self.log = []

        self.nlu.init_session()
        self.policy.init_session(ini_goal=ini_goal)
        self.nlg.init_session()

    def response(self, observation):
        system_response = observation

        user_utterance = super().response(system_response)
        system_action = self.input_action.copy()
        user_action = self.output_action.copy()

        self.log.append({
            "turn_id": self.turn_count,
            "system_response": system_response,
            "nlu": {"system_action": system_action},
            "policy": {"user_action": user_action},
            "nlg": {"user_utterance": user_utterance}
        })
        self.turn_count += 1
        return user_utterance