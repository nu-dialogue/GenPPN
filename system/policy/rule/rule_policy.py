from convlab2.policy.rule.multiwoz import RulePolicy

class WrappedRulePolicy(RulePolicy):
    def __init__(self) -> None:
        super().__init__()