from convlab2.dst.rule.multiwoz import RuleDST

class WrappedRuleDST(RuleDST):
    def __init__(self) -> None:
        super().__init__()