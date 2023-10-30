import os
from convlab2.policy.mle.multiwoz import MLEPolicy

class WrappedMLEPolicy(MLEPolicy):
    def __init__(self) -> None:
        super().__init__()