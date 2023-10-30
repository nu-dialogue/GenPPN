from enum import Enum

class DistributedEnvType(Enum):
    """Choices for the Distributed Environment type."""
    DEFAULT = "default"
    MPI = "mpi"

class AgentType(Enum):
    """Choices for the agent type."""
    USER = "user"
    SYSTEM = "system"

class PromptSymbol(Enum):
    """Identifier prefix in context text."""
    SEPERATOR = "\n"
    USER = "Customer:"
    USER_DA = "Customer Action:"
    SYSTEM = "Chatbot:"
    SYSTEM_DA = "Chatbot Action:"
    NO_REPHRASING = "No rephrasing is necessary"

class RewardFactor(Enum):
    """Choices for the reward factor."""
    INFORM_F1 = "inform_f1"
    BOOK_RATE = "book_rate"
    SYSTEM_DA_F1 = "system_da_f1"
    SYSTEM_DA_F1_WITH_DISTANCE_PENALTY = "system_da_f1_with_distance_penalty"
    BELIEF_ACCURACY = "belief_accuracy"
    FAIL_PENALTY = "fail_penalty"
    USER_DA_CHANGE = "user_da_change"
    USER_DA_CHANGE_WITH_DISTANCE_PENALTY = "user_da_change_with_distance_penalty"
    TASK_SUCCESS = "task_success"
    TASK_SUCCESS_WITH_DISTANCE_PENALTY = "task_success_with_distance_penalty"
    INFORM_BOOK_MATCH = "inform_book_match"
    INFORM_BOOK_MATCH_WITH_DISTANCE_PENALTY = "inform_book_match_with_distance_penalty"
    INFORM_BOOK_MATCH_WITH_USER_DA_CHANGE = "inform_book_match_with_user_da_change"
    INFORM_BOOK_MATCH_WITH_SYSTEM_DA_F1 = "inform_book_match_with_system_da_f1"
    INFORM_BOOK_MATCH_WITH_REQT_GOAL_ACCURACY = "inform_book_match_with_reqt_goal_accuracy"
    DA_MEAN = "da_mean"
    DA_CONTRIBUTION_MEAN = "da_contribution_mean"
    DA_CONTRIBUTION_ABSMAX = "da_contribution_absmax"
