from enum import Enum, auto


class CheetaState(Enum):
    Reached = auto()
    InProgress = auto()
    ApproachJointLimits = auto()
    Fallen = auto()
    Timeout = auto()
    OutOfBounds = auto()
    Undefined = auto()
