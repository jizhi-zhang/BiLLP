from .hotpotqa import HotpotQATask
from .strategyqa import StrategyQATask
from .mmlu import MMLUTask
from .bamboogle import BamboogleTask
from .movie import MovieTask
from .steam import SteamTask
from .amazon import AmazonTask

def get_task(name, split):
    if name == "hotpotqa":
        return HotpotQATask(split)
    elif name == "strategyqa":
        return StrategyQATask(split)
    elif name == "mmlu":
        return MMLUTask(split)
    elif name == "bamboogle":
        return BamboogleTask(split)
    elif name == "movie":
        return MovieTask(split)
    elif name == "steam":
        return SteamTask(split)
    elif name == "amazon":
        return AmazonTask(split)
    else:
        raise ValueError("Unknown task name: {}".format(name))
    
