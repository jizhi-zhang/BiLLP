from .movie import MovieENV, Movie_Grounding_Model
from .steam import SteamENV, Steam_Grounding_Model
from .amazon import AmazonENV, Amazon_Grounding_Model


def get_envs(name, config, split):
    if name == "movie":
        return MovieENV(config)
    elif name == 'steam':
        return SteamENV(config, split)
    elif name == 'amazon':
        return AmazonENV(config, split)
    else:
        raise ValueError("Unknown env name: {}".format(name))
    
def get_groundingmodel(name, path, config, split):
    if name == "movie":
        return Movie_Grounding_Model(path, config)
    elif name == "steam":
        return Steam_Grounding_Model(path, config)
    elif name == 'amazon':
        return Amazon_Grounding_Model(path, config)
    else:
        raise ValueError("Unknown env name: {}".format(name))