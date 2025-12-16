
from lightgbm import Booster
import os

def load_recent_rb_model():
    """
    :return: The most recent rb model that was saved
    """
    path = "fantasy_football_projections/data_loading/models/rb_model.txt"
    loaded_booster = Booster(model_file=path)
    return loaded_booster

def load_recent_wr_model():
    """
    :return: The most recent wr model that was saved
    """
    path = "fantasy_football_projections/data_loading/models/wr_model.txt"
    loaded_booster = Booster(model_file=path)
    return loaded_booster