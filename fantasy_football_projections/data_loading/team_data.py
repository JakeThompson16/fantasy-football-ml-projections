
from functools import lru_cache
import nflreadpy as nfl
import polars as pl
from fantasy_football_projections.data_loading.player_data import load_pbp_data


# Caches and returns team stats for seasons
@lru_cache(maxsize=None)
def load_team_data(*seasons):
    return nfl.load_team_stats(list(seasons))

# Caches and returns offensive pbp data for a team during seasons
@lru_cache(maxsize=None)
def load_team_pbp_data(*seasons, team):
    pbp = load_pbp_data(*seasons)
    team_data = pbp.filter(pl.col("posteam") == team)
    return team_data

# Caches and returns defensive pbp data for a team during seasons
@lru_cache(maxsize=None)
def load_team_def_pbp_data(*seasons, team):
    pbp = load_pbp_data(*seasons)
    team_data = pbp.filter(pl.col("defteam") == team)
    return team_data