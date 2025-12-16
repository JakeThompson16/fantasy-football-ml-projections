
from functools import lru_cache
import nflreadpy as nfl
import polars as pl


# Caches and returns player stats for seasons
@lru_cache(maxsize=None)
def load_player_stats(*seasons):
    return nfl.load_player_stats(list(seasons))

# Caches and returns player stats for seasons by team
@lru_cache(maxsize=None)
def load_player_stats_by_team(*seasons, team):
    player_stats = load_player_stats(*seasons)
    by_team = player_stats.filter(pl.col("team") == team)
    return by_team

# Caches and returns pbp data for seasons
@lru_cache(maxsize=None)
def load_pbp_data(*seasons):
    return nfl.load_pbp(list(seasons))

# Caches and returns rushing/receiving pbp data for a player during seasons
@lru_cache(maxsize=None)
def load_player_pbp_data(*seasons, gsis_id):
    pbp = load_pbp_data(*seasons)
    player_data = pbp.filter(
        (pl.col("rusher_player_id") == gsis_id) |
        (pl.col("receiver_player_id") == gsis_id)
    )
    return player_data

# Caches and returns plays where a player was targeted
@lru_cache(maxsize=None)
def load_player_targets(*seasons, gsis_id):
    pbp = load_pbp_data(*seasons)
    player_data = pbp.filter(pl.col("receiver_player_id") == gsis_id)
    return player_data

# Caches and returns player data
@lru_cache(maxsize=None)
def load_player_data(season):
    players = nfl.load_players()
    players = players.filter(
        (pl.col("last_season") >= season - 1) &  # -1 to account for rookies and retirees
        (pl.col("draft_year") <= season)
    )
    return players

# Caches and returns next-gen stats data
@lru_cache(maxsize=None)
def load_nextgen_wr_data(*seasons):
    stats = nfl.load_nextgen_stats(list(seasons), stat_type="receiving")
    return stats

# Caches and returns next-gen stats for a specific receiver
@lru_cache(maxsize=None)
def load_rec_nextgen_stats(*seasons, gsis_id):
    stats = load_nextgen_wr_data(*seasons)
    stats = stats.filter(pl.col("player_gsis_id") == gsis_id)
    return stats

@lru_cache(maxsize=None)
def get_id_map()->dict:
    """
    :return: Dict mapping pfr_id to gsis_id
    """
    # Loads play ID's df and creates pfr_id->gsis_id map
    player_ids = nfl.load_ff_playerids().select("pfr_id", "gsis_id")
    id_map = dict(zip(player_ids["pfr_id"].to_list(), player_ids["gsis_id"].to_list()))

    return id_map
