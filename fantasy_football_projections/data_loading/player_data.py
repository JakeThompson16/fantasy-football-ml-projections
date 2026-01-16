
from functools import lru_cache
from typing import List

import nflreadpy as nfl
import polars as pl


# Caches and returns player stats for seasons
@lru_cache(maxsize=None)
def load_player_stats(*seasons):
    data = nfl.load_player_stats(seasons=list(seasons))
    return data

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

# Caches and returns snap count data
@lru_cache(maxsize=None)
def load_snap_shares(*seasons):
    snap_counts = nfl.load_snap_counts(list(seasons)).select(
        "pfr_player_id",
        "offense_pct",
        "week",
        "season",
        "position"
    )
    # Maps pfr_id to gsis_id
    id_map = get_id_map()
    snap_counts = snap_counts.with_columns(
        pl.col("pfr_player_id").map_elements(lambda x: id_map.get(x))
        .alias("gsis_id")
    )
    return snap_counts

# Caches and returns fantasy football opportunity data
@lru_cache(maxsize=None)
def load_ff_opportunity_data(*seasons):
    ff_data = nfl.load_ff_opportunity(seasons=list(seasons), stat_type="weekly")
    return ff_data

# Caches and returns a list of the ID's of all running backs
@lru_cache(maxsize=None)
def get_rb_ids()->List[str]:
    player_ids = nfl.load_ff_playerids().select("gsis_id", "position")
    player_ids = player_ids.filter(pl.col("position") == "RB")
    return player_ids["gsis_id"].to_list()