
from functools import lru_cache
import polars as pl
import nflreadpy as nfl
from fantasy_football_projections.data_loading.player_data import load_player_stats, load_pbp_data

def get_wr_snap_counts(seasons)->pl.DataFrame:
    """
    :param int[] seasons: Seasons to get snap counts from
    :return: df of relevant snap count information (includes gsis_id)
    """
    # Loads play ID's df and creates pfr_id->gsis_id map
    player_ids = nfl.load_ff_playerids().select("pfr_id", "gsis_id")
    id_map = dict(zip(player_ids["pfr_id"].to_list(), player_ids["gsis_id"].to_list()))

    # Loads snap counts, adds gsis id column
    snap_counts = nfl.load_snap_counts(seasons).select(
        "pfr_player_id",
        "offense_pct",
        "week",
        "season"
    )
    snap_counts = snap_counts.with_columns(
        pl.col("pfr_player_id").map_elements(lambda x: id_map.get(x))
        .alias("gsis_id")
    )

    return snap_counts

def get_wr_weekly_stats(seasons)->pl.DataFrame:
    """
    :param int[] seasons: Seasons to get weekly stats from
    :return: Sorted df of wr weekly stats from nflreadpy.load_player_stats(seasons)
    """
    # Gets df with all wr games (at least one target)
    player_stats = (load_player_stats(*seasons).filter(
        (pl.col("position") == "WR"))
        .sort(["player_id", "season", "week"])
    )

    return player_stats

def get_wr_nextgen_stats(seasons)->pl.DataFrame:
    """
    :param seasons: Seasons to get nextgen stats from
    :return: Sorted df of weekly wr nextgen stats from nflreadpy.load_nextgen_stats(seasons)
    """
    # Gets all nextgen wr games over seasons
    nextgen_stats = (nfl.load_nextgen_stats(seasons, stat_type="receiving").filter(
        (pl.col("player_position") == "WR")
    )).sort(["player_gsis_id", "season", "week"])

    return nextgen_stats

def get_wr_pbp_stats_weekly(seasons):
    """
    :param seasons: Seasons to get pbp stats from
    :return: df of stats from nflreadpy.load_pbp_stats(seasons) grouped by game+gsis id
    """

    # Gets pbp data for all wr games over season
    pbp_stats = (load_pbp_data(*seasons))
    pbp_stats = (pbp_stats
    .sort("receiver_player_id", "week", "season")
    .with_columns(

        # Red zone targets
        (pl.when(pl.col("yardline_100") <= 20)
         .then(1)
         .otherwise(0)
         .alias("redzone_target")),

        # Big play attempts
        pl.when(pl.col("air_yards") >= 20)
        .then(1)
        .otherwise(0)
        .alias("big_play_attempt"),

        # Red zone tds
        pl.when((pl.col("yardline_100") <= 20) & (pl.col("pass_touchdown") == 1))
        .then(1)
        .otherwise(0)
        .alias("redzone_touchdown"),

        # Big play conversions
        pl.when((pl.col("air_yards") >= 20) & (pl.col("complete_pass") == 1))
        .then(1)
        .otherwise(0)
        .alias("big_play_conversions"),

        # Renames receiver_player_id
        pl.col("receiver_player_id").alias("player_id")
    ))

    # Aggregates pbp stats to per game stats
    pbp_stats = pbp_stats.group_by(["player_id", "week", "season"]).agg(
        (pl.col("redzone_target")).sum().alias("redzone_targets"),
        (pl.col("big_play_attempt")).sum().alias("big_play_attempts"),
        (pl.col("comp_yac_epa").sum()).alias("comp_yac_epa"),
        (pl.col("redzone_touchdown")).sum().alias("redzone_touchdowns"),
        (pl.col("big_play_conversions")).sum().alias("big_play_conversions"),
        (pl.col("air_yards")).sum().alias("air_yards_targeted")
    )

    return pbp_stats