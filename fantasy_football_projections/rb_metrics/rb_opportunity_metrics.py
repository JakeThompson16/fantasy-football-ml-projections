
from functools import lru_cache

import polars as pl
from polars import Int32

from fantasy_football_projections.config import CURRENT_SEASON

from fantasy_football_projections.data_loading.player_data import (
    load_snap_shares,
    load_player_stats,
    load_ff_opportunity_data,
    get_rb_ids,
)

from fantasy_football_projections.rb_metrics.utility import rusher_opportunity_bounds, get_rb_opportunity_cols


@lru_cache(maxsize=None)
def get_rb_snap_shares(*seasons):
    """
    :param seasons: Seasons to get rb snap share data for
    :return polars.df: Snap shares filtered to just RB's
    """
    snap_shares = load_snap_shares(*seasons)
    rb_snap_shares = snap_shares.filter(pl.col("position") == "RB")

    return rb_snap_shares

def rb_opportunity_scores(seasons: list[int] | int)->pl.DataFrame:
    """
    Scores signifying a players rushing/receiving opportunity rate using snap share and carries
    - keys: rushing_opportunity, receiving_opportunity
    :param seasons: Seasons to get opportunity rates for
    :return pl.DataFrame: Number in range [0, 1] where closer to 1 implies more opportunities
    for rushing and receiving (week to week scores can technically exceed 1, but averages over
    many games will not)
    """
    if type(seasons) == int:
        seasons = [seasons]
    rb_ids = get_rb_ids()

    # Loads relevant statistics, filtering each df by player id
    opportunity_data = get_rb_snap_shares(*seasons).filter(pl.col("gsis_id").is_in(rb_ids))
    player_stats = load_player_stats(*seasons).filter(pl.col("player_id").is_in(rb_ids))
    ff_opp = load_ff_opportunity_data(*seasons).select(
        [
            "rush_yards_gained_exp", "rush_touchdown_exp", "rush_first_down_exp",
            "rec_yards_gained_exp", "rec_touchdown_exp", "rec_first_down_exp",
            "receptions_exp", "player_id", "week", "season"
        ]
    ).with_columns(
        pl.col("week").cast(Int32),
        pl.col("season").cast(Int32)
    )

    # Renames week, season cols to avoid polars duplicate error
    player_stats = player_stats.rename({
        "week": "p_week",
        "season": "p_season"
    })
    ff_opp = ff_opp.rename({
        "week": "f_week",
        "season": "f_season"
    })

    # Joins all statistics to one data-frame
    opportunity_data = opportunity_data.join(
        player_stats,
        left_on=["week", "season", "gsis_id"],
        right_on=["p_week", "p_season", "player_id"],
        how="outer"
    )
    opportunity_data = opportunity_data.join(
        ff_opp,
        left_on=["week", "season", "player_id"],
        right_on=["f_week", "f_season", "player_id"],
        how="outer"
    )

    opportunity_data = opportunity_data.filter(pl.col("player_id").is_not_null())
    opportunity_data = opportunity_data.fill_null(0)
    opportunity_data = opportunity_data.sort(["week", "season"])

    # Gets statistical bounds for statistics in current season
    bounds = rusher_opportunity_bounds(CURRENT_SEASON)
    car_hi = bounds["carries_upper"]
    rye_hi = bounds["rush_yards_exp_upper"]
    rtd_hi = bounds["rush_td_exp_upper"]
    rfd_hi = bounds["rush_first_down_exp_upper"]
    tar_hi = bounds["targets_upper"]
    rec_hi = bounds["receptions_exp_upper"]
    reye_hi = bounds["rec_yards_exp_upper"]
    retd_hi = bounds["rec_td_exp_upper"]
    refd_hi = bounds["rec_first_down_exp_upper"]

    # Generates weighted statistics bounding data to [0,1]
    opportunity_data = opportunity_data.with_columns(

        # Rushing stats
        (pl.col("rush_yards_gained_exp") / rye_hi).alias("weighted_rush_yards_exp"),
        (pl.col("rush_touchdown_exp") / rtd_hi).alias("weighted_rush_touchdown_exp"),
        (pl.col("rush_first_down_exp") / rfd_hi).alias("weighted_rush_first_down_exp"),
        (pl.col("carries") / car_hi).alias("weighted_carries"),

        # Receiving stats
        (pl.col("targets") / tar_hi).alias("weighted_targets"),
        (pl.col("rec_yards_gained_exp") / reye_hi).alias("weighted_rec_yards_exp"),
        (pl.col("rec_touchdown_exp") / retd_hi).alias("weighted_rec_touchdown_exp"),
        (pl.col("rec_first_down_exp") / refd_hi).alias("weighted_rec_first_down_exp"),
        (pl.col("receptions_exp") / rec_hi).alias("receptions_exp_upper")
    )

    # Generates rushing opportunity score column for each game
    opportunity_data = opportunity_data.with_columns(

        # Rushing opportunity
        (pl.col("offense_pct") * .30 +
        pl.col("weighted_carries") * .25 +
        pl.col("weighted_rush_yards_exp") * .20 +
        pl.col("weighted_rush_touchdown_exp") * .15 +
        pl.col("weighted_rush_first_down_exp") * .10)
        .alias("rushing_opportunity"),

        # Receiving opportunity
        (pl.col("weighted_targets") * .30 +
         pl.col("receptions_exp_upper") * .25 +
        pl.col("weighted_rec_yards_exp") * .20 +
        pl.col("weighted_rec_touchdown_exp") * .15 +
        pl.col("weighted_rec_first_down_exp") * .10)
        .alias("receiving_opportunity")
    )

    cols = get_rb_opportunity_cols() + ["player_id", "week", "season"]
    return opportunity_data.select(cols)
