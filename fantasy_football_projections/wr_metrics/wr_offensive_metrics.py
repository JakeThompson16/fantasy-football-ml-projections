
from functools import lru_cache
import polars as pl
import nflreadpy as nfl

from fantasy_football_projections.wr_modeling.utility import get_stat_cols


@lru_cache(maxsize=None)
def player_id_map():
    player_ids = nfl.load_ff_playerids().select("pfr_id", "gsis_id")
    id_map = dict(zip(player_ids["pfr_id"].to_list(), player_ids["gsis_id"].to_list()))
    return id_map

def generate_offensive_averages(df, training=True)->pl.DataFrame:
    """
    :param pl.DataFrame df: Data-frame containing cols seen in stat_cols (within this method)
    :param bool training: True if averages are for training, False otherwise (True means most
    recent game will not be included in averages)
    :return:
    """
    shift = 1 if training else 0
    df = df.sort(["player_id", "season", "week"])
    df = df.fill_null(0)

    stat_cols = get_stat_cols()


    # Rolling average over 3 games
    for col in stat_cols:
        df = df.with_columns(
            pl.col(col)
            .rolling_mean(window_size=3, min_periods=1)
            .over(["player_id", "season"])
            .shift(shift)
            .alias(f"{col}_3g_avg")
        )

    # Rolling average over 6 games
    for col in stat_cols:
        df = df.with_columns(
            pl.col(col)
            .rolling_mean(window_size=6, min_periods=4)
            .over(["player_id", "season"])
            .shift(shift)
            .alias(f"{col}_6g_avg")
        )

    # Average over whole season
    for col in stat_cols:
        df = df.with_columns(
            (
            pl.col(col).cum_sum().over(["player_id", "season"])
            / pl.col("week").cum_count().over(["player_id", "season"])
            )
            .shift(shift)
            .alias(f"{col}_season_avg")
        )

    return df







