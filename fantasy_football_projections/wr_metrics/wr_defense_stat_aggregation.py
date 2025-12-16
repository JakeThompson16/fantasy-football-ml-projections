
import polars as pl
from fantasy_football_projections.data_loading.player_data import load_pbp_data, load_player_stats


def get_wr_defense_weekly_stats(seasons)->pl.DataFrame:
    """
    :param seasons: Seasons to get defensive weekly stats for
    :return: df of weekly stats grouped by all wr's on an opposing team
    """
    # Gets weekly stats data for defenses
    df = load_player_stats(*seasons)

    # Groups by all wr's on an opposing team
    df = df.group_by(["opponent_team", "week", "season"]).agg(
        pl.col("targets").sum().alias("targets_against"),
        pl.col("receptions").sum().alias("receptions_against"),
        pl.col("receiving_yards").sum().alias("receiving_yards_against"),
        pl.col("receiving_tds").sum().alias("receiving_tds_against"),
        pl.col("receiving_epa").sum().alias("receiving_epa_against"),
        pl.col("racr").sum().alias("racr_against")
    )

    return df

def get_wr_defense_pbp_stats(seasons)->pl.DataFrame:
    """
    :param seasons: Seasons to get defensive pbp data from
    :return: df of stats available in nflreadpy.load_pbp_stats() grouped by game
    per opposing team
    """
    # Gets defensive pbp for defensive metrics
    df = (load_pbp_data(*seasons).filter(
        pl.col("pass_attempt") == 1
    ))
    df = df.with_columns(

        # Red zone targets
        pl.when(pl.col("yardline_100") <= 20)
        .then(1)
        .otherwise(0)
        .alias("redzone_target"),

        # Deep pass attempts
        pl.when(pl.col("air_yards") >= 20)
        .then(1)
        .otherwise(0)
        .alias("big_play_attempt"),

        # Red zone touch downs
        pl.when((pl.col("yardline_100") <= 20) & (pl.col("pass_touchdown") == 1))
        .then(1)
        .otherwise(0)
        .alias("redzone_touchdown"),

        # Deep pass conversions
        pl.when((pl.col("air_yards") >= 20) & (pl.col("complete_pass") == 1))
        .then(1)
        .otherwise(0)
        .alias("big_play_conversion"),

    )

    df = df.group_by(["defteam", "week", "season"]).agg(
        pl.col("yards_after_catch").sum().alias("yards_after_catch_against"),
        pl.col("air_yards").sum().alias("air_yards_against"),
        pl.col("yac_epa").sum().alias("yac_epa_against"),
        pl.col("redzone_target").sum().alias("redzone_targets_against"),
        pl.col("big_play_attempt").sum().alias("big_play_attempts_against"),
        pl.col("redzone_touchdown").sum().alias("redzone_touchdowns_against"),
        pl.col("big_play_conversion").sum().alias("big_play_conversions_against"),
    )

    return df