from fantasy_football_projections.data_loading.player_data import load_player_data
from fantasy_football_projections.data_loading.team_data import load_team_def_pbp_data
import polars as pl

from fantasy_football_projections.wr_modeling.utility import get_defense_cols


def group_defensive_stats(df)->pl.DataFrame:
    """
    :param pl.DataFrame df: Player weekly stats data-frame
    :return: Data-frame where each teams defensive totals allowed are aggregated
    """
    df = df.group_by(["opponent_team", "week", "season"]).agg(
        pl.col("targets").sum().alias("targets_against"),
        pl.col("receptions").sum().alias("receptions_against"),
        pl.col("receiving_yards").sum().alias("receiving_yards_against"),
        pl.col("receiving_tds").sum().alias("receiving_tds_against"),
        pl.col("receiving_epa").sum().alias("receiving_epa_against"),
        pl.col("racr").sum().alias("racr_against")
    )

    return df

def generate_defensive_averages(defense_df, training=True)->pl.DataFrame:
    """
    :param pl.DataFrame defense_df: Data-frame containing cols seen in def_cols (within this method)
    :param bool training: True if averages are for training, False otherwise (True means most
    recent game will not be included in averages)
    :return: Data-frame of average stat against over 6 weeks, and season
    """
    defense_df = defense_df.sort(["opponent_team", "season", "week"])

    def_cols = get_defense_cols()

    defense_df = defense_df.fill_null(0)

    # Rolling defense average over 6 games
    for col in def_cols:
        defense_df = defense_df.with_columns(
            pl.col(col)
            .rolling_mean(window_size=6, min_periods=1)
            .over(["opponent_team", "season"])
            .shift(1)
            .alias(f"{col}_6g_avg")
        )

    # Defense average's over whole season
    for col in def_cols:
        defense_df = defense_df.with_columns(
            (
                    pl.col(col).cum_sum().over(["opponent_team", "season"])
                    / pl.col("week").cum_count().over(["opponent_team", "season"])
            )
            .shift(1)
            .alias(f"{col}_season_avg")
        )

    return defense_df