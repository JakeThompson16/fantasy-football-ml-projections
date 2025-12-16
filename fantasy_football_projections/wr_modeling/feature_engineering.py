import numpy

from fantasy_football_projections.wr_metrics.universal_averages import max_depth_of_target, max_reception_per_game, \
    select_wanted_cols
from fantasy_football_projections.wr_metrics.wr_defense_stat_aggregation import get_wr_defense_pbp_stats, \
    get_wr_defense_weekly_stats
from fantasy_football_projections.wr_metrics.wr_defensive_metrics import generate_defensive_averages
from fantasy_football_projections.wr_metrics.wr_offensive_metrics import generate_offensive_averages
import polars as pl

from fantasy_football_projections.wr_metrics.wr_stat_aggregation import get_wr_snap_counts, get_wr_weekly_stats, \
    get_wr_nextgen_stats, get_wr_pbp_stats_weekly


def get_training_df(seasons) -> pl.DataFrame:
    """
    :param int[] seasons: Seasons to base data on
    :return: Data-frame tailored for training wr points prediction model
    """

    snap_counts = get_wr_snap_counts(seasons)

    # Gets df with all wr games (at least one target)
    player_stats = get_wr_weekly_stats(seasons)

    # Joins player_stats and snap count
    player_stats = player_stats.join(
        snap_counts,
        left_on=["player_id", "week", "season"],
        right_on=["gsis_id", "week", "season"],
        how="left"
    )

    # Filters out games where players played more than 20% of snaps
    player_stats = player_stats.filter(pl.col("offense_pct") > .05)

    nextgen_stats = get_wr_nextgen_stats(seasons)

    # Joins next-gen stats and player_stats together
    player_stats = player_stats.join(
        nextgen_stats,
        left_on=["player_id", "season", "week"],
        right_on=["player_gsis_id", "season", "week"],
        how="left"
    )

    # Gets pbp data for all wr games over season
    pbp_stats = get_wr_pbp_stats_weekly(seasons)

    # Joins pbp_stats to player_stats
    df = player_stats.join(
        pbp_stats,
        left_on=["player_id", "week", "season"],
        right_on=["player_id", "week", "season"],
        how="left"
    )

    # Generates previous relevant averages
    df = generate_offensive_averages(df)

    defense_stats = get_wr_defense_weekly_stats(seasons)

    # Gets defensive pbp for defensive metrics
    defense_pbp = get_wr_defense_pbp_stats(seasons)

    # Joins pbp and weekly stats
    defense_df = defense_stats.join(
        defense_pbp,
        left_on=["opponent_team", "week", "season"],
        right_on=["defteam", "week", "season"],
        how="left"
    )

    # Gets averages for all defenses
    defense_df = generate_defensive_averages(defense_df)

    # Joins df and defense_df
    df = df.join(
        defense_df,
        left_on=["opponent_team", "week", "season"],
        right_on=["opponent_team", "week", "season"]
    )

    # Selects only cols needed for features
    df = select_wanted_cols(df)
    return df

def write_training_df_to_parquet(seasons):
    """
    Generates a training data-frame and writes it to a parquet file
    utils->construct_dataset_location->construct_rb_dataset_location() w/ same args for file location
    :param int[] seasons: Seasons to generate training df from
    :return: The location of training df
    """
    path = "fantasy_football_projections/data_loading/datasets/wr_training_ds.parquet"

    df = get_training_df(seasons)
    df.write_parquet(path)

    return path

def training_df_cols():
    """
    :return: The columns returned by get_training_df()
    """
    ...

def generate_auxiliary_features(training_df, ppr=1):
    """
    Generates auxiliary features for training/predicting
    :param pl.DataFrame training_df: Training df as returned by get_training_df()
    :param float ppr: Points per reception
    :return: df with new features
    """
    df = training_df
    for window in ["_3g_avg", "_6g_avg", "_season_avg"]:
        df = df.with_columns(

            # Average depth of target
            (pl.col(f"air_yards_targeted{window}") / pl.col(f"targets{window}"))
            .alias(f"avg_depth_of_target{window}"),

            # Big play conversion rate
            (pl.col(f"big_play_conversions{window}") / pl.col(f"big_play_attempts{window}"))
            .alias(f"big_play_conversion_rate{window}"),

            # Yards per target
            (pl.col(f"receiving_yards{window}") / pl.col(f"targets{window}"))
            .alias(f"yards_per_target{window}")
        )


        df = df.with_columns(
            # Receiver quality score
            ((pl.col(f"avg_depth_of_target{window}") / pl.col("season").map_elements(max_depth_of_target, return_dtype=pl.Float64)) +
            (pl.col(f"receptions{window}") / pl.col("season").map_elements(max_reception_per_game, return_dtype=pl.Float64)))
            .alias(f"receiver_quality_score{window}"),

            # Boom score
            (pl.col(f"receiving_tds{window}") * 4
             + pl.col(f"redzone_targets{window}")
             + pl.col(f"big_play_conversion_rate{window}") * 3)
            .alias(f"boom_score{window}"),

            # Target quality
            (pl.col(f"avg_depth_of_target{window}") +
             (pl.col(f"avg_separation{window}")**2))
             .alias(f"avg_target_quality{window}"),

            # Weighted target score
            ((pl.col(f"target_share{window}") + pl.col(f"air_yards_share{window}"))*1.5)
            .alias(f"weighted_target_score{window}"),

            # Target value added
            (pl.col(f"receiving_epa{window}") / pl.col(f"targets{window}"))
            .alias(f"target_value_added{window}"),

            # Yard Opportunity Capitalization Score
            (pl.col(f"catch_percentage{window}")*
             pl.col(f"avg_depth_of_target{window}") +
             (pl.col(f"receiving_yards_after_catch{window}")))
            .alias(f"yard_opportunity_capitalization{window}"),
        )

    for window in ["_6g_avg", "_season_avg"]:
        df = df.with_columns(

            # RACR differential
            (pl.col(f"racr{window}") - pl.col(f"racr_against{window}"))
            .alias(f"racr_differential{window}"),

            # Receiving EPA differential
            (pl.col(f"receiving_epa{window}") - pl.col(f"receiving_epa_against{window}"))
            .alias(f"rec_epa_differential{window}")
        )

    df = df.select(["fantasy_points_ppr"] + features())
    return df

def build_feature_df(training_df, ppr=1):
    """
    :param pl.DataFrame training_df: Training data-frame
    :param float ppr: Points per reception
    :return: Data-frame with only metrics relevant to training
    """
    df = generate_auxiliary_features(training_df, ppr)
    return df

def features():
    """
    :return: A list of features used
    """
    window_amt_3 = ["receiving_yards", "receiving_air_yards",
                 "receiving_yards_after_catch", "receiving_epa", "racr",
                 "target_share", "air_yards_share", "wopr", "avg_separation",
                 "catch_percentage", "avg_yac_above_expectation", "receiving_first_downs",
                 "comp_yac_epa", "fantasy_points_ppr", "redzone_targets",
                 "offense_pct", "avg_depth_of_target", "receiver_quality_score",
                 "big_play_conversion_rate", "boom_score", "avg_target_quality", "weighted_target_score",
                 "target_value_added", "yard_opportunity_capitalization", "yards_per_target"]

    window_amt_2 = ["targets_against", "receptions_against", "receiving_yards_against", "receiving_tds_against",
                "receiving_epa_against", "racr_against", "yards_after_catch_against", "air_yards_against",
                "yac_epa_against", "redzone_targets_against", "big_play_attempts_against",
                "redzone_touchdowns_against", "big_play_conversions_against", "racr_differential",
                "rec_epa_differential"]

    window_amt_1 = ["targets", "receptions"]

    f = []
    for window in ["3g_avg", "6g_avg", "season_avg"]:
        f += [f"{col}_{window}" for col in window_amt_3]
    for window in ["6g_avg", "season_avg"]:
        f += [f"{col}_{window}" for col in window_amt_2]
    for col in window_amt_1:
        f += [f"{col}_season_avg"]

    return f