
import nflreadpy as nfl
import polars as pl

from fantasy_football_projections.data_loading.player_data import get_id_map
from fantasy_football_projections.rb_metrics.rb_defensive_metrics import rb_defense_metrics, rb_defense_metrics_cols
from fantasy_football_projections.rb_metrics.rb_offenseive_metrics import opportunity_capitalization_stats, team_opportunities_provided, \
    opportunity_capitalization_stats_cols, team_opportunities_provided_cols
from fantasy_football_projections.utils.constrcut_dataset_location import make_file_path


def get_training_df(seasons, off_game_amt, def_game_amt, ppr=1):
    """
    :param int[] seasons: Seasons to train on
    :param int off_game_amt: amount of games to base averages on for offensive metrics
    :param int def_game_amt: amount of games to base averages on for defensive metrics
    :param float ppr: points per reception
    :return: data-frame with every rb game fpoints during seasons and their averages entering the game
    """

    # Imports rb snap logs
    id_map = get_id_map()

    # Imports rb snap logs
    rb_weekly_snaps = nfl.load_snap_counts(seasons).filter(
        pl.col("position") == "RB"
    ).with_columns(
        pl.col("pfr_player_id").map_elements(lambda x: id_map.get(x))
        .alias("gsis_id")
    )

    rb_snap_logs = rb_weekly_snaps.with_columns(

        # Average snap share over last 2 games
        pl.col("offense_pct")
        .rolling_mean(window_size=3, min_periods=1)
        .over(["gsis_id", "season"])
        .shift(1)
        .alias("snap_share_3g_avg")
    )


    # Gets rb weekly stats and adds an opportunities col
    rb_weekly_stats = ((nfl.load_player_stats(seasons).filter(
        pl.col("position") == "RB")).with_columns(
        (pl.col("carries") + pl.col("targets"))
        .alias("opportunities")
    ))

    # Joins rb_weekly_stats and rb_snap_logs
    rb_weekly_stats = rb_weekly_stats.join(
        rb_snap_logs,
        left_on=["player_id", "week", "season"],
        right_on=["gsis_id", "week", "season"],
        how="inner"
    )

    rb_weekly_stats = rb_weekly_stats.fill_null(0)
    rb_weekly_stats = rb_weekly_stats.filter(
        pl.col("offense_pct") > .05
    )

    print(rb_weekly_stats.describe())

    # Constructs dict gsis_id:player_name
    gsis_to_name = dict(zip(rb_weekly_stats["player_id"].to_list(), rb_weekly_stats["player_name"].to_list()))

    # Constructs dictionary where each (week, season, team, opponent) key points to a list of
    # tuples (gsis_id, carries+targets) sorted by carries+targets
    rb_dict = {
        (row["week"], row["season"], row["team"], row["opponent_team"]):
            sorted(zip(row["gsis_id"], row["snap_share_3g_avg"], row["fantasy_points_ppr"]), key=lambda x: -x[1])
        for row in rb_weekly_stats.group_by(["week", "season", "team", "opponent_team"]).agg([
            pl.col("player_id").alias("gsis_id"),
            pl.col("snap_share_3g_avg").alias("snap_share_3g_avg"),
            pl.col("opportunities").alias("opportunities"),
            pl.col("fantasy_points_ppr").alias("fantasy_points_ppr")
        ]).to_dicts()
    }

    # List to store rows of df
    rows = []
    i = 0
    # Iterates over each game in rb_dict
    for key, rb_list in rb_dict.items():
        week, season, team, opponent = key
        count = 1
        if i%75 == 0:
            print(f"Computed {round(100*i/len(rb_dict), 2)}% of dictionary")
        i+=1
        # Adds row for each rb1, rb2, rb3
        for gsis_id, opportunities, fpoints in rb_list:

            # Dict for situational and id properties
            misc = {
                "gsis_id": gsis_id,
                "player_name": gsis_to_name[gsis_id],
                "team": team,
                "opponent_team": opponent,
                "week": week,
                "season": season,
                "depth_chart_position": count
            }

            # Constructs dicts for each relevant statistic
            fpoints_dict = {"fantasy_points_ppr": fpoints}
            ocs_dict = opportunity_capitalization_stats(gsis_id, week, season, off_game_amt, ppr)
            top_dict = team_opportunities_provided(team, count, week, season, off_game_amt)
            rdm_dict = rb_defense_metrics(opponent, week, season, def_game_amt, ppr)

            if ocs_dict is None or top_dict is None or rdm_dict is None:
                count += 1
                continue

            row = {**misc, **ocs_dict, **top_dict, **rdm_dict, **fpoints_dict}
            rows.append(row)

            # Breaks after logging rb1 rb2 rb3
            if count == 2:
                break
            count += 1

    # Creates df out of the rows
    df = pl.DataFrame(rows)
    return df

def write_training_df_to_parquet(seasons, off_game_amt, def_game_amt, ppr=1):
    """
    Generates a training data-frame and writes it to a parquet file
    utils->construct_dataset_location->construct_rb_dataset_location() w/ same args for file location
    :param int[] seasons: Seasons to generate training df from
    :param int off_game_amt: amount of games to base averages on for offensive metrics
    :param int def_game_amt: amount of games to base averages on for defensive metrics
    :param float ppr: points per reception
    :return: The location of training df
    """
    location = make_file_path("RB", seasons, off_game_amt, def_game_amt, ppr)

    df = get_training_df(seasons, off_game_amt, def_game_amt, ppr)
    df.write_parquet(location)

    return location


def training_df_cols():
    """
    :return: The columns returned by get_training_df()
    """
    r = ["gsis_id", "player_name", "team", "opponent_team",
         "week", "season", "depth_chart_position"]
    r += opportunity_capitalization_stats_cols()
    r += team_opportunities_provided_cols()
    r += rb_defense_metrics_cols()
    r += ["fantasy_points_ppr"]
    return r

def generate_auxiliary_features(df):
    """
    Constructs different features that are relevant to training
    :param pl.DataFrame df: Current training data-frame
    :return: df with new columns
    """
    df = df.with_columns(
        (pl.col("rushing_capitalization_score") - pl.col("rushing_capitalization_score_against"))
        .alias("weighted_rushing_capitalization_score"),
        (pl.col("receiving_capitalization_score") - pl.col("receiving_capitalization_score_against"))
        .alias("weighted_receiving_capitalization_score"),
        ((pl.col("fpoints_per_carry") * pl.col("rushes_per_game")) +
         (pl.col("fpoints_per_target") * pl.col("targets_per_game")))
        .alias("expected_fpoints_scored"),
        ((pl.col("fpoints_per_carry_against") * pl.col("rushes_per_game")) +
         (pl.col("fpoints_per_target_against") * pl.col("targets_per_game")))
        .alias("expected_fpoints_allowed"),
        (pl.col("fpoints_per_game") / pl.col("opportunities_per_game"))
        .alias("fpoints_per_opportunity")
    )
    df = df.with_columns(
        (pl.col("expected_fpoints_scored") - pl.col("expected_fpoints_allowed"))
        .alias("expected_fpoints_differential")
    )
    return df

def build_feature_df(training_df):
    """
    :param pl.DataFrame training_df: Training data-frame
    :return: Data-frame containing only training features
    (includes actual points scored which is not to be used as feature)
    """

    # Generates missing features
    temp = generate_auxiliary_features(training_df)

    df = temp.select(["fantasy_points_ppr"] + features())
    return df

def features():
    """
    :return: The columns returned by build_feature_df()
    """
    r = opportunity_capitalization_stats_cols()
    r += team_opportunities_provided_cols()
    r += rb_defense_metrics_cols()
    r += ["weighted_rushing_capitalization_score", "weighted_receiving_capitalization_score",
          "expected_fpoints_scored", "expected_fpoints_allowed", "expected_fpoints_differential",
          "fpoints_per_opportunity"]
    return r