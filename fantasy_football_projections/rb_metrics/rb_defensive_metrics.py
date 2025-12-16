
from functools import lru_cache
from fantasy_football_projections.data_loading.player_data import load_player_data
from fantasy_football_projections.data_loading.team_data import load_team_def_pbp_data
import polars as pl
from fantasy_football_projections.utils.filtering import select_relevant_plays


@lru_cache(maxsize=None)
def rb_defense_metrics(team, start_week, season, game_amt=16, ppr=1):
    """
    :param team: Team to calculate metrics for
    :param start_week: Week to begin calculating averages for
    :param season: Season to begin calculating averages in
    :param game_amt: Amount of games to base averages on (recommended 16 or every game of current season)
    :param ppr: Amount of points per reception
    :return: Dict of defensive per play metrics conveying the strength of a fantasy defense against rb's
    """
    if game_amt > 17:
        print(f"Game amount out of bounds ({game_amt}), set to 17")
        game_amt = 17
    elif game_amt < 1:
        print(f"Game amount out of bounds ({game_amt}), set to 1")
        game_amt = 1
    seasons = [season]
    if start_week - game_amt < 1:
        seasons.append(season-1)

    plays = load_team_def_pbp_data(*seasons, team=team)

    # Filters player_data by relevant games
    relevant_plays = select_relevant_plays(plays, seasons, game_amt, start_week)

    # Filters relevant data to just plays where opponent rb had carry or target
    # Loads relevant player data
    players = load_player_data(season)

    # Selects relevant columns from players
    players = players.select([
        "gsis_id",
        "position"
    ])

    # Filters relevant_plays to just opposing rb rushing plays
    rb_rush_plays = (relevant_plays.join(
        players,
        left_on="rusher_player_id",
        right_on="gsis_id",
        how="inner"
    ).filter(
        (pl.col("position") == "RB") &
        (pl.col("posteam") != team))
    .with_columns(
        pl.col("rusher_player_id").alias("rb_id")
    ))

    # Filters relevant_plays to just opposing rb receiving plays
    rb_pass_plays = (relevant_plays.join(
        players,
        left_on="receiver_player_id",
        right_on="gsis_id",
        how="inner"
    ).filter((pl.col("position") == "RB") &
             (pl.col("posteam") != team))
    .with_columns(
        pl.col("receiver_player_id").alias("rb_id")
    ))

    # Generates fantasy points gained on every rush attempt
    rb_rush_plays = rb_rush_plays.with_columns(
        (pl.col("rushing_yards") * .1 + pl.col("rush_touchdown") * 6).alias("fpoints")
    )

    # Generates fantasy points gained on every target
    rb_pass_plays = rb_pass_plays.with_columns(
        pl.when(
            (pl.col("complete_pass") == 1))
        .then(pl.col("receiving_yards") * .1 + pl.col("pass_touchdown") * 6 + ppr)
        .otherwise(0)
        .alias("fpoints")
    )

    # Isolates red zone carries and finds carry weights
    rz_rushes = rb_rush_plays.filter(pl.col("yardline_100") <= 20)
    rz_rushes = rz_rushes.with_columns(
        (20 / pl.col("yardline_100")).alias("weight_of_carry")
    )

    # Isolates red zone targets and finds target weights
    rz_targets = rb_pass_plays.filter(pl.col("yardline_100") <= 20)
    rz_targets = rz_targets.with_columns(
        (20 / pl.col("yardline_100")).alias("weight_of_target")
    )

    # Calculates metrics
    epa_per_carry = rb_rush_plays["epa"].mean() or 0
    epa_per_target = rb_pass_plays["epa"].mean() or 0
    fpoints_per_carry = rb_rush_plays["fpoints"].mean() or 0
    fpoints_per_target = rb_pass_plays["fpoints"].mean() or 0

    # Calculates rushing capitalization score against
    rz_td_rushes = rz_rushes["rush_touchdown"].sum()
    rz_carries_amt = len(rz_rushes)
    total_carry_weight = rz_rushes["weight_of_carry"].sum()
    rucs = rz_td_rushes / (total_carry_weight / rz_carries_amt) if rz_td_rushes != 0 else 0

    # Calculates receiving capitalization score against
    rz_td_receptions = rz_targets["pass_touchdown"].sum()
    rz_target_amt = len(rz_targets)
    total_target_weight = rz_targets["weight_of_target"].sum()
    recs = rz_td_receptions / (total_target_weight / rz_target_amt) if rz_target_amt != 0 else 0

    return {
        "epa_per_carry_against": epa_per_carry,
        "epa_per_target_against": epa_per_target,
        "fpoints_per_carry_against": fpoints_per_carry,
        "fpoints_per_target_against": fpoints_per_target,
        "rushing_capitalization_score_against": rucs,
        "receiving_capitalization_score_against": recs
    }

def rb_defense_metrics_cols():
    """
    :return: List of columns returned by dict in rb_defense_metrics
    """
    return ["epa_per_carry_against", "epa_per_target_against", "fpoints_per_carry_against",
            "fpoints_per_target_against","rushing_capitalization_score_against",
            "receiving_capitalization_score_against"]
