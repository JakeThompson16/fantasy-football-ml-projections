from functools import lru_cache
import polars as pl
from fantasy_football_projections.data_loading.player_data import load_player_stats
from fantasy_football_projections.wr_modeling.utility import get_stat_cols, get_defense_cols


@lru_cache(maxsize=None)
def league_wr_averages(season):
    """
    Returns select columns of every nfl players averages over season
    :param int season: Season to get averages for
    :return: Polars data-frame with select columns
    """
    # Load player stats and select relevant cols
    ps = load_player_stats(*[season])
    df = ps.select(
        "receptions",
        "receiving_air_yards",
        "targets",
        "player_id",
        "position"
    )

    # Filter df for receivers with at least 1 target
    df = df.filter((pl.col("position") == "WR"))

    # Collapse dataframe so every row is a players averages
    df = df.group_by("player_id").agg(
        (pl.col("receptions").mean()),
        (pl.col("receiving_air_yards").mean()),
        (pl.col("targets").mean())
    )

    df = df.filter(pl.col("targets") > 2)
    # Find avg_depth_of_target for all players
    avgs = df.with_columns(
        (pl.col("receiving_air_yards") / pl.col("targets")).alias("avg_depth_of_target")
    )

    return avgs

@lru_cache(maxsize=None)
def max_reception_per_game(season):
    avgs = league_wr_averages(season)
    m_rpg = avgs["receptions"].max()
    return m_rpg

@lru_cache(maxsize=None)
def max_depth_of_target(season):
    avgs = league_wr_averages(season)
    m_dot = avgs["avg_depth_of_target"].max()
    return m_dot

def select_wanted_cols(df):
    stat_cols = get_stat_cols()
    def_cols = get_defense_cols()
    # Isolates only relevant stat columns
    cols_wanted = ["season", "fantasy_points_ppr"]
    for window in ["3g_avg", "6g_avg", "season_avg"]:
        cols_wanted += [f"{col}_{window}" for col in stat_cols]
    for window in ["6g_avg", "season_avg"]:
        cols_wanted += [f"{col}_{window}" for col in def_cols]
    df = df.select(cols_wanted)

    return df