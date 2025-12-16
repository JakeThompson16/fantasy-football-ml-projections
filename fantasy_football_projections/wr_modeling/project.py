
import polars as pl
from fantasy_football_projections.data_loading.load_models import load_recent_wr_model
from fantasy_football_projections.data_loading.schedule_data import load_schedule_data
from fantasy_football_projections.wr_metrics.universal_averages import select_wanted_cols
from fantasy_football_projections.wr_metrics.wr_defense_stat_aggregation import get_wr_defense_weekly_stats, \
    get_wr_defense_pbp_stats
from fantasy_football_projections.wr_metrics.wr_defensive_metrics import generate_defensive_averages
from fantasy_football_projections.wr_metrics.wr_offensive_metrics import generate_offensive_averages
from fantasy_football_projections.wr_metrics.wr_stat_aggregation import get_wr_snap_counts, get_wr_weekly_stats, \
    get_wr_pbp_stats_weekly, get_wr_nextgen_stats
from fantasy_football_projections.wr_modeling.feature_engineering import features, generate_auxiliary_features


def prepare_wr_metrics(player_id, season, week) -> pl.DataFrame:
    """
    Returns a data-frame ready to project a fantasy wr's output in next game
    :param player_id:
    :param season:
    :param week:
    :return: polars DataFrame
    """
    # Loads snap counts, filters by player id
    snap_counts = get_wr_snap_counts([season]).filter(
        pl.col("gsis_id") == player_id
    )

    # Loads weekly stats, filters by player
    player_stats = get_wr_weekly_stats([season]).filter(
        pl.col("player_id") == player_id
    )

    # Joins plays stats with snap counts
    df = player_stats.join(
        snap_counts,
        left_on=["player_id", "week", "season"],
        right_on=["gsis_id", "week", "season"],
        how="inner"
    )

    # Loads pbp stats aggregated by week, filters by player
    pbp_stats = get_wr_pbp_stats_weekly([season]).filter(
        pl.col("player_id") == player_id
    )

    # Joins df with pbp_stats
    df = df.join(
        pbp_stats,
        left_on=["player_id", "week", "season"],
        right_on=["player_id", "week", "season"],
        how="inner"
    )

    # Loads nextgen stats
    nextgen = get_wr_nextgen_stats([season]).filter(
        pl.col("player_gsis_id") == player_id
    )

    # Joins nextgen to df
    df = df.join(
        nextgen,
        left_on=["player_id", "week", "season"],
        right_on=["player_gsis_id", "week", "season"],
        how="inner"
    )

    df = generate_offensive_averages(df, training=False)

    team = df.tail(1)["team"].item()

    schedule = load_schedule_data(*[season]).filter(
        ((pl.col("away_team") == team) |
        (pl.col("home_team") == team)) &
        (pl.col("week") == week)
    )

    if schedule.select(["away_team"]).item() == team:
        opponent_team = schedule.select(["home_team"]).item()
    else:
        opponent_team = schedule.select(["away_team"]).item()

    print(opponent_team)

    defense_stats = get_wr_defense_weekly_stats([season]).filter(
        pl.col("opponent_team") == opponent_team
    )

    defense_pbp = get_wr_defense_pbp_stats([season]).filter(
        pl.col("defteam") == opponent_team
    )

    defense_df = defense_stats.join(
        defense_pbp,
        left_on=["opponent_team", "week", "season"],
        right_on=["defteam", "week", "season"],
        how="left"
    )

    print(defense_df.describe())

    defense_df = generate_defensive_averages(defense_df, training=False)

    # Joins df and defense_df
    df = df.join(
        defense_df,
        left_on=["opponent_team", "week", "season"],
        right_on=["opponent_team", "week", "season"],
        how="left"
    )

    df = select_wanted_cols(df)

    return generate_auxiliary_features(df)

def project_player_points(player_id, season, week):
    model = load_recent_wr_model()
    X_pred = prepare_wr_metrics(player_id, season, week).select(features()).to_pandas()
    y_pred = model.predict(X_pred)
    return float(y_pred[0])
