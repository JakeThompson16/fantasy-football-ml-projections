
import polars as pl

from fantasy_football_projections.config import (
    PPR,
    POINTS_PER_RUSH_YARD,
    POINTS_PER_RUSH_TD,
    POINTS_PER_REC_YARD,
    POINTS_PER_REC_TD,
    EXPLOSIVE_RUN,
    EXPLOSIVE_RECEPTION,
)

from fantasy_football_projections.data_loading.player_data import (
    load_player_stats,
    load_pbp_data,
    get_rb_ids,
)

from fantasy_football_projections.rb_metrics.utility import get_rb_efficiency_cols


# Metrics measuring a players ability to capitalize on their opportunities
def rb_efficiency_metrics(seasons: list[int] | int)->pl.DataFrame:
    """
    Returns a data frame of feature cols for rb efficiency metrics, cols found
    in get_rb_efficiency_cols()
    :param seasons: List of seasons to get data for
    :return: Data frame of weekly efficiency metrics for every rb in seasons
    """

    df = load_player_stats(seasons).filter(pl.col("position") == "RB")
    pbp_stats = load_pbp_data(seasons)

    rb_ids = get_rb_ids()

    # Filters pbp carries
    pbp_carries = pbp_stats.filter(pl.col("rusher_player_id").is_in(rb_ids))

    # Filters pbp targets
    pbp_targets = pbp_stats.filter(pl.col("receiver_player_id").is_in(rb_ids))

    # Generates needed stat cols in pbp_carries
    pbp_carries = pbp_carries.with_columns(

        # Redzone Carries
        pl.when(pl.col("yardline_100") <= 20)
        .then(1)
        .otherwise(0)
        .alias("redzone_carry"),

        # Redzone touchdowns
        pl.when((pl.col("yardline_100") <= 20) & (pl.col("rush_touchdown") == 1))
        .then(1)
        .otherwise(0)
        .alias("redzone_td_rush"),

        # Positive epa play
        pl.col("success")
        .alias("successful_rush"),

        # Rush of 10+ yards
        pl.when(pl.col("yards_gained") >= EXPLOSIVE_RUN)
        .then(1)
        .otherwise(0)
        .alias("explosive_rush"),

        # Fantasy points gained
        (pl.col("yards_gained") * POINTS_PER_RUSH_YARD +
        pl.col("rush_touchdown") * POINTS_PER_RUSH_TD)
        .alias("rush_fpoints_gained")
    )

    # Generates needed stat cols in pbp_targets
    pbp_targets = pbp_targets.with_columns(

        # Redzone targets
        pl.when(pl.col("yardline_100") <= 20)
        .then(1)
        .otherwise(0)
        .alias("redzone_target"),

        # Redzone touchdowns
        pl.when((pl.col("yardline_100") <= 20) & (pl.col("pass_touchdown") == 1))
        .then(1)
        .otherwise(0)
        .alias("redzone_td_reception"),

        # Positive epa play
        pl.col("success")
        .alias("successful_target"),

        # Reception of 12+ yards
        pl.when(pl.col("yards_gained") >= EXPLOSIVE_RECEPTION)
        .then(1)
        .otherwise(0)
        .alias("explosive_target"),

        # Fantasy points gained
        pl.when(pl.col("complete_pass") == 0)
        .then(0)
        .otherwise(
            1 * PPR +
            pl.col("yards_gained") * POINTS_PER_REC_YARD +
            pl.col("pass_touchdown") * POINTS_PER_REC_TD
        )
        .alias("rec_fpoints_gained")
    )

    # Aggregates pbp carries to weekly totals
    pbp_carries = pbp_carries.group_by(["week", "season", "rusher_player_id"]).agg(

        # Redzone opportunities
        pl.col("redzone_carry").sum().alias("redzone_carries"),

        # Redzone touchdowns
        pl.col("redzone_td_rush").sum().alias("redzone_td_rushes"),

        # Successful plays
        pl.col("successful_rush").sum().alias("successful_rushes"),

        # Fantasy points gained
        pl.col("rush_fpoints_gained").sum().alias("rush_fpoints_gained"),

        # Explosive plays
        pl.col("explosive_rush").sum().alias("explosive_rushes")
    )

    # Aggregates pbp receptions to weekly totals
    pbp_targets = pbp_targets.group_by(["week", "season", "receiver_player_id"]).agg(

        # Redzone opportunities
        pl.col("redzone_target").sum().alias("redzone_targets"),

        # Redzone touchdowns
        pl.col("redzone_td_reception").sum().alias("redzone_td_receptions"),

        # Successful plays
        pl.col("successful_target").sum().alias("successful_targets"),

        # Fantasy points gained
        pl.col("rec_fpoints_gained").sum().alias("rec_fpoints_gained"),

        # Explosive plays
        pl.col("explosive_target").sum().alias("explosive_receptions")
    )

    # Renames week and season in pbp_carries and pbp_targets to avoid duplicate error on the second join
    pbp_carries = pbp_carries.rename({
        "week": "c_week",      # polars renames these cols to week/season_right and tries to do
        "season": "c_season"   # so with df.join 16 lines down but throws duplicate col error
    })
    pbp_targets = pbp_targets.rename({
        "week": "t_week",
        "season": "t_season"
    })

    # Joins pbp carries and targets
    pbp_stats = pbp_carries.join(
        pbp_targets,
        left_on=["c_week", "c_season", "rusher_player_id"],
        right_on=["t_week", "t_season", "receiver_player_id"],
        how="outer"
    )

    # Joins pbp stats and player stats
    df = df.join(
        pbp_stats,
        left_on=["week", "season", "player_id"],
        right_on=["c_week", "c_season", "rusher_player_id"],
        how="outer"
    ).fill_null(0)

    # Generates efficiency features
    df = df.with_columns(

        # Redzone rushing efficiency
        pl.when(pl.col("redzone_carries") > 0)
        .then(pl.col("redzone_td_rushes") / pl.col("redzone_carries"))
        .otherwise(0)
        .alias("redzone_carry_efficiency"),

        # Redzone target efficiency
        pl.when(pl.col("redzone_targets") > 0)
        .then(pl.col("redzone_td_receptions") / pl.col("redzone_targets"))
        .otherwise(0)
        .alias("redzone_target_efficiency"),

        # Yards per carry
        pl.when(pl.col("carries") > 0)
        .then(pl.col("rushing_yards") / pl.col("carries"))
        .otherwise(0)
        .alias("yards_per_carry"),

        # Yards per target
        pl.when(pl.col("targets") > 0)
        .then(pl.col("receiving_yards") / pl.col("targets"))
        .otherwise(0)
        .alias("yards_per_target"),

        # Successful carry rate
        pl.when(pl.col("carries") > 0)
        .then(pl.col("successful_rushes") / pl.col("carries"))
        .otherwise(0)
        .alias("successful_carry_rate"),

        # Successful target rate
        pl.when(pl.col("targets") > 0)
        .then(pl.col("successful_targets") / pl.col("targets"))
        .otherwise(0)
        .alias("successful_target_rate"),

        # Explosive run rate
        pl.when(pl.col("carries") > 0)
        .then(pl.col("explosive_rushes") / pl.col("carries"))
        .otherwise(0)
        .alias("explosive_carry_rate"),

        # Explosive reception rate
        pl.when(pl.col("targets") > 0)
        .then(pl.col("explosive_receptions") / pl.col("targets"))
        .otherwise(0)
        .alias("explosive_reception_rate"),

        # Fantasy points per carry
        pl.when(pl.col("carries") > 0)
        .then(pl.col("rush_fpoints_gained") / pl.col("carries"))
        .otherwise(0)
        .alias("fpoints_per_carry"),

        # Fantasy points per target
        pl.when(pl.col("targets") > 0)
        .then(pl.col("rec_fpoints_gained") / pl.col("targets"))
        .otherwise(0)
        .alias("fpoints_per_target"),
    )

    cols = get_rb_efficiency_cols() + ["player_id", "week", "season", "opponent_team"]
    return df.select(cols)

