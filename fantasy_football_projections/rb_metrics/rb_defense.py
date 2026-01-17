
import polars as pl

from fantasy_football_projections.config import POINTS_PER_RUSH_YARD, POINTS_PER_RUSH_TD, EXPLOSIVE_RUN, PPR, POINTS_PER_REC_YARD, POINTS_PER_REC_TD, \
    EXPLOSIVE_RECEPTION
from fantasy_football_projections.data_loading.player_data import load_pbp_data, get_rb_ids
from fantasy_football_projections.rb_metrics.utility import get_rb_defensive_cols


def rb_defensive_metrics(seasons: list[int] | int) -> pl.DataFrame:
    """

    :param seasons: Seasons to aggregate data from
    :return: Data-frame with each teams rb defensive metrics per week, max rows: 17 * 32 per season
    """

    pbp_stats = load_pbp_data(*seasons)
    rb_ids = get_rb_ids()

    # Separates to carries and rushing players
    rushes = pbp_stats.filter(pl.col("rusher_player_id").is_in(rb_ids))
    targets = pbp_stats.filter(pl.col("receiver_player_id").is_in(rb_ids))

    # Generates relevant rushing statistics
    rushes = rushes.with_columns(

        # Successful plays
        pl.when(pl.col("epa") > 0)
        .then(1)
        .otherwise(0)
        .alias("successful_rush"),

        # Redzone carries
        pl.when(pl.col("yardline_100") <= 20)
        .then(1)
        .otherwise(0)
        .alias("redzone_carry"),

        # Redzone rush tds
        pl.when((pl.col("yardline_100") <= 20) & (pl.col("rush_touchdown") == 1))
        .then(1)
        .otherwise(0)
        .alias("redzone_rush_td"),

        # Fantasy points gained
        (pl.col("yards_gained") * POINTS_PER_RUSH_YARD +
        pl.col("rush_touchdown") * POINTS_PER_RUSH_TD)
        .alias("rush_fpoints_gained"),

        # Rushing yards
        pl.col("yards_gained")
        .alias("rush_yards"),

        # Carries
        pl.when(pl.col("play_type") == "run")
        .then(1)
        .otherwise(0)
        .alias("carry"),

        # Explosive rush
        pl.when(pl.col("yards_gained") >= EXPLOSIVE_RUN)
        .then(1)
        .otherwise(0)
        .alias("explosive_rush"),

        # Rushing epa
        pl.col("epa")
        .alias("rush_epa")
    )

    # Generates relevant receiving statistics
    targets = targets.with_columns(

        # Successful passing play
        pl.when(pl.col("epa") > 0)
        .then(1)
        .otherwise(0)
        .alias("successful_target"),

        # Fantasy points gained
        pl.when(pl.col("complete_pass") == 1)
        .then(
            1 * PPR +
            pl.col("yards_gained") * POINTS_PER_REC_YARD +
            pl.col("pass_touchdown") * POINTS_PER_REC_TD
        )
        .otherwise(0)
        .alias("rec_fpoints_gained"),

        # Explosive reception
        pl.when(pl.col("yards_gained") >= EXPLOSIVE_RECEPTION)
        .then(1)
        .otherwise(0)
        .alias("explosive_reception"),

        # Targets
        pl.when(pl.col("play_type") == "pass")
        .then(1)
        .otherwise(0)
        .alias("target"),

        # Receiving yards
        pl.col("yards_gained")
        .alias("receiving_yards"),

        # Receiving epa
        pl.col("epa")
        .alias("receiving_epa")
    )

    # Groups rushes to all rb's against a defense during week of season
    rushes = rushes.group_by(["week", "season", "defteam"]).agg(

        # Successful rushes
        pl.col("successful_rush").sum().alias("successful_rushes"),

        # Redzone metrics
        pl.col("redzone_carry").sum().alias("redzone_carries"),
        pl.col("redzone_rush_td").sum().alias("redzone_rush_tds"),

        # Rushing fantasy points
        pl.col("rush_fpoints_gained").sum().alias("rush_fpoints_total"),

        # Rush yards
        pl.col("rush_yards").sum().alias("rush_yards_total"),

        # Carries
        pl.col("carry").sum().alias("carries"),

        # Explosive rushes
        pl.col("explosive_rush").sum().alias("explosive_rushes"),

        # Rushing epa
        pl.col("rush_epa").sum().alias("rush_epa_total")
    )

    # Groups targets to all rb's against a defense during week of season
    targets = targets.group_by(["week", "season", "defteam"]).agg(

        # Successful targets
        pl.col("successful_target").sum().alias("successful_targets"),

        # Fantasy points gained
        pl.col("rec_fpoints_gained").sum().alias("rec_fpoints_total"),

        # Explosive receptions
        pl.col("explosive_reception").sum().alias("explosive_receptions"),

        # Targets
        pl.col("target").sum().alias("targets"),

        # Receiving yards
        pl.col("receiving_yards").sum().alias("receiving_yards_total"),

        # Receiving epa
        pl.col("receiving_epa").sum().alias("receiving_epa_total")
    )

    # Joins rushing and passing plays
    df = rushes.join(
        targets,
        left_on=["week", "season", "defteam"],
        right_on=["week", "season", "defteam"],
        how="outer"
    ).fill_null(0)

    cols = get_rb_defensive_cols() + ["week", "season", "defteam"]
    return df.select(cols)