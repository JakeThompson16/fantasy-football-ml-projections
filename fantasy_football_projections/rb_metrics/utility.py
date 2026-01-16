from functools import lru_cache
from typing import List
import polars as pl
from fantasy_football_projections.data_loading.player_data import load_ff_opportunity_data, load_player_stats

@lru_cache(maxsize=None)
def rusher_opportunity_bounds(season)->dict[str, float]:
    """
    Stats used: carries, rush_yards_exp, rush_td_exp, rush_first_down_exp
    :param int season: Seasons to get rush yards expected bounds for
    :return: Dict containing the 1st percentile and 99th percentile for expected stats
    per game in a given season
    """
    player_stats = load_player_stats(season).filter(
        pl.col("position") == "RB"
    )
    ff_opp = load_ff_opportunity_data(season).filter(
        pl.col("position") == "RB"
    )

    # Carries
    car_lo, car_hi = player_stats.select(
        pl.col("carries").quantile(0.01).alias("car_p01"),
        pl.col("carries").quantile(0.99).alias("car_p99")
    ).row(0)

    # Targets
    tar_lo, tar_hi = player_stats.select(
        pl.col("targets").quantile(0.01).alias("tar_p01"),
        pl.col("targets").quantile(0.99).alias("tar_p99")
    )

    # Rush yards expected
    rye_lo, rye_hi = ff_opp.select(
        pl.col("rush_yards_gained_exp").quantile(0.01).alias("rye_p01"),
        pl.col("rush_yards_gained_exp").quantile(0.99).alias("rye_p99")
    ).row(0)

    # Rush TD expected
    rtd_lo, rtd_hi = ff_opp.select(
        pl.col("rush_touchdown_exp").quantile(0.01).alias("rtd_p01"),
        pl.col("rush_touchdown_exp").quantile(0.99).alias("rtd_p99")
    ).row(0)

    # Rush first downs expected
    rfd_lo, rfd_hi = ff_opp.select(
        pl.col("rush_first_down_exp").quantile(0.01).alias("rfd_p01"),
        pl.col("rush_first_down_exp").quantile(0.99).alias("rfd_p99")
    ).row(0)

    # Receiving yards gained expected
    reye_lo, reye_hi = ff_opp.select(
        pl.col("rec_yards_gained_exp").quantile(0.01).alias("reye_p01"),
        pl.col("rec_yards_gained_exp").quantile(0.99).alias("reye_p99")
    )

    # Receiving touchdown expected
    retd_lo, retd_hi = ff_opp.select(
        pl.col("rec_touchdown_exp").quantile(0.01).alias("retd_p01"),
        pl.col("rec_touchdown_exp").quantile(0.99).alias("retd_p99")
    )

    # Receiving first downs expected
    refd_lo, refd_hi = ff_opp.select(
        pl.col("rec_first_down_exp").quantile(0.01).alias("refd_p01"),
        pl.col("rec_first_down_exp").quantile(0.99).alias("refd_p99")
    )

    # Yards after catch expected
    rec_lo, rec_hi = ff_opp.select(
        pl.col("receptions_exp").quantile(0.01).alias("rec_p01"),
        pl.col("receptions_exp").quantile(0.99).alias("rec_p99")
    )

    return {
        "carries_upper" : car_hi,
        "carries_lower" : car_lo,
        "rush_yards_exp_upper" : rye_hi,
        "rush_yards_exp_lower" : rye_lo,
        "rush_td_exp_upper" : rtd_hi,
        "rush_td_exp_lower" : rtd_lo,
        "rush_first_down_exp_upper" : rfd_hi,
        "rush_first_down_exp_lower" : rfd_lo,
        "targets_upper" : tar_hi,
        "targets_lower" : tar_lo,
        "rec_yards_exp_upper" : reye_hi,
        "rec_yards_exp_lower" : reye_lo,
        "rec_td_exp_upper" : retd_hi,
        "rec_td_exp_lower" : retd_lo,
        "rec_first_down_exp_upper" : refd_hi,
        "rec_first_down_exp_lower" : refd_lo,
        "receptions_exp_upper" : rec_hi,
        "receptions_exp_lower" : rec_lo
    }