

def get_defense_cols():
    """
    :return: Returns list of cols used in defensive averages
    """
    return ["targets_against", "receptions_against", "receiving_yards_against", "receiving_tds_against",
            "receiving_epa_against", "racr_against", "yards_after_catch_against", "air_yards_against",
            "yac_epa_against", "redzone_targets_against", "big_play_attempts_against",
            "redzone_touchdowns_against", "big_play_conversions_against"]

def get_stat_cols():
    """
    :return: Returns list of cols used in offensive averages
    """
    return ["receptions", "targets", "receiving_yards", "receiving_tds", "receiving_air_yards",
            "receiving_yards_after_catch", "receiving_first_downs", "receiving_epa", "racr",
             "target_share", "air_yards_share", "wopr", "avg_cushion", "avg_separation",
             "catch_percentage", "avg_yac_above_expectation", "redzone_targets", "big_play_attempts",
             "comp_yac_epa", "redzone_touchdowns", "big_play_conversions", "air_yards_targeted",
             "offense_pct", "fantasy_points_ppr"]