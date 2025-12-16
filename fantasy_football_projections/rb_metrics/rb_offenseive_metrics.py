
from functools import lru_cache
import polars as pl
from fantasy_football_projections.data_loading.player_data import load_player_pbp_data, load_player_data, load_player_stats_by_team
from fantasy_football_projections.data_loading.team_data import load_team_pbp_data
from fantasy_football_projections.utils.filtering import select_relevant_plays


# Calculates the rate at which a team provides opportunities to a rb with a depth position
@lru_cache(maxsize=None)
def team_opportunities_provided(team, depth_pos, start_week, season, game_amt):
    """
    :param team: Players Team
    :param depth_pos: Players depth chart position for week being projected
    :param start_week: Week being projected, not included in calculations
    :param season: Season being projected
    :param game_amt: Amount of games used for averages (0 < game_amt <= 17)
    :return: Dict with metrics conveying opportunities team provides to rb at depth_pos
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

    # Creates list with range of relevant weeks of season
    end_week = max(1, start_week - game_amt)
    week_range = list(range(start_week-1, end_week-1, -1))
    pbp_data = load_team_pbp_data(*seasons, team=team)

    # Filters df to all rush/pass plays by team in relevant weeks of season
    relevant_plays = pbp_data.filter(
        ((pl.col("rush_attempt") == 1) |
        (pl.col("pass_attempt") == 1)) &
        (pl.col("season") == season) &
        (pl.col("week").is_in(week_range))
        )

    secondary_range = None
    # Gets data from previous season if necessary, concatenates results to relevant_rushing_plays
    if game_amt >= start_week:
        leftover_games = abs(start_week - game_amt)
        secondary_range = list(range(17, 17-leftover_games, -1))
        t = pbp_data.filter(
            ((pl.col("rush_attempt") == 1) |
            (pl.col("pass_attempt") == 1)) &
            (pl.col("season") == season - 1) &
            (pl.col("week").is_in(secondary_range))
        )
        relevant_plays = pl.concat([relevant_plays, t])

    # Generates tuples of each (week, season) pair to iterate over
    games = []
    for week in week_range:
        games.append((week, season))
    if secondary_range is not None:
        for week in secondary_range:
            games.append((week, season-1))

    # Loads relevant player data
    players = load_player_data(season)

    # Selects relevant columns from players
    players = players.select([
        "gsis_id",
        "position"
    ])

    # Filters relevant_plays to just rb rushing plays
    rb_rush_plays = relevant_plays.join(
        players,
        left_on="rusher_player_id",
        right_on="gsis_id",
        how="inner"
    ).filter(pl.col("position") == "RB").with_columns(
        pl.col("rusher_player_id").alias("rb_id")
    )

    # Filters relevant_plays to just rb receiving plays
    rb_pass_plays = relevant_plays.join(
        players,
        left_on="receiver_player_id",
        right_on="gsis_id",
        how="inner"
    ).filter(pl.col("position") == "RB").with_columns(
        pl.col("receiver_player_id").alias("rb_id")
    )

    # Concatenates all rb rushing and receiving plays
    rb_plays = pl.concat([rb_rush_plays, rb_pass_plays])

    # Loads all relevant weekly data
    weekly_data = load_player_stats_by_team(*seasons, team=team)
    weekly_fpoints = weekly_data.select([
        "player_id",
        "fantasy_points_ppr",
        "week",
        "season"
    ])

    # Adds fantasy_points_ppr to rb_plays
    rb_plays = rb_plays.join(
        weekly_fpoints,
        left_on=["rb_id", "week", "season"],
        right_on=["player_id", "week", "season"],
        how="left"
    )

    selected_players_stats = pl.DataFrame()
    # Iterates over each game, finding the rb at depth_pos and logging their stats
    for game in games:
        plays = rb_plays.filter(
            (pl.col("week") == game[0]) &
            (pl.col("season") == game[1])
        )
        # Skips if bye week
        if len(plays) == 0:
            continue

        # Logs individual counting stats for each relevant play in game
        plays = plays.with_columns([
            pl.col("rush_attempt").alias("rush_touch"),
            pl.col("pass_attempt").alias("pass_touch"),

            # Counts weight of each rush attempt
            pl.when(pl.col("rush_attempt") == 1)
            .then(
                pl.when(pl.col("yardline_100") <= 20)  # red zone
                .then(20 / pl.col("yardline_100"))
                .otherwise(1)  # outside red zone
            )
            .otherwise(0)
            .alias("weighted_rush"),

            # Counts weight of each individual target
            pl.when(pl.col("pass_attempt") == 1)
            .then(
                pl.when(pl.col("yardline_100") <= 20)  # red zone
                .then(20 / pl.col("yardline_100"))
                .otherwise(1)  # outside red zone
            )
            .otherwise(0)
            .alias("weighted_target"),

            # Counts each individual touch
            (pl.col("rush_attempt") + pl.col("pass_attempt")).alias("touches")
        ])

        # Sums totals for each rb that played in game
        rb_game_stats = plays.group_by("rb_id").agg([
            pl.sum("touches").alias("total_opportunities"),
            pl.sum("rush_touch").alias("total_rushes"),
            pl.sum("pass_touch").alias("total_targets"),
            pl.sum("weighted_rush").alias("weight_of_rushes"),
            pl.sum("weighted_target").alias("weight_of_targets"),
            pl.first("fantasy_points_ppr").alias("fantasy_points_ppr")
        ])
        try:
            # Select relevant cols and sort
            top_player_row = rb_game_stats.select([
                "rb_id",
                "total_opportunities",
                "total_rushes",
                "total_targets",
                "weight_of_rushes",
                "weight_of_targets",
                "fantasy_points_ppr"
            ]).sort("total_opportunities", descending=True)

            # Take the player at depth_pos
            top_player_row = top_player_row[depth_pos - 1]

            # Append to selected players
            selected_players_stats = pl.concat([selected_players_stats, top_player_row])
        except IndexError:
            continue

    if selected_players_stats.is_empty():
        return None

    return {
        "opportunities_per_game": selected_players_stats.get_column("total_opportunities").mean(),
        "rushes_per_game": selected_players_stats.get_column("total_rushes").mean(),
        "targets_per_game": selected_players_stats.get_column("total_targets").mean(),
        "weighted_targets_per_game": selected_players_stats.get_column("weight_of_targets").mean(),
        "weighted_rushes_per_game": selected_players_stats.get_column("weight_of_rushes").mean(),
        "fpoints_per_game": selected_players_stats.get_column("fantasy_points_ppr").mean()
    }


def team_opportunities_provided_cols():
    """
    :return: The cols in the polars df returned by team_opportunity_rate
    """
    return ["opportunities_per_game", "rushes_per_game", "targets_per_game",
            "weighted_targets_per_game", "weighted_rushes_per_game",
            "fpoints_per_game"]

# Metrics measuring a players ability to capitalize on their opportunities
@lru_cache(maxsize=None)
def opportunity_capitalization_stats(player_id, start_week, season, game_amt, ppr=1):
    """
    :param player_id: ID of player being projected
    :param start_week: Week being projected (not included in averages)
    :param season: Season being projected
    :param game_amt: Amount of games to construct averages on
    :param ppr: Amount of points per reception to use for calculation
    :return: Dictionary of different metrics pointing to a players ability to capitalize on opportunities
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

    # Loads all player rushing and receiving pbp data for seasons
    player_data = load_player_pbp_data(*seasons, gsis_id=player_id)
    relevant_player_data = select_relevant_plays(player_data, seasons, game_amt, start_week)

    # Isolates players carries
    carries = relevant_player_data.filter(pl.col("rush_attempt") == 1)
    targets = relevant_player_data.filter(pl.col("pass_attempt") == 1)

    # Generates fantasy points gained on every rush attempt
    carries = carries.with_columns(
        (pl.col("rushing_yards") * .1 + pl.col("touchdown") * 6).alias("fpoints")
    )

    # Generates fantasy points gained on every target
    targets = targets.with_columns(
        pl.when(
            (pl.col("complete_pass") == 1))
        .then(pl.col("receiving_yards") * .1 + pl.col("touchdown") * 6 + ppr)
        .otherwise(0)
        .alias("fpoints")
    )

    # Isolates red zone carries and calculates their weights
    rz_carries = carries.filter(pl.col("yardline_100") <= 20)
    rz_carries = rz_carries.with_columns(
        (20/pl.col("yardline_100")).alias("weight_of_carry"),
        pl.when(
            (pl.col("touchdown") == 1) & (pl.col("td_player_id") == player_id))
        .then(1)
        .otherwise(0)
        .alias("td_scored?")
    )

    # Isolates red zone targets and calculates their weights
    rz_targets = targets.filter(pl.col("yardline_100") <= 20)
    rz_targets = rz_targets.with_columns(
        (20 / pl.col("yardline_100")).alias("weight_of_target"),
        pl.when(
            (pl.col("touchdown") == 1) & (pl.col("td_player_id") == player_id))
            .then(1).
            otherwise(0).
            alias("td_scored?")
        )

    # Calculate yards per carry
    attempts = carries["rush_attempt"].sum()
    rush_yards = carries["rushing_yards"].sum()
    yards_per_carry = rush_yards / attempts if attempts != 0 else 0

    # Calculates yards per target
    target_amt = targets["pass_attempt"].sum()
    receiving_yards = targets["receiving_yards"].sum()
    yards_per_target = receiving_yards / target_amt if target_amt != 0 else 0

    # Calculates fantasy points per target and carry
    rushing_fpoints = carries["fpoints"].sum()
    fpoints_per_carry = rushing_fpoints / attempts if attempts != 0 else 0
    receiving_fpoints = targets["fpoints"].sum()
    fpoints_per_target = receiving_fpoints / target_amt if target_amt != 0 else 0

    # Calculates red zone capitalization score for carries
    rz_td_rushes = rz_carries["td_scored?"].sum()
    rz_rush_attempts = len(rz_carries)
    total_carry_weight = rz_carries["weight_of_carry"].sum()
    rz_carry_capitalization_score = rz_td_rushes / (total_carry_weight / rz_rush_attempts) if total_carry_weight != 0 else 0

    # Calculates red zone capitalization score for targets
    rz_td_rec = rz_targets["td_scored?"].sum()
    rz_targets_amt = len(rz_targets)
    total_target_weight = rz_targets["weight_of_target"].sum()
    rz_target_capitalization_score = rz_td_rec / (total_target_weight / rz_targets_amt) if rz_targets_amt != 0 else 0

    try:
        return {
            "yards_per_carry": yards_per_carry,
            "yards_per_target": yards_per_target,
            "fpoints_per_carry": fpoints_per_carry,
            "fpoints_per_target": fpoints_per_target,
            "rushing_capitalization_score": rz_carry_capitalization_score,
            "receiving_capitalization_score": rz_target_capitalization_score
        }
    except pl.exceptions.ColumnNotFoundError:
        return None

def opportunity_capitalization_stats_cols():
    """
    :return: The cols in the dict returned by opportunity_capitalization_stats()
    """
    return ["yards_per_carry", "yards_per_target", "fpoints_per_carry",
            "fpoints_per_target", "rushing_capitalization_score",
            "receiving_capitalization_score"]
