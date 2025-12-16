
import polars as pl

def select_relevant_plays(plays, seasons, game_amt, start_week):
    """
    Filters player_data from [start_week-game_amt, start_week-1]
    :param pl.DataFrame plays: Player data to filter
    :param int[] seasons: That fall into range of games
    :param int game_amt: Amount of games
    :param start_week: Week that is being projected (not included in returned df)
    :return: Data frame with all play-by-play data in specified range
    """
    season = seasons[0]

    # Creates list with range of relevant weeks of season
    end_week = max(1, start_week - game_amt)
    week_range = list(range(start_week - 1, end_week - 1, -1))

    # Filters player_data by relevant games
    relevant_player_data = plays.filter(
        (pl.col("season") == season) &
        (pl.col("week").is_in(week_range))
    )

    # Gets data from previous season if necessary, concatenates results to relevant_player_data
    if game_amt >= start_week:
        leftover_games = abs(start_week - game_amt)
        secondary_range = list(range(17, 17 - leftover_games, -1))
        secondary_player_data = plays.filter(
            (pl.col("season") == season - 1) &
            (pl.col("week").is_in(secondary_range))
        )
        relevant_player_data = pl.concat([secondary_player_data, relevant_player_data])
    return relevant_player_data
