
import os

def make_file_path(position, seasons, off_game_amt, def_game_amt, ppr=1):
    """
    ex. position="RB",seasons=[2021,2022], off_game_amt=6, def_game_amt=8, ppr=1:
    "fantasy_football_projections/data_loading/datasets/rb_ds_2122_o6d8p1.parquet"
    :param str position: Players position
    :param int[] seasons: Seasons dataset ranges from
    :param int off_game_amt: Amount of games to construct offensive averages from
    :param int def_game_amt: Amount of games to construct defensive averages from
    :param float ppr: Points per game
    :return: Parquet file path
    """
    # Find directory
    folder = os.path.join("fantasy_football_projections", "data_loading", "datasets")
    os.makedirs(folder, exist_ok=True)  # create directories if they don't exist

    # Find file name
    season_str = f"{str(seasons[0])[-2:]}{str(seasons[-1])[-2:]}"
    filename = f"{position.lower()}_ds_{season_str}_o{off_game_amt}d{def_game_amt}p{ppr}.parquet"

    # Find full path
    full_path = os.path.join(folder, filename)
    return full_path