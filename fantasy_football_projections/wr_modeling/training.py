
import polars as pl
from lightgbm import LGBMRegressor
from sklearn.model_selection import train_test_split

from fantasy_football_projections.utils.model_analysis import training_metrics, visualize_training
from fantasy_football_projections.wr_modeling.feature_engineering import build_feature_df, features


def train(location, show_metrics=False, show_visuals=False):
    """
    :param str location: The file path to the parquet file with the training data-frame
    :param bool show_metrics: Whether to show training metrics or not
    :param bool show_visuals: Whether to show training visuals or not
    :return: A gbdt model for predicting fantasy rb output
    """
    # Uses all rb1, rb2, rb3 games over 2021 - 2024 seasons
    df = pl.read_parquet(location)

    # Gets features df
    features_df = build_feature_df(df)

    # Isolate features and target
    X = features_df.select(features()).to_pandas()
    y = features_df["fantasy_points_ppr"].to_numpy().ravel()

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Create gradiant boosting regression tree model
    model = LGBMRegressor(
        objective='regression',
        boosting_type='gbdt',
        learning_rate=0.02,
        num_leaves=80,  # increased complexity
        max_depth=20,  # deeper trees
        n_estimators=1800,  # more trees for convergence
        min_child_samples=12,
        feature_fraction=0.9,
        lambda_l1=0.1,
        lambda_l2=0.1,
        random_state=42,
        verbosity=-1,
    )

    """
    Current Model:
    
    """

    # Fit model
    model.fit(X_train, y_train)

    if show_metrics:
        training_metrics(y_test, y_pred=model.predict(X_test))
    if show_visuals:
        visualize_training(y_test, y_pred=model.predict(X_test), X_train=X_train, model=model)

    return model

def train_and_save(location, show_metrics=False, show_visuals=False):
    """
    Trains and saves model to file: "rb_model.txt"
    :param location: The file path to the parquet file with the training data-frame
    :param show_metrics: Whether to show training metrics or not
    :param show_visuals: Whether to show training visuals or not
    :return: The trained model
    """
    model = train(location, show_metrics, show_visuals)
    model.booster_.save_model("fantasy_football_projections/data_loading/models/wr_model.txt")
    return model