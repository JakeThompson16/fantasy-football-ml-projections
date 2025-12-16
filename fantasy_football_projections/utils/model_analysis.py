
from sklearn.metrics import r2_score, mean_absolute_error
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def training_metrics(y_test, y_pred):
    """

    :param y_test: Test outputs
    :param y_pred: Test outputs
    :return:
    """
    print("R²:", r2_score(y_test, y_pred))
    print("MAE:", mean_absolute_error(y_test, y_pred))

# Shows visuals of training data
def visualize_training(y_test, y_pred, X_train, model):

    # Predictions vs Actual scores
    plt.figure(figsize=(6, 6))
    sns.scatterplot(x=y_test, y=y_pred)
    plt.plot([0, max(y_test)], [0, max(y_test)], 'r--')  # perfect prediction line
    plt.xlabel("Actual Fantasy Points")
    plt.ylabel("Predicted Fantasy Points")
    plt.title("Model Predictions vs Actual")
    plt.show()

    # After training your LightGBM model
    feature_names = X_train.columns
    importances = model.feature_importances_

    # Create a DataFrame for plotting
    feat_imp_df = pd.DataFrame({
        "feature": feature_names,
        "importance": importances
    }).sort_values(by="importance", ascending=False)

    # Option 1: Bar plot (common)
    plt.figure(figsize=(10, 6))
    sns.barplot(x="importance", y="feature", data=feat_imp_df, palette="viridis")
    plt.title("Feature Importance")
    plt.show()
