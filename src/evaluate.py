import pandas as pd
import numpy as np


def regression_metrics(y_true, y_pred):
    """Calculate MAE, RMSE, MAPE, and R² for regression predictions."""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    r2 = 1 - np.sum((y_true - y_pred) ** 2) / np.sum((y_true - np.mean(y_true)) ** 2)

    return {'MAE': round(mae, 2), 'RMSE': round(rmse, 2), 'MAPE': round(mape, 2), 'R2': round(r2, 4)}


def comparison_table(results_dict):
    """Print a formatted comparison table from a dict of {model_name: metrics_dict}."""
    df = pd.DataFrame(results_dict).T
    df.index.name = 'Model'
    print(df.to_string())
    return df

