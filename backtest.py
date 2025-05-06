# strategic_backtest_ensemble.py

import argparse
import os
import json
import numpy as np
import pandas as pd
import torch
import joblib
import matplotlib.pyplot as plt
from functools import reduce
from darts import TimeSeries
from darts.models import NBEATSModel, TFTModel
from darts.dataprocessing.transformers import Scaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# Metrics

def sharpe_ratio(returns):
    excess = np.array(returns) - np.mean(returns)
    return np.mean(excess) / (np.std(excess) + 1e-9)

def max_drawdown(returns):
    cumulative = np.cumsum(returns)
    peak = np.maximum.accumulate(cumulative)
    drawdown = (peak - cumulative)
    return np.max(drawdown)

def simulate_trading(actual_df, predicted_df, threshold=0.001):
    trades = []
    returns = []
    actual_vals = actual_df['close'].values
    predicted_vals = predicted_df['close'].values
    timestamps = actual_df.index
    
    n = min(len(actual_vals), len(predicted_vals)) - 1

    for i in range(n):
        pred_diff = predicted_vals[i + 1] - actual_vals[i]
        act_diff = actual_vals[i + 1] - actual_vals[i]

        position = 1 if pred_diff > threshold else -1 if pred_diff < -threshold else 0
        ret = position * (act_diff / actual_vals[i])
        returns.append(ret)
        trades.append({
            "timestamp": timestamps[i],
            "position": position,
            "predicted": predicted_vals[i + 1],
            "actual": actual_vals[i + 1],
            "return": ret
        })
    return trades, returns

def evaluate_predictions(actual, predicted):
    return {
        "MAE": float(mean_absolute_error(actual, predicted)),
        "RMSE": float(np.sqrt(mean_squared_error(actual, predicted))),
        "R2": float(r2_score(actual, predicted))
    }

def plot_cumulative_returns(returns, output_dir):
    plt.figure()
    plt.plot(np.cumsum(returns), label='Cumulative Return')
    plt.title("Cumulative Strategy Return")
    plt.xlabel("Step")
    plt.ylabel("Cumulative Return")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "cumulative_return.png"))
    plt.close()

def main(args):
    os.makedirs(args.output_dir, exist_ok=True)

    df_train = pd.read_csv(args.train_csv)
    df_val = pd.read_csv(args.val_csv)
    df_test = pd.read_csv(args.test_csv)

    df = pd.concat([df_train, df_val, df_test], ignore_index=True)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.set_index('timestamp').sort_index()
    df = df[~df.index.duplicated(keep='first')]
    df = df.asfreq('h')
    df = df.replace([np.inf, -np.inf], np.nan).fillna(0)

    with open(os.path.join(args.model_dir, "training_summary.json")) as f:
        summary = json.load(f)
    past_features = summary["past_features"]
    future_features = summary["future_features"]

    series = TimeSeries.from_dataframe(df, value_cols=["close_diff"], freq='h').astype(np.float64)
    close_series = TimeSeries.from_dataframe(df, value_cols=["close"], freq='h').astype(np.float64)
    past_cov = TimeSeries.from_dataframe(df, value_cols=past_features, freq='h').astype(np.float64)
    future_cov = TimeSeries.from_dataframe(df, value_cols=future_features, freq='h').astype(np.float64)

    series_scaler = joblib.load(os.path.join(args.model_dir, 'series_scaler.pkl'))
    series_scaled = series_scaler.transform(series)
    past_scaler = joblib.load(os.path.join(args.model_dir, 'past_cov_scaler.pkl'))
    future_scaler = joblib.load(os.path.join(args.model_dir, 'future_cov_scaler.pkl'))
    past_scaled = past_scaler.transform(past_cov)
    future_scaled = future_scaler.transform(future_cov)

    model_names = args.models.split(',')
    model_classes = {"NBEATSModel": NBEATSModel, "TFTModel": TFTModel}
    models = [model_classes[name].load(os.path.join(args.model_dir, f"{name}_model.pkl")) for name in model_names]

    print("Generating ensemble forecasts...")
    all_forecasts = []
    for model in models:
        forecasts_scaled = model.historical_forecasts(
            series=series_scaled,
            past_covariates=past_scaled,
            future_covariates=future_scaled if isinstance(model, TFTModel) else None,
            forecast_horizon=1,
            stride=1,
            last_points_only=True,
            retrain=False,
            start=0.9,
            verbose=True
        )
        forecast = forecasts_scaled[0]
        for f in forecasts_scaled[1:]:
            forecast = forecast.append(f)
        all_forecasts.append(forecast)

    min_len = min(len(f) for f in all_forecasts)
    aligned_forecasts = [f[-min_len:] for f in all_forecasts]
    avg_forecast_scaled = reduce(lambda a, b: a + b, aligned_forecasts) * (1.0 / len(aligned_forecasts))

    forecast_diff = series_scaler.inverse_transform(avg_forecast_scaled)
    forecast_times = forecast_diff.time_index

    actual_close = close_series.slice_intersect(forecast_diff)
    predicted_close = actual_close[:-1] + forecast_diff[1:]

    predicted_vals = predicted_close.univariate_values()
    actual_vals = actual_close.univariate_values()[:len(predicted_vals)]

    eval_metrics = evaluate_predictions(actual_vals, predicted_vals)
    actual_df = actual_close.to_dataframe()
    predicted_df = predicted_close.to_dataframe()
    trades, strategy_returns = simulate_trading(actual_df, predicted_df)

    eval_metrics.update({
        "SharpeRatio": float(sharpe_ratio(strategy_returns)),
        "MaxDrawdown": float(max_drawdown(strategy_returns)),
        "CumulativeReturn": float(np.sum(strategy_returns))
    })

    with open(os.path.join(args.output_dir, "strategy_metrics.json"), "w") as f:
        json.dump(eval_metrics, f, indent=4)

    pd.DataFrame(trades).to_csv(os.path.join(args.output_dir, "trade_log.csv"), index=False)
    plot_cumulative_returns(strategy_returns, args.output_dir)

    print("\nBacktest complete. Metrics saved. Cumulative return plotted.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ensemble strategic backtest")
    parser.add_argument("--train_csv", type=str, required=True)
    parser.add_argument("--val_csv", type=str, required=True)
    parser.add_argument("--test_csv", type=str, required=True)
    parser.add_argument("--model_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--models", type=str, required=True, help="Comma-separated model names (e.g. NBEATSModel,TFTModel)")
    args = parser.parse_args()
    main(args)
