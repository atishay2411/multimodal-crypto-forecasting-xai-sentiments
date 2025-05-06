# --- Imports ---
import argparse
import os
import logging
import pandas as pd
import numpy as np
import torch
import joblib
from tqdm import tqdm
import matplotlib.pyplot as plt
from darts import TimeSeries
from darts.dataprocessing.transformers import Scaler
from darts.models import NBEATSModel, TFTModel
from darts.utils.likelihood_models import QuantileRegression
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger
import json

# --- Set Global Default Dtype to float64 ---
torch.set_default_dtype(torch.float64)

# --- Set Global Plot Style ---
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    "figure.figsize": (12, 6),
    "axes.titlesize": 16,
    "axes.labelsize": 14,
    "legend.fontsize": 12,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "lines.linewidth": 2,
    "savefig.dpi": 300,
    "savefig.bbox": 'tight'
})

# --- Logging ---
logger = logging.getLogger("TrainingLogger")
logger.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

# --- Utility Functions ---
def save_training_summary(output_dir, past_features, future_features):
    summary = {
        "past_features": past_features,
        "future_features": future_features,
        "notes": "Target variable is close_diff. Scalers and models saved."
    }
    summary_path = os.path.join(output_dir, "training_summary.json")
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=4)
    logger.info(f"Saved training summary to {summary_path}")

def plot_and_save_loss(metrics_path, output_dir, model_name):
    if os.path.exists(metrics_path):
        metrics = pd.read_csv(metrics_path)
        metrics = metrics.groupby('epoch').mean().reset_index()

        plt.figure()

        if 'train_loss' in metrics.columns:
            plt.plot(metrics['epoch'], metrics['train_loss'], label='Train Loss')
        if 'val_loss' in metrics.columns:
            plt.plot(metrics['epoch'], metrics['val_loss'], label='Validation Loss')

        plt.title(f'{model_name} Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)
        plt.tight_layout()

        plot_path_png = os.path.join(output_dir, f"{model_name}_loss_plot.png")
        plot_path_pdf = os.path.join(output_dir, f"{model_name}_loss_plot.pdf")
        plt.savefig(plot_path_png)
        plt.savefig(plot_path_pdf)
        plt.close()

        logger.info(f"Saved {model_name} loss plot at {plot_path_png} and {plot_path_pdf}")

def fix_dataset(df):
    logger.info("=== Checking and Fixing Dataset ===")
    constant_cols = [col for col in df.columns if df[col].std() == 0]
    if constant_cols:
        logger.warning(f"Dropping constant columns: {constant_cols}")
        df = df.drop(columns=constant_cols)

    num_nans = df.isna().sum().sum()
    num_infs = np.isinf(df.select_dtypes(include=[np.number]).to_numpy()).sum()

    if num_nans > 0 or num_infs > 0:
        logger.warning(f"Found NaNs: {num_nans}, Infs: {num_infs}. Filling with 0...")
        df = df.replace([np.inf, -np.inf], np.nan).fillna(0)

    return df

def preprocess_df(df):
    try:
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    except Exception as e:
        logger.warning(f"Timestamp parsing failed: {e}")

    if pd.api.types.is_numeric_dtype(df['timestamp']):
        if df['timestamp'].max() > 1e12:
            df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
        else:
            df['datetime'] = pd.to_datetime(df['timestamp'], unit='s')
    else:
        df['datetime'] = df['timestamp']

    df = df.set_index('datetime').sort_index()
    df = df[~df.index.duplicated(keep='first')]
    df = df.asfreq('h')
    df = fix_dataset(df)
    return df.reset_index()

def clip_validation(val_series, train_series):
    train_min = train_series.to_series().min()
    train_max = train_series.to_series().max()
    val_series_clipped = val_series.to_series().clip(lower=train_min, upper=train_max)
    return TimeSeries.from_series(val_series_clipped)

def train_model(model_class, output_dir, train_series_scaled, val_series_scaled, train_past_scaled, val_past_scaled, train_future_scaled, val_future_scaled):
    if model_class == NBEATSModel:
        model_params = {'input_chunk_length': 24, 'num_blocks': 3, 'layer_widths': 512, 'batch_size': 32}
    else:
        model_params = {'input_chunk_length': 24, 'hidden_size': 128, 'lstm_layers': 2, 'batch_size': 32}

    logger.info(f"Setting up {model_class.__name__}...")
    logger_final = CSVLogger(save_dir=output_dir, name=f"{model_class.__name__}_final")
    callbacks = [
        EarlyStopping(monitor="val_loss", patience=10, mode="min"),
        ModelCheckpoint(dirpath=output_dir, filename=f"{model_class.__name__}_best", monitor="val_loss", mode="min")
    ]

    model = model_class(
        output_chunk_length=24,
        likelihood=QuantileRegression(quantiles=[0.05, 0.5, 0.95]),
        n_epochs=1,
        random_state=42,
        pl_trainer_kwargs={
            "accelerator": "gpu" if torch.cuda.is_available() else "cpu",
            "devices": 1,
            "callbacks": callbacks,
            "logger": logger_final,
            "enable_checkpointing": True,
            "gradient_clip_val": 1.0,
            "gradient_clip_algorithm": "norm"
        },
        optimizer_kwargs={"lr": 1e-4},
        **model_params
    )

    if model_class == NBEATSModel:
        model.fit(series=train_series_scaled, past_covariates=train_past_scaled,
                  val_series=val_series_scaled, val_past_covariates=val_past_scaled)
    else:
        model.fit(series=train_series_scaled, past_covariates=train_past_scaled, future_covariates=train_future_scaled,
                  val_series=val_series_scaled, val_past_covariates=val_past_scaled, val_future_covariates=val_future_scaled)

    save_path = os.path.join(output_dir, f"{model_class.__name__}_model.pkl")
    model.save(save_path)
    logger.info(f"{model_class.__name__} model saved at {save_path}")

    metrics_path = os.path.join(output_dir, f"{model_class.__name__}_final/version_0/metrics.csv")
    plot_and_save_loss(metrics_path, output_dir, model_class.__name__)

# --- Main Training Function ---
def main(args):
    os.makedirs(args.output_dir, exist_ok=True)

    train_df = preprocess_df(pd.read_csv(args.train_csv))
    val_df = preprocess_df(pd.read_csv(args.val_csv))

    all_columns = train_df.columns.tolist()
    logger.info(f"All columns detected: {all_columns}")

    ignored_cols = ['timestamp', 'datetime', 'close', 'close_diff']
    usable_cols = [col for col in all_columns if col not in ignored_cols]

    future_features = ['hour', 'dayofweek', 'month', 'quarter', 'fourier_week', 'fourier_year']
    past_features = [col for col in usable_cols if col not in future_features]

    logger.info(f"Using {len(past_features)} past features: {past_features}")
    logger.info(f"Using {len(future_features)} future features: {future_features}")

    train_series = TimeSeries.from_dataframe(train_df, 'datetime', ['close_diff'], freq='h').astype(np.float64)
    val_series = TimeSeries.from_dataframe(val_df, 'datetime', ['close_diff'], freq='h').astype(np.float64)
    train_past = TimeSeries.from_dataframe(train_df, 'datetime', past_features, freq='h').astype(np.float64)
    val_past = TimeSeries.from_dataframe(val_df, 'datetime', past_features, freq='h').astype(np.float64)
    train_future = TimeSeries.from_dataframe(train_df, 'datetime', future_features, freq='h').astype(np.float64)
    val_future = TimeSeries.from_dataframe(val_df, 'datetime', future_features, freq='h').astype(np.float64)

    val_series = clip_validation(val_series, train_series)

    # Scaling
    series_scaler = Scaler()
    train_series_scaled = series_scaler.fit_transform(train_series)
    val_series_scaled = series_scaler.transform(val_series)

    past_scaler = Scaler()
    train_past_scaled = past_scaler.fit_transform(train_past)
    val_past_scaled = past_scaler.transform(val_past)

    future_scaler = Scaler()
    train_future_scaled = future_scaler.fit_transform(train_future)
    val_future_scaled = future_scaler.transform(val_future)

    joblib.dump(series_scaler, os.path.join(args.output_dir, 'series_scaler.pkl'))
    joblib.dump(past_scaler, os.path.join(args.output_dir, 'past_cov_scaler.pkl'))
    joblib.dump(future_scaler, os.path.join(args.output_dir, 'future_cov_scaler.pkl'))

    save_training_summary(args.output_dir, past_features, future_features)

    for model_class in tqdm([NBEATSModel, TFTModel], desc="Training Models"):
        train_model(model_class, args.output_dir, train_series_scaled, val_series_scaled,
                    train_past_scaled, val_past_scaled, train_future_scaled, val_future_scaled)

# --- CLI Entrypoint ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_csv", type=str, required=True, help="Path to training CSV file.")
    parser.add_argument("--val_csv", type=str, required=True, help="Path to validation CSV file.")
    parser.add_argument("--output_dir", type=str, default="models", help="Directory to save models and scalers.")
    args = parser.parse_args()
    main(args)
