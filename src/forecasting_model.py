import pandas as pd
# FIX 1 & 2: Change RNNModel import to BlockRNNModel
from darts.models import BlockRNNModel
from darts.metrics import mape
from sklearn.preprocessing import StandardScaler
from pathlib import Path
import pickle
import torch
import numpy as np
from darts import TimeSeries
from pandas import ExcelWriter


# --- Configuration ---
TICKER = 'AAPL'
# Prediction horizons (in trading days)
HORIZON_1D = 1
HORIZON_1W = 5    # ~1 week of trading days
HORIZON_1M = 21   # ~1 month of trading days

# --- Model Hyperparameters ---
INPUT_CHUNK = 30
OUTPUT_CHUNK = HORIZON_1M
HIDDEN_DIM = 50
N_EPOCHS = 200
BATCH_SIZE = 64
DROPOUT = 0.0
# FIX: Removed TRAINING_LEN = 40 (BlockRNNModel does not accept this argument)
LEARNING_RATE = 1e-4

# Define paths relative to the project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
TRAIN_SERIES_PATH = PROJECT_ROOT / 'data' / 'processed' / f'{TICKER}_TS_train.pkl'
VAL_SERIES_PATH = PROJECT_ROOT / 'data' / 'processed' / f'{TICKER}_TS_val.pkl'
MODEL_PATH = PROJECT_ROOT / 'models' / f'darts_gru_{TICKER}_v1.pkl'
# NEW PATH FOR EXCEL REPORT
EXCEL_OUTPUT_PATH = PROJECT_ROOT / 'reports' / 'aapl_forecast_results.xlsx'


def load_data(train_path: Path, val_path: Path):
    """Loads the processed Darts TimeSeries objects."""
    try:
        with open(train_path, 'rb') as f:
            train_series = pickle.load(f)
        with open(val_path, 'rb') as f:
            val_series = pickle.load(f)
        return train_series, val_series
    except FileNotFoundError:
        print("!!! ERROR: Processed data not found. Run src/data_pipeline.py first.")
        return None, None


def create_and_fit_model(train_series, val_series=None, force_reset=True):
    """Initializes and fits the BlockRNNModel."""
    model = BlockRNNModel(
        model='GRU',
        input_chunk_length=INPUT_CHUNK,
        output_chunk_length=OUTPUT_CHUNK,
        hidden_dim=HIDDEN_DIM,
        n_rnn_layers=1,
        dropout=DROPOUT,
        batch_size=BATCH_SIZE,
        n_epochs=N_EPOCHS,
        # TRAINING_LENGTH argument is correctly REMOVED
        pl_trainer_kwargs={"accelerator": "cpu", "gradient_clip_val": 1.0},
        optimizer_kwargs={"lr": LEARNING_RATE, "eps": 1e-8},
        force_reset=force_reset,
        random_state=42
    )
    model.fit(train_series, val_series=val_series, verbose=True)
    return model


def train_and_forecast():
    """Trains the GRU model, makes predictions, and calculates MAPE."""
    train_series, val_series = load_data(TRAIN_SERIES_PATH, VAL_SERIES_PATH)

    if train_series is None or val_series is None:
        return

    # --- DATA CLEANING ---
    print("-> Cleaning data: filling NaN/Inf values using direct Pandas methods...")

    train_series = TimeSeries.from_series(
        train_series.to_series().fillna(method='ffill').fillna(method='bfill')
    )
    val_series = TimeSeries.from_series(
        val_series.to_series().fillna(method='ffill').fillna(method='bfill')
    )

    if np.any(np.isnan(train_series.values())) or np.any(np.isinf(train_series.values())):
        print("!!! FATAL ERROR: Data is entirely NaN or Inf even after cleaning. Check raw data source.")
        return

    # --- SCALING ---
    print("-> Scaling data using Scikit-learn StandardScaler directly...")

    train_values = train_series.values().reshape(-1, 1)
    val_values = val_series.values().reshape(-1, 1)

    scaler = StandardScaler()
    scaler.fit(train_values)

    train_scaled_values = scaler.transform(train_values)
    val_scaled_values = scaler.transform(val_values)

    train_scaled = TimeSeries.from_series(
        pd.Series(train_scaled_values.flatten(), index=train_series.time_index)
    )
    val_scaled = TimeSeries.from_series(
        pd.Series(val_scaled_values.flatten(), index=val_series.time_index)
    )

    train_vals = train_scaled.values().flatten()
    print(f"-> Scaled Data Mean: {np.mean(train_vals):.4f}, Std: {np.std(train_vals):.4f}")

    # --- MODEL TRAINING ---
    print(f"\n-> Initializing BlockRNNModel (GRU) with Input={INPUT_CHUNK}, Output={OUTPUT_CHUNK}, Epochs={N_EPOCHS}")

    model = create_and_fit_model(train_scaled, val_series=val_scaled, force_reset=True)

    print("-> Training complete.")
    model.save(str(MODEL_PATH))
    print(f"-> Model saved to: {MODEL_PATH}")

    # --- VALIDATION FORECAST ---
    forecast_scaled_1m = model.predict(n=HORIZON_1M)

    forecast_values_1m = scaler.inverse_transform(forecast_scaled_1m.values())
    forecast_1m = TimeSeries.from_series(
        pd.Series(forecast_values_1m.flatten(), index=forecast_scaled_1m.time_index)
    )

    actual_1m = val_series[:HORIZON_1M]
    forecast_1d = forecast_1m[:HORIZON_1D]
    forecast_1w = forecast_1m[:HORIZON_1W]
    actual_1d = val_series[:HORIZON_1D]
    actual_1w = val_series[:HORIZON_1W]

    print("\n--- Model Evaluation (MAPE %) ---")
    mape_1d = mape(actual_1d, forecast_1d)
    mape_1w = mape(actual_1w, forecast_1w)
    mape_1m = mape(actual_1m, forecast_1m)

    print(f"MAPE (1 Day Ahead): {mape_1d:.4f}%")
    print(f"MAPE (1 Week Ahead): {mape_1w:.4f}%")
    print(f"MAPE (1 Month Ahead): {mape_1m:.4f}%")

    print("\n--- Validation Predicted Prices ---")
    print(f"1-Day Forecast: {forecast_1d.values().flatten()[0]:.2f}")
    print(f"1-Week Forecast (Day 5): {forecast_1w.values().flatten()[-1]:.2f}")
    print(f"1-Month Forecast (Day 21): {forecast_1m.values().flatten()[-1]:.2f}")

    # --- EXCEL EXPORT ---
    print("\n-> Exporting validation results to Excel...")

    df_actual = actual_1m.to_dataframe()
    df_actual.columns = ['Actual Price']

    df_forecast = forecast_1m.to_dataframe()
    df_forecast.columns = ['Forecasted Price']

    df_validation_results = pd.concat([df_actual, df_forecast], axis=1)

    # --- LIVE 2025 FORECAST ---
    print("\n-> Running final model training on full data for 2025 Live Forecast...")

    full_series = train_series.concatenate(val_series)
    full_values = full_series.values().reshape(-1, 1)

    final_scaler = StandardScaler()
    final_scaler.fit(full_values)
    full_scaled_values = final_scaler.transform(full_values)

    full_scaled = TimeSeries.from_series(
        pd.Series(full_scaled_values.flatten(), index=full_series.time_index)
    )

    final_model = create_and_fit_model(full_scaled, force_reset=True)

    live_forecast_scaled = final_model.predict(n=HORIZON_1M)
    live_forecast_values = final_scaler.inverse_transform(live_forecast_scaled.values())
    live_forecast_1m = TimeSeries.from_series(
        pd.Series(live_forecast_values.flatten(), index=live_forecast_scaled.time_index)
    )

    df_live_forecast = live_forecast_1m.to_dataframe()
    df_live_forecast.columns = ['2025 Live Forecast Price']

    print("\n--- 2025 Live Forecast Prices ---")
    print(f"1-Day Forecast: {df_live_forecast.iloc[0]['2025 Live Forecast Price']:.2f}")
    print(f"1-Week Forecast (Day 5): {df_live_forecast.iloc[HORIZON_1W - 1]['2025 Live Forecast Price']:.2f}")
    print(f"1-Month Forecast (Day 21): {df_live_forecast.iloc[HORIZON_1M - 1]['2025 Live Forecast Price']:.2f}")

    # --- FINAL EXCEL WRITE ---
    try:
        EXCEL_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

        with ExcelWriter(EXCEL_OUTPUT_PATH) as writer:
            # Sheet 1: Historical Validation (2023)
            df_validation_results.to_excel(
                writer,
                sheet_name='Validation_Results',
                index=True
            )
            # Sheet 2: Live Forecast (2025)
            df_live_forecast.to_excel(
                writer,
                sheet_name='2025_Live_Forecast',
                index=True
            )

        print(f"\n--- SUCCESS: Results exported to Excel at: {EXCEL_OUTPUT_PATH} (2 Sheets) ---")

    except Exception as e:
        print(f"!!! ERROR: Could not export to Excel. Check openpyxl installation. Error: {e}")

    return {
        "mape_1d": mape_1d,
        "mape_1w": mape_1w,
        "mape_1m": mape_1m,
    }


if __name__ == "__main__":
    train_and_forecast()
