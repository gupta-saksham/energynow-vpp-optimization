import requests
import pandas as pd
import time
import numpy as np
from datetime import date, timedelta
from pathlib import Path

from .paths import DATA_DIR

# --- KONFIGURATION ---
YEAR = 2024
FILE_FREQ = DATA_DIR / "frequency_2024_1min.csv"
FILE_MODEL = DATA_DIR / "FCR_Energy_2024_15min.csv"


# FCR Parameter
DEADBAND = 0.010   # 10 mHz
FULL_POWER = 0.200 # 200 mHz


def calculate_fcr_factors(freq_series):
    delta_f = freq_series - 50.0
    slope = FULL_POWER - DEADBAND

    conditions = [
        (delta_f > DEADBAND) & (delta_f < FULL_POWER),
        (delta_f >= FULL_POWER),
        (delta_f < -DEADBAND) & (delta_f > -FULL_POWER),
        (delta_f <= -FULL_POWER),
        (delta_f.abs() <= DEADBAND)
    ]

    choices = [
        (delta_f - DEADBAND) / slope,
        1.0,
        (delta_f + DEADBAND) / slope,
        -1.0,
        0.0
    ]

    return pd.Series(
        np.select(conditions, choices, default=0.0),
        index=freq_series.index
    )


def ensure_output_files_exist():
    """Create CSV files with headers if they do not exist."""

    if not FILE_FREQ.exists():
        pd.DataFrame(
            columns=["Datum (MEZ)", "Frequenz (Netzfrequenz)"]
        ).to_csv(FILE_FREQ, index=False)

    if not FILE_MODEL.exists():
        pd.DataFrame(
            columns=[
                "Time_Slot_Start",
                "FCR_Power_Factor_Up_Sum",
                "FCR_Power_Factor_Down_Sum"
            ]
        ).to_csv(FILE_MODEL, index=False)


def fetch_and_process_all_2024_final():
    base_url = "https://api.energy-charts.info/frequency"

    ensure_output_files_exist()

    start_date = date(2024, 1, 1)
    end_date = date(YEAR, 12, 31)
    current_date = start_date

    print(f"Starte Prozess für {YEAR}...")

    while current_date <= end_date:
        day_str = current_date.strftime("%Y-%m-%d")

        params = {
            "country": "de",
            "start": day_str,
            "end": day_str
        }

        try:
            print(f"Verarbeite: {day_str} ...", end="\r")

            response = requests.get(base_url, params=params, timeout=30)

            if response.status_code != 200:
                raise RuntimeError(f"HTTP {response.status_code}")

            json_resp = response.json()

            if 'unix_seconds' not in json_resp or 'data' not in json_resp:
                raise RuntimeError("Invalid API payload")

            # --- BASIS DATAFRAME ---
            df = pd.DataFrame({
                'timestamp': pd.to_datetime(
                    json_resp['unix_seconds'], unit='s', utc=True
                ).tz_convert('Europe/Berlin'),
                'frequency': json_resp['data']
            }).set_index('timestamp')

            # --- TEIL 1: FREQUENZ (Minutenwerte) ---
            df_freq_1min = df[['frequency']].resample('1min').mean()

            # ---------- COMPLETENESS CHECK ----------
            if len(df_freq_1min) != 1440:
                raise ValueError(
                    f"Incomplete minute data: {len(df_freq_1min)} / 1440"
                )

            save_freq = pd.DataFrame()
            save_freq["Datum (MEZ)"] = df_freq_1min.index.map(lambda x: x.isoformat())
            save_freq["Frequenz (Netzfrequenz)"] = df_freq_1min["frequency"].round(4).values

            save_freq.to_csv(FILE_FREQ, mode="a", header=False, index=False)

            # --- TEIL 2: MODELL (15 Min Summen) ---
            df["p_factor"] = calculate_fcr_factors(df["frequency"])
            df["p_up"] = df["p_factor"].apply(lambda x: x if x > 0 else 0)
            df["p_down"] = df["p_factor"].apply(lambda x: abs(x) if x < 0 else 0)

            df_model_15min = df[["p_up", "p_down"]].resample("15min").sum()

            if len(df_model_15min) != 96:
                raise ValueError(
                    f"Incomplete 15-min data: {len(df_model_15min)} / 96"
                )

            save_model = pd.DataFrame()
            save_model["Time_Slot_Start"] = df_model_15min.index.map(lambda x: x.isoformat())
            save_model["FCR_Power_Factor_Up_Sum"] = df_model_15min["p_up"].round(4).values
            save_model["FCR_Power_Factor_Down_Sum"] = df_model_15min["p_down"].round(4).values

            save_model.to_csv(FILE_MODEL, mode="a", header=False, index=False)

            print(f"✔ {day_str} OK")

        except Exception as e:
            print(f"\n❌ Fehler bei {day_str}: {e}")

        current_date += timedelta(days=1)
        time.sleep(0.3)

    print("\nFertig! Bitte prüfen Sie jetzt die CSV Dateien.")


if __name__ == "__main__":
    fetch_and_process_all_2024_final()
