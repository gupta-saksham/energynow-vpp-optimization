import requests
import pandas as pd
import numpy as np
import time
from datetime import date, timedelta
from pathlib import Path

# ================= CONFIG =================
YEAR = 2024
BASE_DIR = Path(__file__).resolve().parent
FILE_FREQ = BASE_DIR / "frequency_2024_1min.csv"
FILE_MODEL = BASE_DIR / "FCR_Energy_2024_15min.csv"

DEADBAND = 0.010
FULL_POWER = 0.200

MAX_RETRIES = 10
RETRY_SLEEP_BASE = 2.0   # seconds
# =========================================


def calculate_fcr_factors(freq):
    delta_f = freq - 50.0
    slope = FULL_POWER - DEADBAND

    return np.where(
        delta_f > FULL_POWER, 1.0,
        np.where(
            delta_f < -FULL_POWER, -1.0,
            np.where(
                np.abs(delta_f) <= DEADBAND, 0.0,
                np.where(
                    delta_f > 0,
                    (delta_f - DEADBAND) / slope,
                    (delta_f + DEADBAND) / slope
                )
            )
        )
    )


def fetch_day_with_retry(base_url, day):
    """Fetch one day of data with retry & backoff."""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            r = requests.get(
                base_url,
                params={"country": "de", "start": day, "end": day},
                timeout=30
            )
            r.raise_for_status()
            js = r.json()

            if (
                js.get("data")
                and js.get("unix_seconds")
                and len(js["data"]) == len(js["unix_seconds"])
            ):
                return js

            raise ValueError("Empty or incomplete payload")

        except Exception as e:
            wait = RETRY_SLEEP_BASE * attempt
            print(f"  Retry {attempt}/{MAX_RETRIES} failed: {e} → sleeping {wait:.1f}s")
            time.sleep(wait)

    raise RuntimeError(f"FAILED after {MAX_RETRIES} retries for {day}")


def fetch_and_process_all_2024():
    base_url = "https://api.energy-charts.info/frequency"

    # ---------- CREATE OUTPUT FILES ----------
    pd.DataFrame(
        columns=["Datum (MEZ)", "Frequenz (Netzfrequenz)"]
    ).to_csv(FILE_FREQ, index=False)

    pd.DataFrame(
        columns=[
            "Time_Slot_Start",
            "FCR_Power_Factor_Up_Sum",
            "FCR_Power_Factor_Down_Sum"
        ]
    ).to_csv(FILE_MODEL, index=False)

    start = date(2024, 1, 1)
    end = date(2024, 12, 31)

    while start <= end:
        day = start.strftime("%Y-%m-%d")
        print(f"Processing {day}")

        try:
            js = fetch_day_with_retry(base_url, day)

            # ---------- BASE DATA ----------
            df = pd.DataFrame({
                "timestamp": pd.to_datetime(
                    js["unix_seconds"], unit="s", utc=True
                ).tz_convert("Europe/Berlin"),
                "frequency": js["data"]
            }).set_index("timestamp")

            # ---------- FORCE FULL DAY GRID ----------
            full_index = pd.date_range(
                start=pd.Timestamp(day, tz="Europe/Berlin"),
                end=pd.Timestamp(day, tz="Europe/Berlin") + pd.Timedelta(days=1) - pd.Timedelta(minutes=1),
                freq="1min"
            )

            freq_1min = (
                df["frequency"]
                .resample("1min")
                .mean()
                .reindex(full_index)
                .interpolate("time")
            )

            # ---------- HARD VALIDATION ----------
            if len(freq_1min) != 1440:
                raise ValueError(f"Expected 1440 minutes, got {len(freq_1min)}")

            # ---------- WRITE FREQUENCY ----------
            pd.DataFrame({
                "Datum (MEZ)": freq_1min.index.astype(str),
                "Frequenz (Netzfrequenz)": freq_1min.round(4)
            }).to_csv(FILE_FREQ, mode="a", header=False, index=False)

            # ---------- FCR MODEL ----------
            p = calculate_fcr_factors(freq_1min.values)
            model_15 = pd.DataFrame(
                {
                    "p_up": np.clip(p, 0, None),
                    "p_down": np.clip(-p, 0, None)
                },
                index=freq_1min.index
            ).resample("15min").sum()

            if len(model_15) != 96:
                raise ValueError(f"Expected 96 slots, got {len(model_15)}")

            pd.DataFrame({
                "Time_Slot_Start": model_15.index.astype(str),
                "FCR_Power_Factor_Up_Sum": model_15["p_up"].round(4),
                "FCR_Power_Factor_Down_Sum": model_15["p_down"].round(4)
            }).to_csv(FILE_MODEL, mode="a", header=False, index=False)

            print("  OK")

            start += timedelta(days=1)
            time.sleep(0.5)

        except Exception as e:
            print(f"  DAY FAILED → will retry later: {e}")
            time.sleep(5)

    print("\nFinished successfully.")


if __name__ == "__main__":
    fetch_and_process_all_2024()