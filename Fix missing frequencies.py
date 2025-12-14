import pandas as pd
from pathlib import Path

# --- CONFIG ---
BASE_DIR = Path(__file__).resolve().parent
INPUT_CSV  = BASE_DIR / "FCR_Energy_2024_15min.csv"
OUTPUT_CSV = BASE_DIR / "FCR_Energy_2024_15min_full.csv"

TIME_COL = "Time_Slot_Start"   # name of your datetime column
FREQ = "15min"

df = pd.read_csv(INPUT_CSV)

# parse timezone-aware, then remove tz
df[TIME_COL] = pd.to_datetime(df[TIME_COL], utc=True, errors="raise")
df[TIME_COL] = df[TIME_COL].dt.tz_localize(None)

df = df.set_index(TIME_COL).sort_index()

assert isinstance(df.index, pd.DatetimeIndex), "Index is not DatetimeIndex"

year = df.index.year.unique()
assert len(year) == 1, f"Data spans multiple years: {year}"
year = year[0]

start = df.index.min().floor("D")
end   = df.index.max().ceil("D") - pd.Timedelta(minutes=15)

full_index = pd.date_range(start=start, end=end, freq=FREQ)

assert full_index.max().year == year, (
    f"Index overflow detected: {full_index.max()} (expected year {year})"
)

df_full = df.reindex(full_index).fillna(0)

df_full = df_full.reset_index().rename(columns={"index": TIME_COL})

df_full.to_csv(OUTPUT_CSV, index=False)