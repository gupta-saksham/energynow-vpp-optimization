#%%
import numpy as np
import pandas as pd
from pathlib import Path
this_file = Path(__file__).parent
#%%
#Data Loading 
def load_and_process_data(this_file, specific_load = 'LG 18'):
    print("--- Loading and Processing Data ---")
    
    day_ahead_df = pd.read_csv(this_file / "Day ahead.csv")
    
    day_ahead_df['Start date'] = pd.to_datetime(
        day_ahead_df['Start date'], 
        format="%d/%m/%Y %H:%M"
    )
    day_ahead_df.set_index('Start date', inplace=True)
    
    # Convert MWh -> kWh
    day_ahead = day_ahead_df["Germany/Luxembourg [Eur/MWh]"] / 1000 
    
    # --- 2. Load FCR (Block Data) ---
    fcr_df = pd.read_csv(this_file / "FCR prices 2024.csv", sep=";", decimal=",")
    fcr_df = fcr_df[fcr_df["TENDER_NUMBER"] == 1].copy()
    
    fcr_df["price"] = (
        fcr_df["GERMANY_SETTLEMENTCAPACITY_PRICE_[EUR/MW]"]
        .astype(str)
        .str.replace(",", ".", regex=False)
        .astype(float)
    )
    
    # Create the FCR Index (Start times of the blocks)
    # We ensure this starts exactly when the Day Ahead starts
    start_time = day_ahead.index[0]
    fcr_block_index = pd.date_range(start=start_time, periods=len(fcr_df), freq="4h")
    fcr_df.index = fcr_block_index
    fcr_prices = fcr_df["price"].reindex(day_ahead.index, method='ffill')
    
    # --- 4. Normalize Units ---
    # Assumption: Price is EUR/MW (per hour implied). Convert to EUR/kW.
    fcr_prices_normalized = fcr_prices / 1000.0
    
    # --- 5. Load Demand ---
    load_df = pd.read_csv(this_file / "Load profile.csv", index_col='Time stamp')
    load_df.index = pd.to_datetime(load_df.index, format='%d/%m/%Y %H:%M')
    load_df.sort_index(inplace=True)
    load_df.index = load_df.index - pd.Timedelta('15min')

    if load_df.index[0].hour != 0 or load_df.index[0].minute != 0:
        print(f"WARNING: Data starts at {load_df.index[0]} instead of 00:00:00. Check your source file.")
    else:
        print(f"Success: First timestep aligned to {load_df.index[0]}")

    load_index = pd.date_range(start=start_time, periods=len(load_df), freq="15min")
    load_series = pd.Series(load_df[specific_load].values * 3, index=load_index)
    
    # Align to Master Timeline
    load = load_series.reindex(day_ahead.index).fillna(0)

    print(f"Data Successfully Aligned.")
    print(f"Start Time: {day_ahead.index[0]}")
    print(f"End Time:   {day_ahead.index[-1]}")
    print(f"Total Steps: {len(day_ahead)}")
    
    return fcr_prices_normalized.values, day_ahead.values, load.values

#generate random activation for FCR
def generate_fcr_activation_profile(num_steps, block_size, eta_ch, eta_dis, seed=None):
    """
    Build a random FCR activation profile that triggers exactly once per block.
    A positive activation means charging (absorbing power), negative means discharging.
    """
    rng = np.random.default_rng(seed)
    profile = np.zeros(num_steps)
    num_blocks = int(np.ceil(num_steps / block_size))
    for block in range(num_blocks):
        start = block * block_size
        end = min(start + block_size, num_steps)
        activation_idx = rng.integers(start, end)
        direction = rng.choice([-1, 1])
        profile[activation_idx] = eta_ch if direction > 0 else -1.0 / eta_dis
    return profile


#old function
def load_fcr_prices():
    fcr_df = pd.read_csv(this_file / "FCR prices 2024.csv", sep=";", decimal=",")
    
    fcr_df = fcr_df[fcr_df["TENDER_NUMBER"] == 1]
    

    fcr_df["GERMANY_SETTLEMENTCAPACITY_PRICE_[EUR/MW]"] = (
        fcr_df["GERMANY_SETTLEMENTCAPACITY_PRICE_[EUR/MW]"]
        .str.replace(",", ".", regex=False)
        .astype(float)
    )

    fcr_prices = fcr_df["GERMANY_SETTLEMENTCAPACITY_PRICE_[EUR/MW]"]/1000 #EUR/kW
    print(fcr_prices)
    return fcr_prices
#old function
def load_day_ahead_prices():
    day_ahead_df = pd.read_csv(this_file / "Day ahead.csv")
    day_ahead = day_ahead_df["Germany/Luxembourg [Eur/MWh]"]/1000 # to get EUR/KWh
    return day_ahead



# %%
