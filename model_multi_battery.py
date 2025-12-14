#%%
"""
Multi-Battery VPP Optimization Model
=====================================
Each customer has a dedicated battery with BTM/FTM split:
- BTM: Serves local load (behind customer's meter)
- FTM: Aggregated for FCR bidding (1 MW minimum)
"""

from pyomo.environ import *
import pandas as pd
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Optional

this_file = Path(__file__).parent

# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class BatterySpec:
    """Specification for a single battery unit."""
    name: str
    E_max: float        # Energy capacity (kWh)
    P_max: float        # Power capacity (kW)
    eta_ch: float       # Charging efficiency
    eta_dis: float      # Discharging efficiency
    I0: float           # Investment cost (EUR)
    V_bat: float        # Nominal voltage (V)

@dataclass  
class SiteConfig:
    """Configuration for a single customer site."""
    site_id: str
    load_column: str    # Column name in Load profile.csv
    battery: BatterySpec
    btm_ratio: float    # Fraction for BTM (rest goes to FTM)
    P_buy_max: float    # Max grid import (kW)
    P_sell_max: float   # Max grid export (kW)

# Default Luna 2000-215 battery specs
LUNA_2000_215 = BatterySpec(
    name="Luna 2000-215",
    E_max=215.0,
    P_max=108.0,
    eta_ch=0.974,
    eta_dis=0.974,
    I0=73000.0,
    V_bat=777.0
)

# =============================================================================
# DATA LOADING
# =============================================================================

def filter_data_by_date_range(
    data: Dict,
    start_date: str = None,
    end_date: str = None
) -> Dict:
    """
    Filter loaded data to a specific date range.
    
    Args:
        data: Dictionary from load_multi_site_data()
        start_date: Start date string (e.g., "2024-01-15" or "2024-01-15 00:00")
        end_date: End date string (e.g., "2024-02-15" or "2024-02-15 23:45")
    
    Returns:
        Filtered data dictionary with updated indices
    """
    time_index = data['time_index']
    
    # Ensure time_index is timezone-naive for consistent comparison
    if hasattr(time_index, 'tz') and time_index.tz is not None:
        time_index = time_index.tz_localize(None)
    
    # Parse dates (timezone-naive)
    if start_date is not None:
        start_dt = pd.to_datetime(start_date)
        # Remove timezone if present
        if hasattr(start_dt, 'tz') and start_dt.tz is not None:
            start_dt = start_dt.tz_localize(None)
    else:
        start_dt = time_index[0]
    
    if end_date is not None:
        end_dt = pd.to_datetime(end_date)
        # Remove timezone if present
        if hasattr(end_dt, 'tz') and end_dt.tz is not None:
            end_dt = end_dt.tz_localize(None)
    else:
        end_dt = time_index[-1]
    
    # Create mask for the date range
    mask = (time_index >= start_dt) & (time_index <= end_dt)
    indices = np.where(mask)[0]
    
    if len(indices) == 0:
        raise ValueError(f"No data found in range {start_dt} to {end_dt}. "
                        f"Available range: {time_index[0]} to {time_index[-1]}")
    
    start_idx = indices[0]
    end_idx = indices[-1] + 1  # +1 for slicing
    
    # Filter all data arrays
    filtered_data = {
        'day_ahead': data['day_ahead'][start_idx:end_idx],
        'industrial_tarifs': data['industrial_tarifs'][start_idx:end_idx],
        'fcr_prices': data['fcr_prices'][start_idx:end_idx],
        'site_loads': {k: v[start_idx:end_idx] for k, v in data['site_loads'].items()},
        'scale_factors': data['scale_factors'],
        'T': end_idx - start_idx - 1,
        'num_steps': end_idx - start_idx,
        'time_index': time_index[start_idx:end_idx],
        'start_date': start_dt,
        'end_date': end_dt
    }
    
    print(f"\n📅 Date Range Filter Applied:")
    print(f"   From: {filtered_data['time_index'][0]}")
    print(f"   To:   {filtered_data['time_index'][-1]}")
    print(f"   Timesteps: {filtered_data['num_steps']} ({filtered_data['num_steps'] / 96:.1f} days)")
    
    return filtered_data


def load_multi_site_data(
    data_dir: Path,
    site_configs: List[SiteConfig],
    scale_loads_to_battery: bool = True,
    scaler_input: float = 1.0,
    start_date: str = None,
    end_date: str = None
) -> Dict:
    """
    Load and process data for multiple sites.
    
    Args:
        data_dir: Directory containing data files
        site_configs: List of SiteConfig for each site
        scale_loads_to_battery: If True, scale each load profile to match battery power
        scaler_input: Additional scaling factor for loads
        start_date: Optional start date (e.g., "2024-03-01" or "2024-03-01 08:00")
        end_date: Optional end date (e.g., "2024-03-31" or "2024-03-31 23:45")
        
    Returns:
        Dictionary with processed data arrays
        
    Examples:
        # Load full year
        data = load_multi_site_data(data_dir, configs)
        
        # Load specific month
        data = load_multi_site_data(data_dir, configs, 
                                    start_date="2024-06-01", 
                                    end_date="2024-06-30")
        
        # Load specific week
        data = load_multi_site_data(data_dir, configs,
                                    start_date="2024-07-15",
                                    end_date="2024-07-21")
    """
    print("=== Loading Multi-Site Data ===")
    
    # --- 1. Load Day-Ahead Prices ---
    day_ahead_df = pd.read_csv(data_dir / "Day ahead.csv")
    day_ahead_df['Start date'] = pd.to_datetime(
        day_ahead_df['Start date'], 
        format="%d/%m/%Y %H:%M"
    )
    day_ahead_df.set_index('Start date', inplace=True)
    
    # Convert EUR/MWh to EUR/kWh
    day_ahead = day_ahead_df["Germany/Luxembourg [Eur/MWh]"] / 1000.0

    # Load industrial tarifs
    industrial_tarifs_df = pd.read_csv(data_dir / "industrial_tarrifs.csv")
    industrial_tarifs_averages = industrial_tarifs_df.iloc[:,2:].mean()
    tarifs_15min = industrial_tarifs_averages.loc[industrial_tarifs_averages.index.repeat(4)].reset_index(drop=True)
    tarifs_year = pd.concat([tarifs_15min] * 366, ignore_index=True)
    tarifs_year = tarifs_year / 1000 # Convert EUR/MWh to EUR/kWh
    
    # --- 2. Load FCR Prices (4-hour blocks) ---
    fcr_df = pd.read_csv(data_dir / "FCR prices 2024.csv", sep=";", decimal=",")
    fcr_df = fcr_df[fcr_df["TENDER_NUMBER"] == 1].copy()
    
    fcr_df["price"] = (
        fcr_df["GERMANY_SETTLEMENTCAPACITY_PRICE_[EUR/MW]"]
        .astype(str)
        .str.replace(",", ".", regex=False)
        .astype(float)
    )
    
    # Align FCR to day-ahead timeline
    start_time = day_ahead.index[0]
    fcr_block_index = pd.date_range(start=start_time, periods=len(fcr_df), freq="4h")
    fcr_df.index = fcr_block_index
    fcr_prices = fcr_df["price"].reindex(day_ahead.index, method='ffill')
    
    # Convert EUR/MW to EUR/kW
    fcr_prices_normalized = fcr_prices / 1000.0

    # Normalize to 15min intervals
    fcr_prices_normalized = fcr_prices_normalized / 16.0
    
    # --- 3. Load Demand Profiles ---
    load_df = pd.read_csv(data_dir / "Load profile.csv", index_col='Time stamp')
    load_df.index = pd.to_datetime(load_df.index, format='%d/%m/%Y %H:%M')
    load_df.sort_index(inplace=True)
    load_df.index = load_df.index - pd.Timedelta('15min')
    
    load_index = pd.date_range(start=start_time, periods=len(load_df), freq="15min")
    
    # --- 4. Process Each Site ---
    site_loads = {}
    scale_factors = {}
    
    for config in site_configs:
        raw_load = load_df[config.load_column].values
        
        if scale_loads_to_battery:
            # Scale so peak load ~ battery power capacity
            max_load = raw_load.max()
            if max_load > 0:
                scale = config.battery.P_max / max_load
                scale = scaler_input * scale  # Optional additional scaling
            else:
                scale = 1.0
            scaled_load = raw_load * scale
            scale_factors[config.site_id] = scale
            print(f"  Site {config.site_id}: Scaled by {scale:.2f}x (peak: {max_load:.1f} -> {scaled_load.max():.1f} kW)")
        else:
            scaled_load = raw_load
            scale_factors[config.site_id] = 1.0
        
        # Align to day-ahead timeline
        load_series = pd.Series(scaled_load, index=load_index)
        aligned_load = load_series.reindex(day_ahead.index).fillna(0).values
        site_loads[config.site_id] = aligned_load
    
    T = len(day_ahead) - 1
    num_steps = T + 1
    
    print(f"\nTimeline: {start_time} to {day_ahead.index[-1]}")
    print(f"Total timesteps: {num_steps}")
    print(f"Number of sites: {len(site_configs)}")
    
    full_data = {
        'day_ahead': day_ahead.values,
        'industrial_tarifs': tarifs_year.values,
        'fcr_prices': fcr_prices_normalized.values,
        'site_loads': site_loads,
        'scale_factors': scale_factors,
        'T': T,
        'num_steps': num_steps,
        'time_index': day_ahead.index
    }
    
    # Apply date range filter if specified
    if start_date is not None or end_date is not None:
        return filter_data_by_date_range(full_data, start_date, end_date)
    
    return full_data

def load_fcr_activation_profile(
    data_dir: Path,
    start_date: str = None,
    end_date: str = None
):
    """
    Load FCR activation profile from CSV file.
    
    Args:
        data_dir: Directory containing FCR_Energy_2024_15min.csv
        start_date: Optional start date for filtering
        end_date: Optional end date for filtering
        
    Returns:
        Tuple of (fcr_up, fcr_down) arrays
    """
    # Try both possible filenames
    fcr_file = data_dir / "FCR_Energy_2024_15min.csv"
    if not fcr_file.exists():
        fcr_file = data_dir / "FCR_Energy_2024_15min_full.csv"
    
    fcr_profile_df = pd.read_csv(fcr_file, sep=",")
    
    # Parse timestamps and remove timezone info for consistency
    fcr_profile_df['Time_Slot_Start'] = pd.to_datetime(fcr_profile_df['Time_Slot_Start'], utc=True)
    # Convert to timezone-naive (remove timezone info) for easier comparison
    fcr_profile_df['Time_Slot_Start'] = fcr_profile_df['Time_Slot_Start'].dt.tz_localize(None)
    fcr_profile_df.set_index('Time_Slot_Start', inplace=True)
    
    # Apply date range filter if specified
    if start_date is not None or end_date is not None:
        start_dt = pd.to_datetime(start_date) if start_date else fcr_profile_df.index[0]
        end_dt = pd.to_datetime(end_date) if end_date else fcr_profile_df.index[-1]
        
        # Ensure comparison timestamps are also timezone-naive
        if hasattr(start_dt, 'tz') and start_dt.tz is not None:
            start_dt = start_dt.tz_localize(None)
        if hasattr(end_dt, 'tz') and end_dt.tz is not None:
            end_dt = end_dt.tz_localize(None)
        
        mask = (fcr_profile_df.index >= start_dt) & (fcr_profile_df.index <= end_dt)
        fcr_profile_df = fcr_profile_df[mask]
        
        if len(fcr_profile_df) == 0:
            raise ValueError(f"No FCR data found in range {start_dt} to {end_dt}")
    
    fcr_up = fcr_profile_df["FCR_Power_Factor_Up_Sum"].values
    fcr_down = fcr_profile_df["FCR_Power_Factor_Down_Sum"].values
    
    return fcr_up, fcr_down

def generate_fcr_activation_profile(num_steps: int, block_size: int, 
                                     eta_ch: float, eta_dis: float, 
                                     seed: int = None) -> np.ndarray:
    """
    Generate random FCR activation profile (one activation per 4h block).
    Positive = charging (absorb), Negative = discharging (inject).
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


# =============================================================================
# MULTI-BATTERY PYOMO MODEL
# =============================================================================

def build_multi_battery_model(
    site_configs: List[SiteConfig],
    data: Dict,
    fcr_signal_up: np.ndarray,
    fcr_signal_down: np.ndarray,
    delta_t: float = 0.25,
    SOC0: float = 0.5,
    SOH0: float = 1.0,
    C_peak: float = 192.66,     # EUR/kW peak tariff
    min_fcr_bid: float = 1000.0, # 1 MW minimum FCR bid
    # SOH degradation parameters (small values for linear approximation)
    a: float = 1e-11,
    b: float = 1e-11,
    c: float = 1e-10
) -> ConcreteModel:
    """
    Build multi-site VPP optimization model.
    
    Key features:
    - Each site has dedicated BTM partition for local load
    - FTM partitions are aggregated for FCR bidding
    - 1 MW minimum FCR bid constraint
    - Per-site peak tracking and grid exchange
    """
    
    model = ConcreteModel(name="Multi-Battery VPP")
    
    T = data['T']
    C_buy_BTM = data['industrial_tarifs']
    C_buy_FTM = data['day_ahead']
    C_sell = data['day_ahead']  # Symmetric pricing
    C_FCR = data['fcr_prices']
    site_loads = data['site_loads']
    
    # Extract site IDs
    site_ids = [cfg.site_id for cfg in site_configs]
    
    # ==========================================================================
    # SETS
    # ==========================================================================
    
    model.S = Set(initialize=site_ids, doc="Sites")
    model.T = RangeSet(0, T, doc="Time steps")
    model.Tstep = RangeSet(0, T-1, doc="Transition steps")
    
    # FCR block structure (4 hours = 16 x 15-min steps)
    block_size = int(round(4.0 / delta_t))
    num_blocks = int(np.ceil((T + 1) / block_size))
    model.B = RangeSet(0, num_blocks - 1, doc="FCR blocks")
    
    # ==========================================================================
    # PARAMETERS
    # ==========================================================================
    
    model.delta_t = Param(initialize=delta_t)
    model.min_fcr_bid = Param(initialize=min_fcr_bid)
    model.C_peak = Param(initialize=C_peak)
    
    # Market prices (time-indexed)
    model.C_buy_BTM = Param(model.T, initialize=lambda m, t: C_buy_BTM[t])
    model.C_buy_FTM = Param(model.T, initialize=lambda m, t: C_buy_FTM[t])
    model.C_sell = Param(model.T, initialize=lambda m, t: C_sell[t])
    model.C_FCR = Param(model.T, initialize=lambda m, t: C_FCR[t])
    model.FCR_signal_up = Param(model.T, initialize=lambda m, t: fcr_signal_up[t])
    model.FCR_signal_down = Param(model.T, initialize=lambda m, t: fcr_signal_down[t])
    model.FCR_max_factor = Param(initialize=max(max(fcr_signal_up), max(fcr_signal_down)) * 1.1) # Safety factor of 10 %
    
    # Site-specific parameters
    site_config_map = {cfg.site_id: cfg for cfg in site_configs}
    
    def init_demand(m, s, t):
        return site_loads[s][t]
    model.D = Param(model.S, model.T, initialize=init_demand, doc="Demand per site")
    
    def init_E_max(m, s):
        return site_config_map[s].battery.E_max
    model.E_bat_max = Param(model.S, initialize=init_E_max)
    
    def init_P_max(m, s):
        return site_config_map[s].battery.P_max
    model.P_bat_max = Param(model.S, initialize=init_P_max)
    
    def init_eta_ch(m, s):
        return site_config_map[s].battery.eta_ch
    model.eta_ch = Param(model.S, initialize=init_eta_ch)
    
    def init_eta_dis(m, s):
        return site_config_map[s].battery.eta_dis
    model.eta_dis = Param(model.S, initialize=init_eta_dis)
    
    def init_btm_ratio(m, s):
        return site_config_map[s].btm_ratio
    model.btm_ratio = Param(model.S, initialize=init_btm_ratio)
    
    def init_ftm_ratio(m, s):
        return 1.0 - site_config_map[s].btm_ratio
    model.ftm_ratio = Param(model.S, initialize=init_ftm_ratio)
    
    def init_I0(m, s):
        return site_config_map[s].battery.I0
    model.I0 = Param(model.S, initialize=init_I0)
    
    def init_V_bat(m, s):
        return site_config_map[s].battery.V_bat
    model.V_bat = Param(model.S, initialize=init_V_bat)
    
    def init_P_buy_max(m, s):
        return site_config_map[s].P_buy_max
    model.P_buy_max = Param(model.S, initialize=init_P_buy_max)
    
    def init_P_sell_max(m, s):
        return site_config_map[s].P_sell_max
    model.P_sell_max = Param(model.S, initialize=init_P_sell_max)
    
    # SOH parameters
    model.a = Param(initialize=a)
    model.b = Param(initialize=b)
    model.c = Param(initialize=c)
    
    # ==========================================================================
    # VARIABLES - Per Site
    # ==========================================================================
    
    # BTM partition (serves local load)
    def btm_power_bounds(m, s, t):
        return (0, site_config_map[s].battery.P_max * site_config_map[s].btm_ratio)
    
    model.P_ch_BTM = Var(model.S, model.T, within=NonNegativeReals, bounds=btm_power_bounds)
    model.P_dis_BTM = Var(model.S, model.T, within=NonNegativeReals, bounds=btm_power_bounds)
    
    def btm_soc_bounds(m, s, t):
        return (0, site_config_map[s].battery.E_max * site_config_map[s].btm_ratio)
    model.SOC_BTM = Var(model.S, model.T, within=NonNegativeReals, bounds=btm_soc_bounds)
    
    # FTM partition (for FCR)
    def ftm_power_bounds(m, s, t):
        return (0, site_config_map[s].battery.P_max * (1 - site_config_map[s].btm_ratio))
    
    model.P_ch_FTM = Var(model.S, model.T, within=NonNegativeReals, bounds=ftm_power_bounds)
    model.P_dis_FTM = Var(model.S, model.T, within=NonNegativeReals, bounds=ftm_power_bounds)
    
    def ftm_soc_bounds(m, s, t):
        return (0, site_config_map[s].battery.E_max * (1 - site_config_map[s].btm_ratio))
    model.SOC_FTM = Var(model.S, model.T, within=NonNegativeReals, bounds=ftm_soc_bounds)
    
    # Per-site FCR bid contribution
    model.P_FCR_bid = Var(model.S, model.T, within=NonNegativeReals, bounds=ftm_power_bounds)
    
    # Grid exchange per site
    def P_buy_BTM_bounds(m, s, t):
        return (0, m.P_buy_max[s] * m.btm_ratio[s])
    model.P_buy_BTM = Var(model.S, model.T, within=NonNegativeReals, bounds=P_buy_BTM_bounds)

    def P_buy_FTM_bounds(m, s, t):
        return (0, m.P_buy_max[s] * m.ftm_ratio[s])
    model.P_buy_FTM = Var(model.S, model.T, within=NonNegativeReals, bounds=P_buy_FTM_bounds)
    
    def P_sell_bounds(m, s, t):
        return (0, site_config_map[s].P_sell_max)
    model.P_sell = Var(model.S, model.T, within=NonNegativeReals, bounds=P_sell_bounds)
    
    # Binary for buy/sell exclusivity per site
    model.u_buy = Var(model.S, model.T, within=Binary)

    # Binary for charge/discharge exclusivity per site
    model.u_ch_BTM = Var(model.S, model.T, within=Binary)
    
    # Peak demand per site
    model.P_peak = Var(model.S, within=NonNegativeReals)
    
    # State of Health per site
    model.SOH = Var(model.S, model.T, within=NonNegativeReals, bounds=(0, 1.0))
    model.E_bat_current = Var(model.S, model.T, within=NonNegativeReals)
    
    # ==========================================================================
    # VARIABLES - Aggregated FCR
    # ==========================================================================
    
    # Total FCR bid (sum across all sites)
    model.P_FCR_total = Var(model.T, within=NonNegativeReals)
    
    # Binary: is the VPP participating in FCR this block?
    model.u_FCR = Var(model.T, within=Binary)
    
    # ==========================================================================
    # CONSTRAINTS - Initialization
    # ==========================================================================
    
    def init_soc_btm(m, s):
        E_max = site_config_map[s].battery.E_max
        ratio = site_config_map[s].btm_ratio
        return m.SOC_BTM[s, 0] == SOC0 * E_max * ratio
    model.Init_SOC_BTM = Constraint(model.S, rule=init_soc_btm)

    def end_soc_btm(m, s):
        E_max = site_config_map[s].battery.E_max
        ratio = site_config_map[s].btm_ratio
        return m.SOC_BTM[s, T] == SOC0 * E_max * ratio
    model.End_SOC_BTM = Constraint(model.S, rule=end_soc_btm)
    
    def init_soc_ftm(m, s):
        E_max = site_config_map[s].battery.E_max
        ratio = 1 - site_config_map[s].btm_ratio
        return m.SOC_FTM[s, 0] == SOC0 * E_max * ratio
    model.Init_SOC_FTM = Constraint(model.S, rule=init_soc_ftm)

    def end_soc_ftm(m, s):
        E_max = site_config_map[s].battery.E_max
        ratio = 1 - site_config_map[s].btm_ratio
        return m.SOC_FTM[s, T] == SOC0 * E_max * ratio
    model.End_SOC_FTM = Constraint(model.S, rule=end_soc_ftm)
    
    def init_soh(m, s):
        return m.SOH[s, 0] == SOH0
    model.Init_SOH = Constraint(model.S, rule=init_soh)
    
    # ==========================================================================
    # CONSTRAINTS - SOC Balance
    # ==========================================================================
    
    def soc_balance_btm(m, s, t):
        eta_ch = m.eta_ch[s]
        eta_dis = m.eta_dis[s]
        return m.SOC_BTM[s, t+1] == m.SOC_BTM[s, t] + \
               (eta_ch * m.P_ch_BTM[s, t] - (1/eta_dis) * m.P_dis_BTM[s, t]) * m.delta_t
    model.SOC_Balance_BTM = Constraint(model.S, model.Tstep, rule=soc_balance_btm)
    
    def soc_balance_ftm(m, s, t):
        eta_ch = m.eta_ch[s]
        eta_dis = m.eta_dis[s]
        # FCR activation affects FTM SOC
        # FCR_signal is sum of per-second activation factors over 15 min (range ~0-100)
        # To get effective power: P_FCR_bid * (signal / 900) = average power during slot
        # Positive signal (up) = battery charges (absorbs power)
        # Negative signal (down) = battery discharges (injects power)
        fcr_activation_fraction = (m.FCR_signal_up[t] - m.FCR_signal_down[t]) / 3600.0
        fcr_avg_power = m.P_FCR_bid[s, t] * fcr_activation_fraction  # kW (can be negative)
        return m.SOC_FTM[s, t+1] == m.SOC_FTM[s, t] + \
               (eta_ch * m.P_ch_FTM[s, t] - (1/eta_dis) * m.P_dis_FTM[s, t] + fcr_avg_power) * m.delta_t
    model.SOC_Balance_FTM = Constraint(model.S, model.Tstep, rule=soc_balance_ftm)
    
    # ==========================================================================
    # CONSTRAINTS - Capacity Limits
    # ==========================================================================
    
    def update_capacity(m, s, t):
        return m.E_bat_current[s, t] == m.E_bat_max[s] * m.SOH[s, t]
    model.Update_Cap = Constraint(model.S, model.T, rule=update_capacity)
    
    def max_soc_btm(m, s, t):
        return m.SOC_BTM[s, t] <= m.E_bat_current[s, t] * m.btm_ratio[s]
    model.Max_SOC_BTM = Constraint(model.S, model.T, rule=max_soc_btm)
    
    def max_soc_ftm(m, s, t):
        # Reserve headroom for FCR charging
        # Worst case: full upward activation for entire 15-min slot
        # Energy reserved = P_FCR_bid (kW) * delta_t (h) * eta_ch = kWh
        fcr_headroom = m.P_FCR_bid[s, t] * m.delta_t * m.eta_ch[s] * (m.FCR_max_factor / 3600.0)
        return m.SOC_FTM[s, t] <= (m.E_bat_current[s, t] * m.ftm_ratio[s]) - fcr_headroom
    model.Max_SOC_FTM = Constraint(model.S, model.T, rule=max_soc_ftm)
    
    def min_soc_ftm(m, s, t):
        # Reserve floor for FCR discharging
        # Worst case: full downward activation for entire 15-min slot
        # Energy reserved = P_FCR_bid (kW) * delta_t (h) / eta_dis = kWh
        # FCR_max_factor normalizes: actual activation seconds / 900 seconds per 15min
        fcr_floor = m.P_FCR_bid[s, t] * m.delta_t * (1/m.eta_dis[s]) * (m.FCR_max_factor / 3600.0)
        return m.SOC_FTM[s, t] >= fcr_floor
    model.Min_SOC_FTM = Constraint(model.S, model.T, rule=min_soc_ftm)
    
    # ==========================================================================
    # CONSTRAINTS - Power Balance (Per Site)
    # ==========================================================================
    
    def power_balance_BTM(m, s, t):
        return m.D[s, t] + m.P_ch_BTM[s, t] == m.P_buy_BTM[s, t] + m.P_dis_BTM[s, t]
    model.Power_Balance_BTM = Constraint(model.S, model.T, rule=power_balance_BTM)

    def power_balance_FTM(m, s, t):
        return m.P_ch_FTM[s, t] + m.P_sell[s, t] == m.P_buy_FTM[s, t] + m.P_dis_FTM[s, t]
    model.Power_Balance_FTM = Constraint(model.S, model.T, rule=power_balance_FTM)
    
    # Buy/Sell exclusivity
    def buy_binary_FTM(m, s, t):
        return m.P_buy_FTM[s, t] <= m.u_buy[s, t] * m.P_buy_max[s]
    model.Buy_Binary_FTM = Constraint(model.S, model.T, rule=buy_binary_FTM)
    
    def sell_binary_FTM(m, s, t):
        return m.P_sell[s, t] <= (1 - m.u_buy[s, t]) * m.P_sell_max[s]
    model.Sell_Binary_FTM = Constraint(model.S, model.T, rule=sell_binary_FTM)

    # Charge/Discharge exclusivity
    def ch_binary_BTM(m, s, t):
        return m.P_ch_BTM[s, t] <= m.u_ch_BTM[s, t] * m.P_bat_max[s] * m.btm_ratio[s]
    model.Ch_Binary_BTM = Constraint(model.S, model.T, rule=ch_binary_BTM)
    
    def dis_binary_BTM(m, s, t):
        return m.P_dis_BTM[s, t] <= (1 - m.u_ch_BTM[s, t]) * m.P_bat_max[s] * m.btm_ratio[s]
    model.Dis_Binary_FTM = Constraint(model.S, model.T, rule=dis_binary_BTM)
    
    # ==========================================================================
    # CONSTRAINTS - Peak Tracking
    # ==========================================================================
    
    def peak_rule(m, s, t):
        return m.P_buy_FTM[s, t] + m.P_buy_BTM[s, t] <= m.P_peak[s]
    model.Peak_Def = Constraint(model.S, model.T, rule=peak_rule)
    
    # ==========================================================================
    # CONSTRAINTS - Aggregated FCR (1 MW Minimum)
    # ==========================================================================
    
    # Total FCR bid is sum of all site contributions
    def fcr_total_def(m, t):
        return m.P_FCR_total[t] == sum(m.P_FCR_bid[s, t] for s in m.S)
    model.FCR_Total_Def = Constraint(model.T, rule=fcr_total_def)
    
    # If participating in FCR, must bid at least 1 MW
    def fcr_min_bid(m, t):
        return m.P_FCR_total[t] >= m.u_FCR[t] * m.min_fcr_bid
    model.FCR_Min_Bid = Constraint(model.T, rule=fcr_min_bid)
    
    # Max bid limited by total FTM capacity
    total_ftm_capacity = sum(
        cfg.battery.P_max * (1 - cfg.btm_ratio) 
        for cfg in site_configs
    )
    
    def fcr_max_bid(m, t):
        return m.P_FCR_total[t] <= m.u_FCR[t] * total_ftm_capacity
    model.FCR_Max_Bid = Constraint(model.T, rule=fcr_max_bid)
    
    # FCR bid must be constant within 4-hour blocks
    def fcr_block_consistency(m, t):
        if t == 0 or t % block_size == 0:
            return Constraint.Skip
        return m.u_FCR[t] == m.u_FCR[t-1]
    model.FCR_Block_U = Constraint(model.T, rule=fcr_block_consistency)
    
    def fcr_bid_block_consistency(m, t):
        if t == 0 or t % block_size == 0:
            return Constraint.Skip
        return m.P_FCR_total[t] == m.P_FCR_total[t-1]
    model.FCR_Block_Bid = Constraint(model.T, rule=fcr_bid_block_consistency)
    
    # Per-site FCR bid consistency within blocks
    def fcr_site_block_consistency(m, s, t):
        if t == 0 or t % block_size == 0:
            return Constraint.Skip
        return m.P_FCR_bid[s, t] == m.P_FCR_bid[s, t-1]
    model.FCR_Site_Block = Constraint(model.S, model.T, rule=fcr_site_block_consistency)
    
    # ==========================================================================
    # CONSTRAINTS - SOH Degradation
    # ==========================================================================
    
    def soh_update(m, s, t):
        total_power = (m.P_ch_BTM[s, t] + m.P_dis_BTM[s, t] + 
                       m.P_ch_FTM[s, t] + m.P_dis_FTM[s, t])
        return m.SOH[s, t+1] == m.SOH[s, t] - (m.b * total_power / m.V_bat[s] + m.c * m.SOH[s, t]) * m.delta_t * 3600
    model.SOH_Update = Constraint(model.S, model.Tstep, rule=soh_update)
    
    # ==========================================================================
    # CONSTRAINTS - Daily Cycle Limit (Optional: 2 full cycles/day)
    # ==========================================================================
    
    steps_per_day = int(24 / delta_t)
    num_days = int((T + 1) // steps_per_day)
    model.Days = RangeSet(0, max(0, num_days - 1))
    
    def daily_cycle_limit(m, s, d):
        start_t = d * steps_per_day
        end_t = min(start_t + steps_per_day - 1, T)
        daily_discharge = sum(
            (m.P_dis_BTM[s, t] + m.P_dis_FTM[s, t]) * m.delta_t
            for t in range(start_t, end_t + 1)
        )
        return daily_discharge <= 2.0 * m.E_bat_max[s]
    model.Daily_Cycle_Limit = Constraint(model.S, model.Days, rule=daily_cycle_limit)
    
    # ==========================================================================
    # OBJECTIVE
    # ==========================================================================
    
    def objective_rule(m):
        # Energy costs (per site)
        energy_cost = sum(
            (m.C_buy_FTM[t] * m.P_buy_FTM[s, t] + m.C_buy_BTM[t] * m.P_buy_BTM[s, t] - m.C_sell[t] * m.P_sell[s, t]) * m.delta_t
            for s in m.S for t in m.T
        )
        
        # Peak demand costs (per site)
        peak_cost = sum(m.P_peak[s] * m.C_peak for s in m.S)
        
        # Degradation costs (per site)
        deg_cost = sum(
            m.I0[s] * (SOH0 - m.SOH[s, T]) / (SOH0 - 0.8)
            for s in m.S
        )
        
        # FCR revenue (aggregated - payment is per MW bid)
        fcr_revenue = sum(m.C_FCR[t] * m.P_FCR_total[t] for t in m.T)
        
        return energy_cost + peak_cost + deg_cost - fcr_revenue
    
    model.Obj = Objective(rule=objective_rule, sense=minimize)
    
    return model


# =============================================================================
# RESULTS EXTRACTION
# =============================================================================

def extract_results(model, site_configs: List[SiteConfig], data: Dict) -> pd.DataFrame:
    """Extract optimization results into a DataFrame for analysis."""
    
    records = []
    site_ids = [cfg.site_id for cfg in site_configs]
    site_config_map = {cfg.site_id: cfg for cfg in site_configs}
    
    for t in model.T:
        for s in site_ids:
            cfg = site_config_map[s]
            
            # Power flows
            p_ch_btm = value(model.P_ch_BTM[s, t])
            p_dis_btm = value(model.P_dis_BTM[s, t])
            p_ch_ftm = value(model.P_ch_FTM[s, t])
            p_dis_ftm = value(model.P_dis_FTM[s, t])
            p_fcr_bid = value(model.P_FCR_bid[s, t])
            
            # Grid exchange
            p_buy = value(model.P_buy_FTM[s, t]) + value(model.P_buy_BTM[s, t])
            p_buy_btm = value(model.P_buy_BTM[s, t])
            p_buy_ftm = value(model.P_buy_FTM[s, t])
            p_sell = value(model.P_sell[s, t])
            
            # State
            soc_btm = value(model.SOC_BTM[s, t])
            soc_ftm = value(model.SOC_FTM[s, t])
            soh = value(model.SOH[s, t])
            
            # Market data
            price = value(model.C_buy_FTM[t])
            industrial_price = value(model.C_buy_BTM[t])
            fcr_price = value(model.C_FCR[t])
            demand = value(model.D[s, t])
            
            # Aggregated FCR
            fcr_total = value(model.P_FCR_total[t])
            u_fcr = value(model.u_FCR[t])
            fcr_signal = value(model.FCR_signal_up[t]) - value(model.FCR_signal_down[t]) 
            
            records.append({
                't': t,
                'site': s,
                'demand': demand,
                'P_ch_BTM': p_ch_btm,
                'P_dis_BTM': p_dis_btm,
                'P_ch_FTM': p_ch_ftm,
                'P_dis_FTM': p_dis_ftm,
                'P_FCR_bid': p_fcr_bid,
                'P_buy': p_buy,
                'P_buy_BTM': p_buy_btm,
                'P_buy_FTM': p_buy_ftm,
                'P_sell': p_sell,
                'Grid_Net': p_buy - p_sell,
                'SOC_BTM': soc_btm,
                'SOC_FTM': soc_ftm,
                'SOC_total': soc_btm + soc_ftm,
                'SOH': soh,
                'Price': price,
                'Price industrial':industrial_price,
                'FCR_price': fcr_price,
                'FCR_total': fcr_total,
                'u_FCR': u_fcr,
                'FCR_signal': fcr_signal,
                'P_peak': value(model.P_peak[s]),
                'E_max': cfg.battery.E_max,
                'P_max': cfg.battery.P_max,
            })
    
    df = pd.DataFrame(records)
    
    # Add datetime index
    start_time = data['time_index'][0]
    df['datetime'] = df['t'].apply(lambda x: start_time + pd.Timedelta(minutes=15*x))
    
    # Calculate SOC percentages (handle btm_ratio=0 or btm_ratio=1 edge cases)
    def calc_btm_pct(r):
        btm_capacity = r['E_max'] * site_config_map[r['site']].btm_ratio
        if btm_capacity > 0:
            return 100 * r['SOC_BTM'] / btm_capacity
        return 50  # Default when no BTM capacity
    
    def calc_ftm_pct(r):
        ftm_capacity = r['E_max'] * (1 - site_config_map[r['site']].btm_ratio)
        if ftm_capacity > 0:
            return 100 * r['SOC_FTM'] / ftm_capacity
        return 50  # Default when no FTM capacity
    
    df['SOC_BTM_pct'] = df.apply(calc_btm_pct, axis=1)
    df['SOC_FTM_pct'] = df.apply(calc_ftm_pct, axis=1)
    df['SOC_total_pct'] = df.apply(
        lambda r: 100 * r['SOC_total'] / r['E_max'] if r['E_max'] > 0 else 0, axis=1
    )
    
    return df


def calculate_financials(df: pd.DataFrame, site_configs: List[SiteConfig], 
                         C_peak: float, delta_t: float = 0.25) -> Dict:
    """Calculate financial metrics per site and portfolio-wide."""
    
    site_config_map = {cfg.site_id: cfg for cfg in site_configs}
    financials = {'sites': {}, 'portfolio': {}}
    
    for site_id in df['site'].unique():
        site_df = df[df['site'] == site_id].copy()
        cfg = site_config_map[site_id]
        
        # Energy costs
        energy_cost = ((site_df['P_buy_BTM'] * site_df['Price industrial'] + site_df['P_buy_FTM'] * site_df['Price'])* delta_t).sum()
        energy_revenue = (site_df['P_sell'] * site_df['Price'] * delta_t).sum()
        net_energy = energy_cost - energy_revenue
        
        # Peak cost
        peak_kw = site_df['P_peak'].iloc[0]
        peak_cost = peak_kw * C_peak
        
        # Baseline (no battery)
        baseline_energy = (site_df['demand'] * site_df['Price industrial'] * delta_t).sum()
        baseline_peak = site_df['demand'].max() * C_peak
        
        # BTM savings
        btm_savings = baseline_energy - net_energy
        peak_savings = baseline_peak - peak_cost
        
        # FCR revenue (allocated proportionally)
        site_fcr_frac = site_df['P_FCR_bid'].sum() / max(df['FCR_total'].sum(), 1e-6)
        fcr_revenue = (site_df['FCR_price'] * site_df['P_FCR_bid']).sum()

        # Degradation cost
        soh_start = site_df['SOH'].iloc[0]
        soh_end = site_df['SOH'].iloc[-1]
        deg_cost = cfg.battery.I0 * (soh_start - soh_end) / (soh_start - 0.6)
        
        financials['sites'][site_id] = {
            'energy_cost': net_energy,
            'peak_cost': peak_cost,
            'baseline_energy': baseline_energy,
            'baseline_peak': baseline_peak,
            'btm_savings': btm_savings,
            'peak_savings': peak_savings,
            'fcr_revenue': fcr_revenue,
            'degradation_cost': deg_cost,
            'net_benefit': btm_savings + peak_savings + fcr_revenue - deg_cost,
            'peak_kw_old': site_df['demand'].max(),
            'peak_kw_new': peak_kw,
        }
    
    # Portfolio totals
    portfolio = {k: sum(v[k] for v in financials['sites'].values()) 
                 for k in financials['sites'][list(financials['sites'].keys())[0]].keys()}
    financials['portfolio'] = portfolio
    
    return financials


# =============================================================================
# MAIN EXECUTION BLOCK
# =============================================================================

if __name__ == "__main__":
    from pyomo.environ import SolverFactory
    
    print("=" * 60)
    print("MULTI-BATTERY VPP OPTIMIZATION")
    print("=" * 60)
    
    # =========================================================================
    # CONFIGURATION: Define sites
    # =========================================================================
    
    # Create site configurations (using different load profiles)
    # We need 16 batteries to reach >1 MW aggregated FTM capacity
    # 16 * 108 kW * 0.6 FTM = 1036.8 kW > 1 MW minimum FCR bid
    load_columns = ['LG 01', 'LG 02', 'LG 03', 'LG 04', 'LG 05',
                    'LG 06', 'LG 07', 'LG 08', 'LG 09', 'LG 10',
                    'LG 11', 'LG 12', 'LG 13', 'LG 14', 'LG 15', 'LG 18', 'LG 19', 'LG 20', 'LG 21', 'LG 22', 'LG 23', 'LG 24','LG 25','LG 26','LG 27','LG 28','LG 29','LG 30']
    
    site_configs = []
    for i, col in enumerate(load_columns):
        site_configs.append(SiteConfig(
            site_id=f"Site_{i+1:02d}",
            load_column=col,
            battery=BatterySpec(
                name=f"Luna_2000_215_{i+1}",
                E_max=215.0,       # kWh capacity
                P_max=108.0,       # kW power rating
                eta_ch=0.974,      # Round-trip efficiency split
                eta_dis=0.974,
                I0=73000.0,        # Investment cost EUR
                V_bat=777.0        # Nominal voltage
            ),
            btm_ratio=0.4,         # 40% BTM (local load), 60% FTM (FCR)
            P_buy_max=2000.0,       # Max grid import kW
            P_sell_max=2000.0       # Max grid export kW
        ))
    
    print(f"\nConfigured {len(site_configs)} sites")
    total_ftm = sum(cfg.battery.P_max * (1 - cfg.btm_ratio) for cfg in site_configs)
    total_btm = sum(cfg.battery.P_max * cfg.btm_ratio for cfg in site_configs)
    total_capacity = sum(cfg.battery.E_max for cfg in site_configs)
    print(f"Total battery capacity: {total_capacity:.0f} kWh")
    print(f"Total BTM power: {total_btm:.0f} kW")
    print(f"Total FTM power: {total_ftm:.0f} kW")
    print(f"FCR minimum bid: 1000 kW")
    
    if total_ftm < 1000:
        print("\n⚠️  WARNING: Total FTM capacity below 1 MW!")
        print("   Consider adding more batteries or reducing btm_ratio.")
        # Auto-adjust if needed
        required_ratio = 1 - (1000 / (len(site_configs) * 108))
        print(f"   Minimum btm_ratio for 1MW: {required_ratio:.2f}")
    else:
        print(f"✓ FCR capacity sufficient ({total_ftm:.0f} kW > 1000 kW)")
    
    # =========================================================================
    # DATE RANGE CONFIGURATION
    # =========================================================================
    # Option 1: Specify exact dates (recommended)
    # Option 2: Set both to None for full year (warning: may take hours)
    
    # Examples:
    #   START_DATE = "2024-01-01"    # Start of year
    #   END_DATE = "2024-01-31"      # End of January (1 month)
    #
    #   START_DATE = "2024-06-15"    # Mid-June
    #   END_DATE = "2024-06-21"      # One week
    #
    #   START_DATE = None            # From beginning of data
    #   END_DATE = None              # To end of data (full year)
    
    START_DATE = "2024-01-01"   # Start date (YYYY-MM-DD or YYYY-MM-DD HH:MM)
    END_DATE = "2024-01-07"     # End date - 1 week test
    
    # Peak tariff (EUR/kW/year) - will be prorated for simulation period
    C_peak_annual = 192.66
    
    # =========================================================================
    # LOAD DATA WITH DATE RANGE
    # =========================================================================
    
    data = load_multi_site_data(
        data_dir=this_file,
        site_configs=site_configs,
        scale_loads_to_battery=True,
        scaler_input=1.0,
        start_date=START_DATE,
        end_date=END_DATE
    )
    print([name for name in data])
    # Prorate peak tariff based on simulation period
    # (Peak charges are typically annual, so we scale for shorter periods)
    simulation_days = data['num_steps'] / 96  # 96 steps per day
    C_peak = C_peak_annual * (simulation_days / 365.0)
    print(f"   Peak tariff prorated: €{C_peak_annual:.2f}/kW/year → €{C_peak:.2f}/kW for {simulation_days:.1f} days")
    
    # Load FCR activation profile for the same date range
    delta_t = 0.25
    fcr_up, fcr_down = load_fcr_activation_profile(
        data_dir=this_file,
        start_date=START_DATE,
        end_date=END_DATE
    )
    
    # Handle FCR array length mismatch (pad or truncate to match data)
    num_steps = data['num_steps']
    if len(fcr_up) >= num_steps:
        fcr_up = fcr_up[:num_steps]
        fcr_down = fcr_down[:num_steps]
    else:
        # FCR data shorter - pad with zeros (no FCR activation)
        pad_length = num_steps - len(fcr_up)
        print(f"   ⚠️ FCR data shorter by {pad_length} steps - padding with zeros")
        fcr_up = np.concatenate([fcr_up, np.zeros(pad_length)])
        fcr_down = np.concatenate([fcr_down, np.zeros(pad_length)])
    
    # =========================================================================
    # BUILD AND SOLVE MODEL
    # =========================================================================
    
    print("\nBuilding optimization model...")
    model = build_multi_battery_model(
        site_configs=site_configs,
        data=data,
        fcr_signal_up=fcr_up,
        fcr_signal_down=fcr_down,
        delta_t=delta_t,
        SOC0=0.5,
        SOH0=1.0,
        C_peak=C_peak,
        min_fcr_bid=1000.0,  # 1 MW
    )
    
    # Count model components
    n_vars = sum(1 for v in model.component_data_objects(Var))
    n_cons = sum(1 for c in model.component_data_objects(Constraint))
    print(f"Model statistics:")
    print(f"  Variables: {n_vars:,}")
    print(f"  Constraints: {n_cons:,}")
    
    # =========================================================================
    # SOLVER CONFIGURATION (Optimized for Large Problems)
    # =========================================================================
    
    # Performance tuning options (adjust these for your needs)
    SOLVER_CONFIG = {
        'MIPGap': 0.05,          # Accept 2% suboptimality (faster, usually good enough)
        'TimeLimit': 3600,        # 1 hour max for full year (increase if needed)
        'Threads': 0,             # 0 = use all available CPU cores
        'Method': 2,              # 2 = Barrier method (often faster for large LPs)
        'Presolve': 2,            # 2 = Aggressive presolve (reduces problem size)
        'MIPFocus': 1,            # 1 = Focus on finding feasible solutions quickly
        'Cuts': 2,                # 2 = Aggressive cuts (helps prove optimality)
        'NodefileStart': 0.5,     # Start writing nodes to disk after 0.5 GB RAM
        'NodefileDir': '/tmp',    # Directory for node files (saves RAM)
        # 'PoolSearchMode': 2,    # Uncomment to find multiple good solutions
    }
    
    # For very large problems (full year), consider these additional settings:
    # SOLVER_CONFIG['MIPGap'] = 0.05      # Accept 5% gap for faster solve
    # SOLVER_CONFIG['Heuristics'] = 0.2   # Spend 20% of time on heuristics
    # SOLVER_CONFIG['RINS'] = 50          # Apply RINS heuristic every 50 nodes
    
    # Try solvers in order of preference
    SOLVER_OPTIONS = [
        ('gurobi', SOLVER_CONFIG),
        ('cplex', {
            'mip.tolerances.mipgap': SOLVER_CONFIG['MIPGap'],
            'timelimit': SOLVER_CONFIG['TimeLimit'],
            'threads': SOLVER_CONFIG['Threads'],
            'preprocessing.presolve': 1,
            'mip.strategy.heuristicfreq': 10,
        }),
        ('glpk', {'mipgap': SOLVER_CONFIG['MIPGap']}),
        ('cbc', {'ratio': SOLVER_CONFIG['MIPGap'], 'threads': SOLVER_CONFIG['Threads']}),
    ]
    
    solver = None
    for solver_name, options in SOLVER_OPTIONS:
        try:
            test_solver = SolverFactory(solver_name)
            if test_solver.available():
                solver = test_solver
                for opt, val in options.items():
                    solver.options[opt] = val
                print(f"\n✓ Using solver: {solver_name}")
                print(f"  MIPGap: {SOLVER_CONFIG['MIPGap']*100:.0f}%")
                print(f"  TimeLimit: {SOLVER_CONFIG['TimeLimit']}s")
                print(f"  Threads: {'all' if SOLVER_CONFIG['Threads'] == 0 else SOLVER_CONFIG['Threads']}")
                break
        except:
            continue
    
    if solver is None:
        raise RuntimeError("No MILP solver available. Install Gurobi, CPLEX, GLPK, or CBC.")
    
    print("\n🚀 Starting optimization...")
    print(f"   Problem size: {n_vars:,} variables, {n_cons:,} constraints")
    print(f"   This may take a while for large problems. Progress shown below.\n")
    
    import time as time_module
    solve_start = time_module.time()
    results = solver.solve(model, tee=True)
    solve_time = time_module.time() - solve_start
    print(f"\n⏱️  Solve time: {solve_time/60:.1f} minutes")
    
    # =========================================================================
    # EXTRACT RESULTS
    # =========================================================================
    
    print("\n" + "=" * 60)
    print("EXTRACTING RESULTS")
    print("=" * 60)
    
    df = extract_results(model, site_configs, data)
    financials = calculate_financials(df, site_configs, C_peak=C_peak, delta_t=delta_t)
    
    # Print summary
    print("\n--- PORTFOLIO SUMMARY ---")
    port = financials['portfolio']
    print(f"Total Baseline Cost:   €{port['baseline_energy'] + port['baseline_peak']:,.2f}")
    print(f"Total Optimized Cost:  €{port['energy_cost'] + port['peak_cost']:,.2f}")
    print(f"BTM Savings:           €{port['btm_savings']:,.2f}")
    print(f"Peak Savings:          €{port['peak_savings']:,.2f}")
    print(f"FCR Revenue:           €{port['fcr_revenue']:,.2f}")
    print(f"Degradation Cost:      €{port['degradation_cost']:,.2f}")
    print(f"NET BENEFIT:           €{port['net_benefit']:,.2f}")
    
    # Save CSV results
    df.to_csv(this_file / "multi_battery_results.csv", index=False)
    print(f"\nResults saved to multi_battery_results.csv")
    
    # =========================================================================
    # SAVE RESULTS FOR LATER DASHBOARD GENERATION
    # =========================================================================
    
    SAVE_RESULTS = True  # Set to True to save results for later use
    
    if SAVE_RESULTS:
        try:
            from results_io import save_optimization_results
            
            results_package = {
                'df': df,
                'financials': financials,
                'model_status': 'optimal',
            }
            
            save_optimization_results(
                results=results_package,
                data=data,
                site_configs=site_configs,
                metadata={
                    'start_date': START_DATE,
                    'end_date': END_DATE,
                    'C_peak': C_peak,
                    'delta_t': delta_t,
                }
            )
        except Exception as e:
            print(f"⚠️  Could not save results: {e}")
    
    # =========================================================================
    # DASHBOARD (can be disabled if just saving results)
    # =========================================================================
    
    GENERATE_DASHBOARD = True  # Set to False to skip dashboard generation
    
    if GENERATE_DASHBOARD:
        print("\n" + "=" * 60)
        print("GENERATING DASHBOARD")
        print("=" * 60)
        
        try:
            from dashboard_multi_battery import (
                create_multi_battery_dashboard, 
                create_detailed_site_dashboard,
                print_financial_summary,
                generate_comparison_report
            )
            
            # Print detailed financial summary
            print_financial_summary(financials)
            
            # Generate main dashboard
            print("\nCreating portfolio dashboard...")
            create_multi_battery_dashboard(
                df, financials, site_configs, data, 
                output_file=this_file / "multi_battery_dashboard.html"
            )
            
            # Generate comparison report CSV
            print("\nGenerating comparison report...")
            generate_comparison_report(
                df, financials, 
                output_file=this_file / "site_comparison_report.csv"
            )
            
            # Generate detailed dashboard for first site as example
            first_site = site_configs[0].site_id
            print(f"\nCreating detailed dashboard for {first_site}...")
            create_detailed_site_dashboard(
                df, first_site, financials, site_configs[0],
                output_file=this_file / f"dashboard_{first_site}.html"
            )
            
            print("\n✓ All dashboards generated successfully!")
            print(f"  - multi_battery_dashboard.html (interactive portfolio view)")
            print(f"  - site_comparison_report.csv (site-by-site metrics)")
            print(f"  - dashboard_{first_site}.html (detailed site view)")
            
        except ImportError as e:
            print(f"\n⚠️  Dashboard generation skipped: {e}")
            print("   Run: python generate_dashboard.py to create dashboards from saved results")
    else:
        print("\n⚠️  Dashboard generation skipped (GENERATE_DASHBOARD = False)")
        print("   Run: python generate_dashboard.py to create dashboards from saved results")
    
    print("\n" + "=" * 60)
    print("OPTIMIZATION COMPLETE")
    print("=" * 60)

# %%

