"""
Results I/O Module
==================

Handles saving and loading of optimization results to enable separation of:
1. Model execution (slow, expensive)
2. Dashboard generation (fast, iterative)

Supported formats:
- Pickle (.pkl): Full Python objects, fast, includes all metadata
- Parquet (.parquet): Efficient for DataFrames, cross-platform compatible
"""

import pickle
import os
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, Tuple
import pandas as pd
import numpy as np

from .paths import RESULTS_DIR


def ensure_results_dir(results_dir: Optional[Path] = None) -> Path:
    """Ensure results directory exists and return it."""
    dir_path = Path(results_dir) if results_dir else RESULTS_DIR
    dir_path.mkdir(parents=True, exist_ok=True)
    return dir_path


def generate_filename(prefix: str, start_date: str, end_date: str, suffix: str = "") -> str:
    """Generate a standardized filename for results."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if suffix:
        return f"{prefix}_{start_date}_to_{end_date}_{suffix}_{timestamp}"
    return f"{prefix}_{start_date}_to_{end_date}_{timestamp}"


# =============================================================================
# SINGLE OPTIMIZATION RESULTS (model_multi_battery.py)
# =============================================================================

def save_optimization_results(
    results: Dict[str, Any],
    data: Dict[str, Any],
    site_configs: list,
    filename: Optional[str] = None,
    results_dir: Optional[Path] = None,
    metadata: Optional[Dict] = None
) -> str:
    """
    Save optimization results from model_multi_battery.py.
    
    Parameters
    ----------
    results : dict
        The results dictionary from run_multi_battery_vpp()
    data : dict
        The data dictionary used in optimization
    site_configs : list
        List of SiteConfig objects
    filename : str, optional
        Custom filename (without extension)
    results_dir : Path, optional
        Directory to save to (default: ./results/)
    metadata : dict, optional
        Additional metadata to save
        
    Returns
    -------
    str
        Full path to saved file
    """
    dir_path = ensure_results_dir(results_dir)
    
    # Generate filename if not provided
    if filename is None:
        time_index = data.get('time_index', [])
        if len(time_index) > 0:
            start = pd.Timestamp(time_index[0]).strftime("%Y-%m-%d")
            end = pd.Timestamp(time_index[-1]).strftime("%Y-%m-%d")
        else:
            start = end = "unknown"
        filename = generate_filename("optimization", start, end)
    
    # Convert site_configs to serializable format
    site_configs_data = []
    for cfg in site_configs:
        cfg_dict = {
            'site_id': cfg.site_id,
            'load_column': cfg.load_column,
            'battery': {
                'name': cfg.battery.name,
                'E_max': cfg.battery.E_max,
                'P_max': cfg.battery.P_max,
                'eta_ch': cfg.battery.eta_ch,
                'eta_dis': cfg.battery.eta_dis,
                'I0': getattr(cfg.battery, 'I0', 73000.0),
                'V_bat': getattr(cfg.battery, 'V_bat', 777.0),
            },
            'btm_ratio': cfg.btm_ratio,
            'P_buy_max': cfg.P_buy_max,
            'P_sell_max': cfg.P_sell_max,
        }
        site_configs_data.append(cfg_dict)
    
    # Prepare save package
    save_data = {
        'version': '1.0',
        'saved_at': datetime.now().isoformat(),
        'type': 'single_optimization',
        
        # Core results
        'results': results,
        
        # Data used (converted to serializable)
        'data': {
            'time_index': [str(t) for t in data.get('time_index', [])],
            'num_steps': data.get('num_steps'),
            'num_sites': data.get('num_sites'),
            'site_names': data.get('site_names', []),
            'P_demand': {k: v.tolist() if hasattr(v, 'tolist') else v 
                        for k, v in data.get('P_demand', {}).items()},
            'P_pv': {k: v.tolist() if hasattr(v, 'tolist') else v 
                    for k, v in data.get('P_pv', {}).items()},
            'spot_price': data.get('spot_price').tolist() if hasattr(data.get('spot_price', []), 'tolist') else data.get('spot_price', []),
            'fcr_price': data.get('fcr_price').tolist() if hasattr(data.get('fcr_price', []), 'tolist') else data.get('fcr_price', []),
        },
        
        # Config
        'site_configs': site_configs_data,
        
        # Optional metadata
        'metadata': metadata or {},
    }
    
    # Save as pickle
    filepath = dir_path / f"{filename}.pkl"
    with open(filepath, 'wb') as f:
        pickle.dump(save_data, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    print(f"\n✅ Results saved to: {filepath}")
    print(f"   File size: {filepath.stat().st_size / 1024 / 1024:.2f} MB")
    
    return str(filepath)


def load_optimization_results(filepath: str) -> Tuple[Dict, Dict, list, Dict]:
    """
    Load optimization results.
    
    Parameters
    ----------
    filepath : str
        Path to the saved results file (.pkl)
        
    Returns
    -------
    tuple
        (results, data, site_configs_data, metadata)
    """
    from model_multi_battery import BatterySpec, SiteConfig
    
    with open(filepath, 'rb') as f:
        save_data = pickle.load(f)
    
    print(f"\n📂 Loaded results from: {filepath}")
    print(f"   Saved at: {save_data.get('saved_at', 'unknown')}")
    print(f"   Version: {save_data.get('version', 'unknown')}")
    
    # Reconstruct data with numpy arrays
    data = save_data['data'].copy()
    data['time_index'] = pd.to_datetime(data['time_index'])
    data['P_demand'] = {k: np.array(v) for k, v in data['P_demand'].items()}
    data['P_pv'] = {k: np.array(v) for k, v in data['P_pv'].items()}
    data['spot_price'] = np.array(data['spot_price'])
    data['fcr_price'] = np.array(data['fcr_price'])
    
    # Reconstruct site_configs
    site_configs = []
    for cfg_data in save_data['site_configs']:
        battery = BatterySpec(
            name=cfg_data['battery']['name'],
            E_max=cfg_data['battery']['E_max'],
            P_max=cfg_data['battery']['P_max'],
            eta_ch=cfg_data['battery']['eta_ch'],
            eta_dis=cfg_data['battery']['eta_dis'],
            I0=cfg_data['battery'].get('I0', 73000.0),
            V_bat=cfg_data['battery'].get('V_bat', 777.0),
        )
        site_config = SiteConfig(
            site_id=cfg_data['site_id'],
            load_column=cfg_data['load_column'],
            battery=battery,
            btm_ratio=cfg_data['btm_ratio'],
            P_buy_max=cfg_data['P_buy_max'],
            P_sell_max=cfg_data['P_sell_max'],
        )
        site_configs.append(site_config)
    
    return save_data['results'], data, site_configs, save_data.get('metadata', {})


# =============================================================================
# SCENARIO ANALYSIS RESULTS (scenario_analysis.py)
# =============================================================================

def save_scenario_results(
    results_df: pd.DataFrame,
    all_results: list,
    base_data: Dict[str, Any],
    filename: Optional[str] = None,
    results_dir: Optional[Path] = None,
    metadata: Optional[Dict] = None
) -> str:
    """
    Save scenario analysis results.
    
    Parameters
    ----------
    results_df : pd.DataFrame
        Summary DataFrame with scenario metrics
    all_results : list
        List of (scenario_name, results_dict, site_configs) tuples
    base_data : dict
        The base data dictionary
    filename : str, optional
        Custom filename (without extension)
    results_dir : Path, optional
        Directory to save to
    metadata : dict, optional
        Additional metadata
        
    Returns
    -------
    str
        Full path to saved file
    """
    dir_path = ensure_results_dir(results_dir)
    
    # Generate filename if not provided
    if filename is None:
        time_index = base_data.get('time_index', [])
        if len(time_index) > 0:
            start = pd.Timestamp(time_index[0]).strftime("%Y-%m-%d")
            end = pd.Timestamp(time_index[-1]).strftime("%Y-%m-%d")
        else:
            start = end = "unknown"
        filename = generate_filename("scenarios", start, end)
    
    # Prepare save package
    save_data = {
        'version': '1.0',
        'saved_at': datetime.now().isoformat(),
        'type': 'scenario_analysis',
        
        # Summary DataFrame
        'results_df': results_df,
        
        # Full results for each scenario
        'all_results': all_results,
        
        # Base data info
        'base_data_info': {
            'time_index': [str(t) for t in base_data.get('time_index', [])],
            'num_steps': base_data.get('num_steps'),
            'num_sites': base_data.get('num_sites'),
            'site_names': base_data.get('site_names', []),
        },
        
        # Metadata
        'metadata': metadata or {},
    }
    
    # Save as pickle
    filepath = dir_path / f"{filename}.pkl"
    with open(filepath, 'wb') as f:
        pickle.dump(save_data, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    # Also save summary as CSV for easy viewing
    csv_path = dir_path / f"{filename}_summary.csv"
    results_df.to_csv(csv_path, index=False)
    
    print(f"\n✅ Scenario results saved to: {filepath}")
    print(f"   Summary CSV: {csv_path}")
    print(f"   File size: {filepath.stat().st_size / 1024 / 1024:.2f} MB")
    
    return str(filepath)


def load_scenario_results(filepath: str) -> Tuple[pd.DataFrame, list, Dict, Dict]:
    """
    Load scenario analysis results.
    
    Parameters
    ----------
    filepath : str
        Path to saved results file (.pkl)
        
    Returns
    -------
    tuple
        (results_df, all_results, base_data_info, metadata)
    """
    with open(filepath, 'rb') as f:
        save_data = pickle.load(f)
    
    print(f"\n📂 Loaded scenario results from: {filepath}")
    print(f"   Saved at: {save_data.get('saved_at', 'unknown')}")
    print(f"   Scenarios: {len(save_data.get('all_results', []))}")
    
    return (
        save_data['results_df'],
        save_data['all_results'],
        save_data['base_data_info'],
        save_data.get('metadata', {})
    )


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def list_saved_results(results_dir: Optional[Path] = None) -> pd.DataFrame:
    """
    List all saved results in the results directory.
    
    Returns a DataFrame with file info.
    """
    dir_path = ensure_results_dir(results_dir)
    
    files = []
    for f in dir_path.glob("*.pkl"):
        # Try to read metadata
        try:
            with open(f, 'rb') as fp:
                data = pickle.load(fp)
            result_type = data.get('type', 'unknown')
            saved_at = data.get('saved_at', 'unknown')
        except:
            result_type = 'unknown'
            saved_at = 'unknown'
        
        files.append({
            'filename': f.name,
            'type': result_type,
            'saved_at': saved_at,
            'size_mb': f.stat().st_size / 1024 / 1024,
            'path': str(f),
        })
    
    df = pd.DataFrame(files)
    if len(df) > 0:
        df = df.sort_values('saved_at', ascending=False)
    
    return df


def get_latest_results(result_type: str = 'single_optimization', 
                       results_dir: Optional[Path] = None) -> Optional[str]:
    """
    Get the path to the most recent results file of a given type.
    
    Parameters
    ----------
    result_type : str
        'single_optimization' or 'scenario_analysis'
    results_dir : Path, optional
        Directory to search
        
    Returns
    -------
    str or None
        Path to most recent file, or None if not found
    """
    df = list_saved_results(results_dir)
    if len(df) == 0:
        return None
    
    filtered = df[df['type'] == result_type]
    if len(filtered) == 0:
        return None
    
    return filtered.iloc[0]['path']


if __name__ == "__main__":
    # Show available results
    print("\n" + "="*60)
    print("SAVED RESULTS")
    print("="*60)
    
    df = list_saved_results()
    if len(df) == 0:
        print("\nNo saved results found in ./results/")
        print("Run model_multi_battery.py or scenario_analysis.py first.")
    else:
        print(df.to_string(index=False))

