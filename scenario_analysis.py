"""
Scenario Analysis for Multi-Battery VPP
=========================================
Runs optimization across combinations of:
- btm_ratio: [0, 0.2, 0.4, 0.6, 0.8, 1.0]
- scaler_input: [0.2, 0.5, 1.0, 1.5, 5.0]

Generates comprehensive comparison visualizations.
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Optional
import time
import warnings
warnings.filterwarnings('ignore')

from pyomo.environ import SolverFactory, value, Var, Constraint

from model_multi_battery import (
    SiteConfig, BatterySpec, 
    load_multi_site_data, generate_fcr_activation_profile, load_fcr_activation_profile,
    filter_data_by_date_range,
    build_multi_battery_model, extract_results, calculate_financials
)

from lib.paths import DATA_DIR, OUTPUTS_DIR

# =============================================================================
# SCENARIO CONFIGURATION
# =============================================================================

@dataclass
class ScenarioResult:
    """Results from a single scenario run."""
    btm_ratio: float
    scaler_input: float
    scenario_id: str
    
    # Feasibility
    feasible: bool
    solver_status: str
    solve_time: float
    
    # Financial metrics
    net_benefit: float
    btm_savings: float
    peak_savings: float
    fcr_revenue: float
    export_revenue: float
    degradation_cost: float
    baseline_cost: float
    optimized_cost: float
    
    # Operational metrics
    total_demand_kwh: float
    total_import_kwh: float
    total_export_kwh: float
    self_consumption_rate: float  # % of demand met by battery
    grid_dependency: float  # % of demand from grid
    
    avg_soc_btm: float
    avg_soc_ftm: float
    avg_fcr_bid: float
    fcr_participation_rate: float  # % of time with FCR bid
    
    peak_demand: float
    peak_import: float
    peak_reduction_pct: float
    
    # Attribution (% of net benefit)
    pct_from_btm: float
    pct_from_peak: float
    pct_from_fcr: float
    pct_from_export: float
    
    # Battery utilization
    total_charge_kwh: float
    total_discharge_kwh: float
    equivalent_cycles: float
    avg_power_utilization: float  # % of max power used on average


def run_single_scenario(
    btm_ratio: float,
    scaler_input: float,
    site_configs_template: List[SiteConfig],
    base_data: Dict,
    fcr_signal_up: np.ndarray,
    fcr_signal_down: np.ndarray,
    delta_t: float = 0.25,
    C_peak: float = 192.66,
    start_date: str = None,
    end_date: str = None,
    verbose: bool = False
) -> ScenarioResult:
    """
    Run a single scenario with given btm_ratio and scaler_input.
    
    Args:
        btm_ratio: Fraction of battery for BTM (0.0 to 1.0)
        scaler_input: Load scaling factor
        site_configs_template: Template site configurations
        base_data: Pre-loaded data dictionary
        fcr_signal_up: FCR upward activation profile
        fcr_signal_down: FCR downward activation profile
        delta_t: Timestep in hours (default 0.25 = 15 min)
        C_peak: Peak tariff (already prorated for simulation period)
        start_date: Start date for simulation (e.g., "2024-03-01")
        end_date: End date for simulation (e.g., "2024-03-31")
        verbose: Print detailed output
    """
    
    scenario_id = f"btm{btm_ratio:.1f}_scale{scaler_input:.1f}"
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"Running Scenario: {scenario_id}")
        print(f"  BTM Ratio: {btm_ratio:.1%}")
        print(f"  Load Scaler: {scaler_input:.1f}x")
        print(f"{'='*60}")
    
    start_time = time.time()
    
    # Create site configs with updated btm_ratio
    # Scale P_buy_max and P_sell_max based on scaler_input to ensure feasibility
    # Battery P_max is 108 kW, so with scaler=5 loads can be ~540 kW
    adjusted_P_max = max(500, 200 * scaler_input * 3)  # Ensure grid can handle scaled loads
    
    site_configs = []
    for template in site_configs_template:
        site_configs.append(SiteConfig(
            site_id=template.site_id,
            load_column=template.load_column,
            battery=template.battery,
            btm_ratio=btm_ratio,
            P_buy_max=adjusted_P_max,  # Scaled to handle larger loads
            P_sell_max=adjusted_P_max   # Scaled to handle larger exports
        ))
    
    # Load data with scaler and date range
    data = load_multi_site_data(
        data_dir=DATA_DIR,
        site_configs=site_configs,
        scale_loads_to_battery=True,
        scaler_input=scaler_input,
        start_date=start_date,
        end_date=end_date
    )
    
    # FCR signals should match the data length
    # Handle length mismatch: pad with zeros or truncate as needed
    num_steps = data['num_steps']
    
    if len(fcr_signal_up) >= num_steps:
        fcr_signal_up_truncated = fcr_signal_up[:num_steps]
        fcr_signal_down_truncated = fcr_signal_down[:num_steps]
    else:
        # FCR data shorter than needed - pad with zeros (no FCR activation)
        pad_length = num_steps - len(fcr_signal_up)
        if verbose:
            print(f"  ⚠️ FCR data shorter by {pad_length} steps - padding with zeros")
        fcr_signal_up_truncated = np.concatenate([fcr_signal_up, np.zeros(pad_length)])
        fcr_signal_down_truncated = np.concatenate([fcr_signal_down, np.zeros(pad_length)])
    
    # Calculate total FTM capacity
    total_ftm_capacity = sum(cfg.battery.P_max * (1 - cfg.btm_ratio) for cfg in site_configs)
    
    # Determine FCR eligibility
    # If FTM < 1 MW, we CANNOT participate in FCR market (hard requirement)
    fcr_enabled = total_ftm_capacity >= 1000
    
    if not fcr_enabled:
        min_fcr_bid = 0  
        if verbose:
            print(f"  ⚠️ FTM capacity ({total_ftm_capacity:.0f} kW) < 1 MW → FCR DISABLED")
    else:
        min_fcr_bid = 1000
        if verbose:
            print(f"  ✓ FTM capacity ({total_ftm_capacity:.0f} kW) ≥ 1 MW → FCR enabled")
    
    try:
        # If FCR is disabled, zero out FCR prices so model doesn't bid
        if not fcr_enabled:
            # Create zero FCR prices - model will have no incentive to bid
            fcr_prices_adjusted = np.zeros(len(data['fcr_prices']))
        else:
            fcr_prices_adjusted = data['fcr_prices']
        
        # Adjust data with FCR prices
        data_adjusted = data.copy()
        data_adjusted['fcr_prices'] = fcr_prices_adjusted
        
        # Build model
        model = build_multi_battery_model(
            site_configs=site_configs,
            data=data_adjusted,
            fcr_signal_up=fcr_signal_up_truncated,
            fcr_signal_down=fcr_signal_down_truncated,
            delta_t=delta_t,
            SOC0=0.5,
            SOH0=1.0,
            C_peak=C_peak,
            min_fcr_bid=min_fcr_bid,
        )
        
        # Solve - configure based on problem size
        solver = SolverFactory('gurobi')
        
        # Calculate appropriate time limit based on problem size
        num_timesteps = data['num_steps']
        num_sites = len(site_configs)
        problem_size = num_timesteps * num_sites
        
        # Time limit: 5 min for small, up to 2 hours for full year
        if problem_size < 10000:
            time_limit = 300       # 5 min for ~1 week
        elif problem_size < 100000:
            time_limit = 1800      # 30 min for ~1 month
        else:
            time_limit = 7200      # 2 hours for full year
        
        solver.options['MIPGap'] = 0.05           # Accept 5% gap for faster solving
        solver.options['TimeLimit'] = time_limit
        solver.options['Threads'] = 0             # Use all CPU cores
        solver.options['Method'] = 2              # Barrier method
        solver.options['Presolve'] = 2            # Aggressive presolve
        solver.options['MIPFocus'] = 1            # Focus on feasibility
        solver.options['OutputFlag'] = 1          # Show output for debugging
        
        if verbose:
            print(f"    Problem: {num_timesteps:,} timesteps × {num_sites} sites")
            print(f"    TimeLimit: {time_limit//60} min, MIPGap: 5%")
        
        results = solver.solve(model, tee=verbose)
        
        solve_time = time.time() - start_time
        
        # Check if feasible
        from pyomo.opt import TerminationCondition
        term_cond = results.solver.termination_condition
        if term_cond not in [TerminationCondition.optimal, TerminationCondition.feasible]:
            # Provide helpful error message
            if term_cond == TerminationCondition.maxTimeLimit:
                print(f"    ⚠️  TIMEOUT after {solve_time:.0f}s - try shorter date range or increase TimeLimit")
            elif term_cond == TerminationCondition.infeasible:
                print(f"    ❌ INFEASIBLE - model constraints cannot be satisfied")
            else:
                print(f"    ❌ FAILED: {term_cond}")
            
            infeasible_result = ScenarioResult(
                btm_ratio=btm_ratio, scaler_input=scaler_input, scenario_id=scenario_id,
                feasible=False, solver_status=str(term_cond), solve_time=solve_time,
                net_benefit=0, btm_savings=0, peak_savings=0, fcr_revenue=0, export_revenue=0,
                degradation_cost=0, baseline_cost=0, optimized_cost=0,
                total_demand_kwh=0, total_import_kwh=0, total_export_kwh=0,
                self_consumption_rate=0, grid_dependency=0,
                avg_soc_btm=0, avg_soc_ftm=0, avg_fcr_bid=0, fcr_participation_rate=0,
                peak_demand=0, peak_import=0, peak_reduction_pct=0,
                pct_from_btm=0, pct_from_peak=0, pct_from_fcr=0, pct_from_export=0,
                total_charge_kwh=0, total_discharge_kwh=0, equivalent_cycles=0, avg_power_utilization=0
            )
            return infeasible_result, None, None, None, None
        
        # Extract results (use adjusted data with correct FCR prices)
        df = extract_results(model, site_configs, data_adjusted)
        financials = calculate_financials(df, site_configs, C_peak, delta_t)
        
        # Override FCR revenue to 0 if FCR was disabled
        if not fcr_enabled:
            for site_id in financials['sites']:
                financials['sites'][site_id]['fcr_revenue'] = 0
            financials['portfolio']['fcr_revenue'] = 0
        port = financials['portfolio']
        
        # Aggregate metrics
        agg = df.groupby('t').agg({
            'demand': 'sum',
            'P_buy': 'sum',
            'P_sell': 'sum',
            'P_ch_BTM': 'sum',
            'P_dis_BTM': 'sum',
            'P_ch_FTM': 'sum',
            'P_dis_FTM': 'sum',
            'P_FCR_bid': 'sum',
            'SOC_BTM': 'sum',
            'SOC_FTM': 'sum',
        })
        
        total_E_max = sum(cfg.battery.E_max for cfg in site_configs)
        total_E_btm = sum(cfg.battery.E_max * cfg.btm_ratio for cfg in site_configs)
        total_E_ftm = sum(cfg.battery.E_max * (1 - cfg.btm_ratio) for cfg in site_configs)
        total_P_max = sum(cfg.battery.P_max for cfg in site_configs)
        
        # Energy totals (kWh)
        total_demand_kwh = agg['demand'].sum() * delta_t
        total_import_kwh = agg['P_buy'].sum() * delta_t
        total_export_kwh = agg['P_sell'].sum() * delta_t
        total_charge_kwh = (agg['P_ch_BTM'].sum() + agg['P_ch_FTM'].sum()) * delta_t
        total_discharge_kwh = (agg['P_dis_BTM'].sum() + agg['P_dis_FTM'].sum()) * delta_t
        
        # Self-consumption = demand met by battery discharge
        battery_to_load = min(total_discharge_kwh, total_demand_kwh - total_export_kwh)
        self_consumption_rate = battery_to_load / max(total_demand_kwh, 1) * 100
        grid_dependency = total_import_kwh / max(total_demand_kwh, 1) * 100
        
        # SOC averages (handle division by zero when btm_ratio=0 or 1)
        avg_soc_btm = (agg['SOC_BTM'].mean() / total_E_btm * 100) if total_E_btm > 10 else 50  # Default 50% if no BTM
        avg_soc_ftm = (agg['SOC_FTM'].mean() / total_E_ftm * 100) if total_E_ftm > 10 else 50  # Default 50% if no FTM
        
        # FCR metrics
        avg_fcr_bid = agg['P_FCR_bid'].mean()
        fcr_participation_rate = (agg['P_FCR_bid'] > 0).sum() / len(agg) * 100
        
        # Peak metrics
        peak_demand = agg['demand'].max()
        peak_import = agg['P_buy'].max()
        peak_reduction_pct = (1 - peak_import / max(peak_demand, 1)) * 100
        
        # Export revenue (separate calculation)
        export_revenue = (agg['P_sell'] * data['day_ahead'][:len(agg)] * delta_t).sum()
        
        # Attribution (normalize to 100%)
        # BTM savings, Peak savings, FCR revenue are the main value sources
        total_value = abs(port['btm_savings']) + abs(port['peak_savings']) + abs(port['fcr_revenue'])
        if total_value > 0:
            pct_from_btm = abs(port['btm_savings']) / total_value * 100
            pct_from_peak = abs(port['peak_savings']) / total_value * 100
            pct_from_fcr = abs(port['fcr_revenue']) / total_value * 100
        else:
            pct_from_btm = pct_from_peak = pct_from_fcr = 0
        
        pct_from_export = export_revenue / max(total_value, 1) * 100 if total_value > 0 else 0
        
        # Battery utilization
        equivalent_cycles = total_discharge_kwh / max(total_E_max, 1)
        avg_power = (agg['P_ch_BTM'] + agg['P_ch_FTM'] + agg['P_dis_BTM'] + agg['P_dis_FTM']).mean()
        avg_power_utilization = avg_power / max(total_P_max, 1) * 100
        
        if verbose:
            print(f"  ✓ Solved in {solve_time:.1f}s")
            print(f"  Net Benefit: €{port['net_benefit']:,.0f}")
        
        # Create result object
        scenario_result = ScenarioResult(
            btm_ratio=btm_ratio,
            scaler_input=scaler_input,
            scenario_id=scenario_id,
            feasible=True,
            solver_status='optimal',
            solve_time=solve_time,
            net_benefit=port['net_benefit'],
            btm_savings=port['btm_savings'],
            peak_savings=port['peak_savings'],
            fcr_revenue=port['fcr_revenue'],
            export_revenue=export_revenue,
            degradation_cost=port['degradation_cost'],
            baseline_cost=port['baseline_energy'] + port['baseline_peak'],
            optimized_cost=port['energy_cost'] + port['peak_cost'],
            total_demand_kwh=total_demand_kwh,
            total_import_kwh=total_import_kwh,
            total_export_kwh=total_export_kwh,
            self_consumption_rate=self_consumption_rate,
            grid_dependency=grid_dependency,
            avg_soc_btm=avg_soc_btm,
            avg_soc_ftm=avg_soc_ftm,
            avg_fcr_bid=avg_fcr_bid,
            fcr_participation_rate=fcr_participation_rate,
            peak_demand=peak_demand,
            peak_import=peak_import,
            peak_reduction_pct=peak_reduction_pct,
            pct_from_btm=pct_from_btm,
            pct_from_peak=pct_from_peak,
            pct_from_fcr=pct_from_fcr,
            pct_from_export=pct_from_export,
            total_charge_kwh=total_charge_kwh,
            total_discharge_kwh=total_discharge_kwh,
            equivalent_cycles=equivalent_cycles,
            avg_power_utilization=avg_power_utilization
        )
        
        # Return result with detailed data for dashboard generation
        return scenario_result, df, financials, site_configs, data
        
    except Exception as e:
        if verbose:
            print(f"  ✗ Error: {e}")
        failed_result = ScenarioResult(
            btm_ratio=btm_ratio, scaler_input=scaler_input, scenario_id=scenario_id,
            feasible=False, solver_status=str(e), solve_time=time.time() - start_time,
            net_benefit=0, btm_savings=0, peak_savings=0, fcr_revenue=0, export_revenue=0,
            degradation_cost=0, baseline_cost=0, optimized_cost=0,
            total_demand_kwh=0, total_import_kwh=0, total_export_kwh=0,
            self_consumption_rate=0, grid_dependency=0,
            avg_soc_btm=0, avg_soc_ftm=0, avg_fcr_bid=0, fcr_participation_rate=0,
            peak_demand=0, peak_import=0, peak_reduction_pct=0,
            pct_from_btm=0, pct_from_peak=0, pct_from_fcr=0, pct_from_export=0,
            total_charge_kwh=0, total_discharge_kwh=0, equivalent_cycles=0, avg_power_utilization=0
        )
        return failed_result, None, None, None, None


def run_all_scenarios(
    btm_ratios: List[float],
    scaler_inputs: List[float],
    site_configs_template: List[SiteConfig],
    start_date: str = None,
    end_date: str = None,
    C_peak_annual: float = 192.66,
    verbose: bool = True,
    output_dir: Path = None,
    generate_individual_dashboards: bool = True
) -> Tuple[pd.DataFrame, Dict]:
    """
    Run all scenario combinations, generate dashboards, and return results.
    
    Args:
        btm_ratios: List of BTM ratios to test
        scaler_inputs: List of load scalers to test
        site_configs_template: Template site configurations
        start_date: Start date for simulation (e.g., "2024-03-01")
        end_date: End date for simulation (e.g., "2024-03-31")
        C_peak_annual: Annual peak tariff in EUR/kW (will be prorated)
        verbose: Print progress
        output_dir: Directory to save outputs (created if not exists)
        generate_individual_dashboards: Generate per-scenario dashboards
    
    Returns:
        Tuple of (results_df, scenario_data_dict)
    
    Examples:
        # Run scenarios for one week
        results_df, data = run_all_scenarios(
            btm_ratios=[0.2, 0.4, 0.6],
            scaler_inputs=[0.5, 1.0, 2.0],
            site_configs_template=configs,
            start_date="2024-06-01",
            end_date="2024-06-07"
        )
        
        # Run scenarios for full month
        results_df, data = run_all_scenarios(
            btm_ratios=[0.4],
            scaler_inputs=[1.0],
            site_configs_template=configs,
            start_date="2024-07-01",
            end_date="2024-07-31"
        )
    """
    
    # Import dashboard functions
    from lib.dashboard_multi_battery import create_multi_battery_dashboard
    
    # Pre-load base data to get date range info
    print("\nPre-loading base data...")
    base_configs = [SiteConfig(
        site_id=cfg.site_id,
        load_column=cfg.load_column,
        battery=cfg.battery,
        btm_ratio=0.4,
        P_buy_max=cfg.P_buy_max,
        P_sell_max=cfg.P_sell_max
    ) for cfg in site_configs_template]
    
    base_data = load_multi_site_data(
        data_dir=DATA_DIR,
        site_configs=base_configs,
        scale_loads_to_battery=True,
        scaler_input=1.0,
        start_date=start_date,
        end_date=end_date
    )
    
    # Calculate prorated peak tariff
    simulation_days = base_data['num_steps'] / 96
    C_peak = C_peak_annual * (simulation_days / 365.0)
    
    print("\n" + "="*70)
    print("SCENARIO ANALYSIS")
    print("="*70)
    print(f"BTM Ratios: {btm_ratios}")
    print(f"Scaler Inputs: {scaler_inputs}")
    print(f"Total Scenarios: {len(btm_ratios) * len(scaler_inputs)}")
    print(f"Date Range: {base_data['time_index'][0]} to {base_data['time_index'][-1]}")
    print(f"Duration: {simulation_days:.1f} days ({base_data['num_steps']} timesteps)")
    print(f"Peak Tariff: €{C_peak:.2f} (prorated from €{C_peak_annual:.2f}/year)")
    print("="*70)
    
    # Create output directory
    if output_dir is None:
        output_dir = OUTPUTS_DIR / "scenario_outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load FCR activation profiles for the date range
    delta_t = 0.25
    fcr_signal_up, fcr_signal_down = load_fcr_activation_profile(
        data_dir=DATA_DIR,
        start_date=start_date,
        end_date=end_date
    )
    
    # Run all scenarios
    results = []
    scenario_data = {}  # Store data for each scenario
    total_scenarios = len(btm_ratios) * len(scaler_inputs)
    current = 0
    
    for btm_ratio in btm_ratios:
        for scaler in scaler_inputs:
            current += 1
            print(f"\n[{current}/{total_scenarios}] btm_ratio={btm_ratio:.1f}, scaler={scaler:.1f}")
            
            result, df_detail, financials, site_configs, data = run_single_scenario(
                btm_ratio=btm_ratio,
                scaler_input=scaler,
                site_configs_template=site_configs_template,
                base_data=base_data,
                fcr_signal_up=fcr_signal_up,
                fcr_signal_down=fcr_signal_down,
                delta_t=delta_t,
                C_peak=C_peak,
                start_date=start_date,
                end_date=end_date,
                verbose=verbose
            )
            
            results.append(asdict(result))
            
            if result.feasible:
                print(f"    Net Benefit: €{result.net_benefit:,.0f} | "
                      f"Peak↓: {result.peak_reduction_pct:.0f}% | "
                      f"FCR: €{result.fcr_revenue:,.0f}")
                
                # Store scenario data
                scenario_data[result.scenario_id] = {
                    'result': result,
                    'df': df_detail,
                    'financials': financials,
                    'site_configs': site_configs,
                    'data': data
                }
                
                # Generate individual dashboard
                if generate_individual_dashboards and df_detail is not None:
                    dashboard_file = output_dir / f"dashboard_{result.scenario_id}.html"
                    try:
                        create_multi_battery_dashboard(
                            df_detail, financials, site_configs, data,
                            output_file=dashboard_file,
                            delta_t=delta_t
                        )
                    except Exception as e:
                        print(f"    ⚠️ Dashboard generation failed: {e}")
    
    results_df = pd.DataFrame(results)
    
    print("\n" + "="*70)
    print("SCENARIO ANALYSIS COMPLETE")
    print(f"Feasible: {results_df['feasible'].sum()} / {len(results_df)}")
    print("="*70)
    
    return results_df, scenario_data


# =============================================================================
# MASTER NAVIGATION PAGE
# =============================================================================

def create_master_navigation(
    results_df: pd.DataFrame,
    output_dir: Path,
    output_file: Path = None
) -> str:
    """
    Create a master HTML navigation page linking to all dashboards.
    """
    
    df = results_df[results_df['feasible']].copy()
    
    # Sort by net benefit descending
    df = df.sort_values('net_benefit', ascending=False)
    
    # Build HTML
    html = """<!DOCTYPE html>
<html>
<head>
    <title>VPP Scenario Analysis - Master Navigation</title>
    <style>
        :root {
            --primary: #2c3e50;
            --secondary: #3498db;
            --success: #27ae60;
            --warning: #f39c12;
            --danger: #e74c3c;
            --light: #ecf0f1;
            --dark: #1a252f;
        }
        
        * { box-sizing: border-box; margin: 0; padding: 0; }
        
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, var(--dark) 0%, var(--primary) 100%);
            min-height: 100vh;
            color: white;
            padding: 20px;
        }
        
        .container { max-width: 1400px; margin: 0 auto; }
        
        h1 {
            text-align: center;
            font-size: 2.5em;
            margin-bottom: 10px;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        }
        
        .subtitle {
            text-align: center;
            color: var(--light);
            margin-bottom: 30px;
            font-size: 1.1em;
        }
        
        .nav-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }
        
        .nav-card {
            background: rgba(255,255,255,0.1);
            border-radius: 12px;
            padding: 20px;
            transition: all 0.3s ease;
            border: 1px solid rgba(255,255,255,0.1);
        }
        
        .nav-card:hover {
            transform: translateY(-5px);
            background: rgba(255,255,255,0.15);
            box-shadow: 0 10px 30px rgba(0,0,0,0.3);
        }
        
        .nav-card h3 {
            color: var(--secondary);
            margin-bottom: 10px;
            font-size: 1.3em;
        }
        
        .nav-card a {
            display: inline-block;
            color: white;
            text-decoration: none;
            background: var(--secondary);
            padding: 8px 16px;
            border-radius: 6px;
            margin-top: 10px;
            transition: background 0.3s;
        }
        
        .nav-card a:hover { background: #2980b9; }
        
        .metric {
            display: flex;
            justify-content: space-between;
            padding: 5px 0;
            border-bottom: 1px solid rgba(255,255,255,0.1);
        }
        
        .metric-label { color: var(--light); }
        .metric-value { font-weight: bold; }
        .metric-value.positive { color: var(--success); }
        .metric-value.negative { color: var(--danger); }
        
        .main-links {
            display: flex;
            justify-content: center;
            gap: 20px;
            margin-bottom: 30px;
            flex-wrap: wrap;
        }
        
        .main-link {
            background: var(--secondary);
            color: white;
            padding: 15px 30px;
            border-radius: 8px;
            text-decoration: none;
            font-weight: bold;
            transition: all 0.3s;
        }
        
        .main-link:hover {
            background: #2980b9;
            transform: scale(1.05);
        }
        
        .main-link.highlight {
            background: var(--success);
        }
        
        .section-title {
            font-size: 1.5em;
            margin: 30px 0 20px 0;
            padding-bottom: 10px;
            border-bottom: 2px solid var(--secondary);
        }
        
        .rank-badge {
            display: inline-block;
            background: var(--warning);
            color: var(--dark);
            padding: 2px 8px;
            border-radius: 4px;
            font-size: 0.8em;
            font-weight: bold;
            margin-right: 10px;
        }
        
        .rank-badge.gold { background: #f1c40f; }
        .rank-badge.silver { background: #bdc3c7; }
        .rank-badge.bronze { background: #e67e22; }
    </style>
</head>
<body>
    <div class="container">
        <h1>🔋 VPP Scenario Analysis</h1>
        <p class="subtitle">Multi-Battery Virtual Power Plant Optimization Results</p>
        
        <div class="main-links">
            <a href="scenario_comparison_dashboard.html" class="main-link highlight">📊 Comparison Dashboard</a>
            <a href="scenario_insights.html" class="main-link">📈 Detailed Insights</a>
            <a href="scenario_results.csv" class="main-link">📁 Download CSV</a>
        </div>
        
        <h2 class="section-title">Individual Scenario Dashboards</h2>
        <div class="nav-grid">
"""
    
    # Add cards for each scenario
    for rank, (_, row) in enumerate(df.iterrows(), 1):
        scenario_id = row['scenario_id']
        
        # Rank badge
        if rank == 1:
            badge = '<span class="rank-badge gold">#1 BEST</span>'
        elif rank == 2:
            badge = '<span class="rank-badge silver">#2</span>'
        elif rank == 3:
            badge = '<span class="rank-badge bronze">#3</span>'
        elif rank <= 5:
            badge = f'<span class="rank-badge">#{rank}</span>'
        else:
            badge = ''
        
        # Value class
        benefit_class = 'positive' if row['net_benefit'] > 0 else 'negative'
        peak_class = 'positive' if row['peak_reduction_pct'] > 0 else 'negative'
        
        html += f"""
            <div class="nav-card">
                <h3>{badge}BTM {row['btm_ratio']:.0%} | Scale {row['scaler_input']:.1f}x</h3>
                <div class="metric">
                    <span class="metric-label">Net Benefit</span>
                    <span class="metric-value {benefit_class}">€{row['net_benefit']:,.0f}</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Peak Reduction</span>
                    <span class="metric-value {peak_class}">{row['peak_reduction_pct']:.1f}%</span>
                </div>
                <div class="metric">
                    <span class="metric-label">FCR Revenue</span>
                    <span class="metric-value">€{row['fcr_revenue']:,.0f}</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Strategy</span>
                    <span class="metric-value">{'Peak' if row['pct_from_peak'] > max(row['pct_from_btm'], row['pct_from_fcr']) else ('FCR' if row['pct_from_fcr'] > row['pct_from_btm'] else 'BTM')}</span>
                </div>
                <a href="scenario_outputs/dashboard_{scenario_id}.html">View Dashboard →</a>
            </div>
"""
    
    html += """
        </div>
    </div>
</body>
</html>
"""
    
    # Save
    if output_file is None:
        output_file = OUTPUTS_DIR / "scenario_master.html"
    
    with open(output_file, 'w') as f:
        f.write(html)
    
    print(f"Master navigation saved to {output_file}")
    return html


# =============================================================================
# VISUALIZATION
# =============================================================================

def create_scenario_comparison_dashboard(
    results_df: pd.DataFrame,
    output_file: Path = None
) -> go.Figure:
    """
    Create comprehensive scenario comparison dashboard.
    """
    
    # Filter to feasible scenarios
    df = results_df[results_df['feasible']].copy()
    
    if len(df) == 0:
        print("No feasible scenarios to visualize!")
        return None
    
    # Create pivot tables for heatmaps
    btm_ratios = sorted(df['btm_ratio'].unique())
    scaler_inputs = sorted(df['scaler_input'].unique())
    
    # Pivot for each metric
    def pivot_metric(metric):
        pivot = df.pivot(index='scaler_input', columns='btm_ratio', values=metric)
        return pivot.reindex(index=scaler_inputs, columns=btm_ratios)
    
    pivot_net_benefit = pivot_metric('net_benefit')
    pivot_peak_savings = pivot_metric('peak_savings')
    pivot_fcr_revenue = pivot_metric('fcr_revenue')
    pivot_btm_savings = pivot_metric('btm_savings')
    pivot_peak_reduction = pivot_metric('peak_reduction_pct')
    pivot_self_consumption = pivot_metric('self_consumption_rate')
    pivot_fcr_participation = pivot_metric('fcr_participation_rate')
    pivot_cycles = pivot_metric('equivalent_cycles')
    
    # Create figure with subplots
    fig = make_subplots(
        rows=4, cols=2,
        subplot_titles=(
            '<b>Net Benefit (€)</b> — Total value created',
            '<b>Value Attribution (%)</b> — Where does value come from?',
            '<b>Peak Reduction (%)</b> — Grid import peak vs demand peak',
            '<b>FCR Participation Rate (%)</b> — Time with active FCR bid',
            '<b>Self-Consumption Rate (%)</b> — Demand met by battery',
            '<b>Battery Cycles</b> — Equivalent full cycles',
            '<b>Optimal Strategy Map</b> — Best value source by scenario',
            '<b>Total Value Breakdown by Scaler</b>'
        ),
        specs=[
            [{"type": "heatmap"}, {"type": "bar"}],
            [{"type": "heatmap"}, {"type": "heatmap"}],
            [{"type": "heatmap"}, {"type": "heatmap"}],
            [{"type": "heatmap"}, {"type": "bar"}]
        ],
        vertical_spacing=0.08,
        horizontal_spacing=0.1
    )
    
    # Color scales
    money_colorscale = 'Greens'
    pct_colorscale = 'Blues'
    
    # --- ROW 1 ---
    
    # 1. Net Benefit Heatmap
    fig.add_trace(go.Heatmap(
        z=pivot_net_benefit.values,
        x=[f'{r:.0%}' for r in btm_ratios],
        y=[f'{s:.1f}x' for s in scaler_inputs],
        colorscale=money_colorscale,
        colorbar=dict(title='€', x=0.45, len=0.2, y=0.88),
        hovertemplate='BTM: %{x}<br>Scaler: %{y}<br>Net Benefit: €%{z:,.0f}<extra></extra>',
        showscale=True
    ), row=1, col=1)
    
    # 2. Value Attribution Stacked Bar
    for i, scaler in enumerate(scaler_inputs):
        scaler_df = df[df['scaler_input'] == scaler]
        
        # BTM savings
        fig.add_trace(go.Bar(
            x=[f'{r:.0%}' for r in scaler_df['btm_ratio']],
            y=scaler_df['pct_from_btm'],
            name=f'BTM ({scaler:.1f}x)' if i == 0 else None,
            marker_color='#27ae60',
            legendgroup='btm',
            showlegend=(i == 0),
            hovertemplate='BTM: %{y:.1f}%<extra></extra>',
            visible=True if i == 2 else 'legendonly'  # Show scaler=1.0 by default
        ), row=1, col=2)
        
        # Peak savings
        fig.add_trace(go.Bar(
            x=[f'{r:.0%}' for r in scaler_df['btm_ratio']],
            y=scaler_df['pct_from_peak'],
            name=f'Peak ({scaler:.1f}x)' if i == 0 else None,
            marker_color='#e67e22',
            legendgroup='peak',
            showlegend=(i == 0),
            hovertemplate='Peak: %{y:.1f}%<extra></extra>',
            visible=True if i == 2 else 'legendonly'
        ), row=1, col=2)
        
        # FCR revenue
        fig.add_trace(go.Bar(
            x=[f'{r:.0%}' for r in scaler_df['btm_ratio']],
            y=scaler_df['pct_from_fcr'],
            name=f'FCR ({scaler:.1f}x)' if i == 0 else None,
            marker_color='#8e44ad',
            legendgroup='fcr',
            showlegend=(i == 0),
            hovertemplate='FCR: %{y:.1f}%<extra></extra>',
            visible=True if i == 2 else 'legendonly'
        ), row=1, col=2)
    
    # --- ROW 2 ---
    
    # 3. Peak Reduction Heatmap
    fig.add_trace(go.Heatmap(
        z=pivot_peak_reduction.values,
        x=[f'{r:.0%}' for r in btm_ratios],
        y=[f'{s:.1f}x' for s in scaler_inputs],
        colorscale='Oranges',
        colorbar=dict(title='%', x=0.45, len=0.2, y=0.62),
        hovertemplate='BTM: %{x}<br>Scaler: %{y}<br>Peak↓: %{z:.1f}%<extra></extra>'
    ), row=2, col=1)
    
    # 4. FCR Participation Heatmap
    fig.add_trace(go.Heatmap(
        z=pivot_fcr_participation.values,
        x=[f'{r:.0%}' for r in btm_ratios],
        y=[f'{s:.1f}x' for s in scaler_inputs],
        colorscale='Purples',
        colorbar=dict(title='%', x=1.0, len=0.2, y=0.62),
        hovertemplate='BTM: %{x}<br>Scaler: %{y}<br>FCR: %{z:.1f}%<extra></extra>'
    ), row=2, col=2)
    
    # --- ROW 3 ---
    
    # 5. Self-Consumption Heatmap
    fig.add_trace(go.Heatmap(
        z=pivot_self_consumption.values,
        x=[f'{r:.0%}' for r in btm_ratios],
        y=[f'{s:.1f}x' for s in scaler_inputs],
        colorscale='Teal',
        colorbar=dict(title='%', x=0.45, len=0.2, y=0.36),
        hovertemplate='BTM: %{x}<br>Scaler: %{y}<br>Self-Cons: %{z:.1f}%<extra></extra>'
    ), row=3, col=1)
    
    # 6. Battery Cycles Heatmap
    fig.add_trace(go.Heatmap(
        z=pivot_cycles.values,
        x=[f'{r:.0%}' for r in btm_ratios],
        y=[f'{s:.1f}x' for s in scaler_inputs],
        colorscale='Reds',
        colorbar=dict(title='cycles', x=1.0, len=0.2, y=0.36),
        hovertemplate='BTM: %{x}<br>Scaler: %{y}<br>Cycles: %{z:.1f}<extra></extra>'
    ), row=3, col=2)
    
    # --- ROW 4 ---
    
    # 7. Optimal Strategy Map (which value source dominates)
    strategy_matrix = np.zeros((len(scaler_inputs), len(btm_ratios)))
    strategy_labels = []
    
    for i, scaler in enumerate(scaler_inputs):
        row_labels = []
        for j, btm in enumerate(btm_ratios):
            row = df[(df['scaler_input'] == scaler) & (df['btm_ratio'] == btm)]
            if len(row) > 0:
                row = row.iloc[0]
                # Determine dominant strategy
                max_source = max(row['pct_from_btm'], row['pct_from_peak'], row['pct_from_fcr'])
                if row['pct_from_peak'] == max_source:
                    strategy_matrix[i, j] = 1  # Peak
                    row_labels.append('Peak')
                elif row['pct_from_fcr'] == max_source:
                    strategy_matrix[i, j] = 2  # FCR
                    row_labels.append('FCR')
                else:
                    strategy_matrix[i, j] = 0  # BTM
                    row_labels.append('BTM')
            else:
                strategy_matrix[i, j] = -1
                row_labels.append('N/A')
        strategy_labels.append(row_labels)
    
    # Custom colorscale for strategies
    strategy_colorscale = [
        [0, '#27ae60'],    # BTM - Green
        [0.33, '#27ae60'],
        [0.34, '#e67e22'], # Peak - Orange
        [0.66, '#e67e22'],
        [0.67, '#8e44ad'], # FCR - Purple
        [1.0, '#8e44ad']
    ]
    
    fig.add_trace(go.Heatmap(
        z=strategy_matrix,
        x=[f'{r:.0%}' for r in btm_ratios],
        y=[f'{s:.1f}x' for s in scaler_inputs],
        colorscale=strategy_colorscale,
        showscale=False,
        hovertemplate='BTM: %{x}<br>Scaler: %{y}<br>Strategy: %{text}<extra></extra>',
        text=strategy_labels
    ), row=4, col=1)
    
    # Add strategy legend annotation
    fig.add_annotation(
        xref='x7', yref='y7',
        x=0.5, y=-0.3,
        text='<b>Legend:</b> 🟢 BTM | 🟠 Peak | 🟣 FCR',
        showarrow=False,
        font=dict(size=10)
    )
    
    # 8. Total Value Breakdown by Scaler (grouped bar)
    scalers_str = [f'{s:.1f}x' for s in scaler_inputs]
    
    # Average across BTM ratios for each scaler
    avg_by_scaler = df.groupby('scaler_input').agg({
        'btm_savings': 'mean',
        'peak_savings': 'mean',
        'fcr_revenue': 'mean',
        'net_benefit': 'mean'
    }).reset_index()
    
    fig.add_trace(go.Bar(
        x=scalers_str,
        y=avg_by_scaler['btm_savings'],
        name='BTM Savings (avg)',
        marker_color='#27ae60',
        showlegend=False
    ), row=4, col=2)
    
    fig.add_trace(go.Bar(
        x=scalers_str,
        y=avg_by_scaler['peak_savings'],
        name='Peak Savings (avg)',
        marker_color='#e67e22',
        showlegend=False
    ), row=4, col=2)
    
    fig.add_trace(go.Bar(
        x=scalers_str,
        y=avg_by_scaler['fcr_revenue'],
        name='FCR Revenue (avg)',
        marker_color='#8e44ad',
        showlegend=False
    ), row=4, col=2)
    
    # Layout
    fig.update_layout(
        title=dict(
            text='<b>Scenario Analysis Dashboard</b> — BTM Ratio vs Load/Battery Scaling',
            font=dict(size=18)
        ),
        height=1600,
        template='plotly_white',
        barmode='stack',
        showlegend=True,
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=-0.05,
            xanchor='center',
            x=0.5
        )
    )
    
    # Axis labels
    for row in range(1, 5):
        fig.update_xaxes(title_text='BTM Ratio', row=row, col=1)
        fig.update_yaxes(title_text='Load Scaler', row=row, col=1)
    
    fig.update_xaxes(title_text='BTM Ratio', row=1, col=2)
    fig.update_yaxes(title_text='% of Value', row=1, col=2)
    fig.update_xaxes(title_text='Load Scaler', row=4, col=2)
    fig.update_yaxes(title_text='€ (average)', row=4, col=2)
    
    if output_file:
        fig.write_html(str(output_file), include_plotlyjs='cdn')
        print(f"Dashboard saved to {output_file}")
    
    return fig


def create_detailed_insights_report(
    results_df: pd.DataFrame,
    output_file: Path = None
) -> go.Figure:
    """
    Create detailed insights visualization with key findings.
    Uses contour plots instead of 3D surfaces for better readability.
    """
    
    df = results_df[results_df['feasible']].copy()
    
    # Handle empty dataframe
    if len(df) == 0:
        print("⚠️  No feasible scenarios to create insights report!")
        return None
    
    fig = make_subplots(
        rows=3, cols=2,
        subplot_titles=(
            '<b>Net Benefit Contours (€)</b> — Higher is better',
            '<b>Optimal BTM Ratio by Scaler</b> — Path to maximum value',
            '<b>Value Composition Shift</b> — Peak vs FCR dominance',
            '<b>Peak Shaving Effectiveness</b> — By load scaler',
            '<b>FCR Revenue vs FTM Capacity</b> — FCR needs FTM allocation',
            '<b>Value per Cycle</b> — Battery utilization efficiency'
        ),
        specs=[
            [{"type": "contour"}, {"type": "scatter"}],
            [{"type": "scatter"}, {"type": "scatter"}],
            [{"type": "scatter"}, {"type": "scatter"}]
        ],
        vertical_spacing=0.12,
        horizontal_spacing=0.12
    )
    
    btm_ratios = sorted(df['btm_ratio'].unique())
    scaler_inputs = sorted(df['scaler_input'].unique())
    
    # 1. Contour Plot of Net Benefit (better than 3D surface for interpretation)
    pivot_benefit = df.pivot(index='scaler_input', columns='btm_ratio', values='net_benefit')
    fig.add_trace(go.Contour(
        z=pivot_benefit.values,
        x=btm_ratios,
        y=scaler_inputs,
        colorscale='Viridis',
        showscale=True,
        colorbar=dict(title='€', x=0.45, len=0.25, y=0.88),
        contours=dict(showlabels=True, labelfont=dict(size=10, color='white')),
        hovertemplate='BTM: %{x:.0%}<br>Scaler: %{y:.1f}x<br>Benefit: €%{z:,.0f}<extra></extra>'
    ), row=1, col=1)
    
    # Mark optimal point on contour
    best_idx = df['net_benefit'].idxmax()
    best_row = df.loc[best_idx]
    fig.add_trace(go.Scatter(
        x=[best_row['btm_ratio']],
        y=[best_row['scaler_input']],
        mode='markers+text',
        marker=dict(size=15, color='red', symbol='star'),
        text=[f"€{best_row['net_benefit']:,.0f}"],
        textposition='top center',
        textfont=dict(color='red', size=12),
        name='Optimal',
        showlegend=False
    ), row=1, col=1)
    
    # 2. Optimal BTM Ratio by Scaler
    optimal_btm = df.loc[df.groupby('scaler_input')['net_benefit'].idxmax()]
    fig.add_trace(go.Scatter(
        x=optimal_btm['scaler_input'],
        y=optimal_btm['btm_ratio'],
        mode='markers+lines+text',
        text=[f'€{b:,.0f}' for b in optimal_btm['net_benefit']],
        textposition='top center',
        marker=dict(size=15, color=optimal_btm['net_benefit'], colorscale='Greens'),
        line=dict(color='#2ecc71', width=2),
        name='Optimal BTM'
    ), row=1, col=2)
    
    # 3. Value Composition by Scaler (lines)
    for scaler in scaler_inputs:
        scaler_df = df[df['scaler_input'] == scaler].sort_values('btm_ratio')
        is_main = (scaler == 1.0)
        fig.add_trace(go.Scatter(
            x=scaler_df['btm_ratio'],
            y=scaler_df['pct_from_peak'],
            mode='lines',
            name=f'Peak {scaler}x',
            line=dict(color='#e67e22', dash='solid' if is_main else 'dot'),
            showlegend=bool(is_main)
        ), row=2, col=1)
        
        fig.add_trace(go.Scatter(
            x=scaler_df['btm_ratio'],
            y=scaler_df['pct_from_fcr'],
            mode='lines',
            name=f'FCR {scaler}x',
            line=dict(color='#8e44ad', dash='solid' if is_main else 'dot'),
            showlegend=bool(is_main)
        ), row=2, col=1)
    
    # 4. Peak Shaving vs BTM Ratio
    for scaler in [0.5, 1.0, 5.0]:
        scaler_df = df[df['scaler_input'] == scaler].sort_values('btm_ratio')
        if len(scaler_df) > 0:
            fig.add_trace(go.Scatter(
                x=scaler_df['btm_ratio'],
                y=scaler_df['peak_reduction_pct'],
                mode='lines+markers',
                name=f'{scaler}x load',
                marker=dict(size=8),
            ), row=2, col=2)
    
    # 5. FCR Revenue vs FTM Capacity (btm_ratio)
    for scaler in [0.5, 1.0, 5.0]:
        scaler_df = df[df['scaler_input'] == scaler].sort_values('btm_ratio')
        if len(scaler_df) > 0:
            fig.add_trace(go.Scatter(
                x=1 - scaler_df['btm_ratio'],  # FTM ratio
                y=scaler_df['fcr_revenue'],
                mode='lines+markers',
                name=f'{scaler}x load',
                marker=dict(size=8),
            ), row=3, col=1)
    
    # 6. Value per Cycle (efficiency metric)
    df['value_per_cycle'] = df['net_benefit'] / df['equivalent_cycles'].replace(0, np.nan)
    
    for scaler in [0.2, 1.0, 5.0]:
        scaler_df = df[df['scaler_input'] == scaler].sort_values('btm_ratio')
        if len(scaler_df) > 0:
            fig.add_trace(go.Scatter(
                x=scaler_df['btm_ratio'],
                y=scaler_df['value_per_cycle'],
                mode='lines+markers',
                name=f'{scaler}x',
                marker=dict(size=8),
                hovertemplate='BTM: %{x:.0%}<br>€/cycle: %{y:,.0f}<extra></extra>'
            ), row=3, col=2)
    
    # Layout
    fig.update_layout(
        title=dict(
            text='<b>Detailed Scenario Insights</b>',
            font=dict(size=18)
        ),
        height=1200,
        template='plotly_white',
        showlegend=True
    )
    
    # Axis labels
    fig.update_xaxes(title_text='BTM Ratio', row=1, col=1)
    fig.update_yaxes(title_text='Load Scaler', row=1, col=1)
    fig.update_xaxes(title_text='Load Scaler', row=1, col=2)
    fig.update_yaxes(title_text='Optimal BTM Ratio', row=1, col=2)
    fig.update_xaxes(title_text='BTM Ratio', row=2, col=1)
    fig.update_yaxes(title_text='% of Value', row=2, col=1)
    fig.update_xaxes(title_text='BTM Ratio', row=2, col=2)
    fig.update_yaxes(title_text='Peak Reduction (%)', row=2, col=2)
    fig.update_xaxes(title_text='FTM Ratio (1 - BTM)', row=3, col=1)
    fig.update_yaxes(title_text='FCR Revenue (€)', row=3, col=1)
    fig.update_xaxes(title_text='BTM Ratio', row=3, col=2)
    fig.update_yaxes(title_text='€ per Cycle', row=3, col=2)
    
    if output_file:
        fig.write_html(str(output_file), include_plotlyjs='cdn')
        print(f"Insights report saved to {output_file}")
    
    return fig


def print_scenario_summary(results_df: pd.DataFrame):
    """Print key insights from scenario analysis."""
    
    df = results_df[results_df['feasible']].copy()
    
    print("\n" + "="*80)
    print("SCENARIO ANALYSIS SUMMARY")
    print("="*80)
    
    if len(df) == 0:
        print("\n⚠️  No feasible scenarios found!")
        print("   Check solver availability and model constraints.")
        return
    
    # Top 5 scenarios ranked by net benefit
    print(f"\n📋 TOP 5 SCENARIOS (by Net Benefit):")
    print(f"   {'Rank':<6}{'BTM':<8}{'Scaler':<10}{'Net Benefit':<15}{'Peak↓':<10}{'FCR Rev':<12}{'Strategy'}")
    print("   " + "-"*75)
    
    top5 = df.nlargest(min(5, len(df)), 'net_benefit')
    for rank, (_, row) in enumerate(top5.iterrows(), 1):
        if row['pct_from_peak'] > max(row['pct_from_btm'], row['pct_from_fcr']):
            strategy = 'Peak'
        elif row['pct_from_fcr'] > max(row['pct_from_btm'], row['pct_from_peak']):
            strategy = 'FCR'
        else:
            strategy = 'BTM'
        print(f"   {rank:<6}{row['btm_ratio']:.0%}{'':<5}{row['scaler_input']:.1f}x{'':<7}"
              f"€{row['net_benefit']:>11,.0f}{'':<3}{row['peak_reduction_pct']:>6.1f}%{'':<3}"
              f"€{row['fcr_revenue']:>9,.0f}{'':<3}{strategy}")
    
    # Best overall scenario
    best = df.loc[df['net_benefit'].idxmax()]
    print(f"\n🏆 BEST SCENARIO:")
    print(f"   BTM Ratio: {best['btm_ratio']:.0%}")
    print(f"   Load Scaler: {best['scaler_input']:.1f}x")
    print(f"   Net Benefit: €{best['net_benefit']:,.0f}")
    print(f"   Peak Reduction: {best['peak_reduction_pct']:.1f}%")
    print(f"   FCR Revenue: €{best['fcr_revenue']:,.0f}")
    
    # Optimal BTM by scaler
    print(f"\n📊 OPTIMAL BTM RATIO BY LOAD SCALER:")
    print(f"   {'Scaler':<10} {'BTM Ratio':<12} {'Net Benefit':<15} {'Strategy':<10}")
    print("   " + "-"*50)
    
    for scaler in sorted(df['scaler_input'].unique()):
        scaler_df = df[df['scaler_input'] == scaler]
        best_row = scaler_df.loc[scaler_df['net_benefit'].idxmax()]
        
        # Determine dominant strategy
        if best_row['pct_from_peak'] > max(best_row['pct_from_btm'], best_row['pct_from_fcr']):
            strategy = 'Peak'
        elif best_row['pct_from_fcr'] > max(best_row['pct_from_btm'], best_row['pct_from_peak']):
            strategy = 'FCR'
        else:
            strategy = 'BTM'
        
        print(f"   {scaler:.1f}x{'':<7} {best_row['btm_ratio']:.0%}{'':<9} "
              f"€{best_row['net_benefit']:>12,.0f}  {strategy:<10}")
    
    # Key insights
    print(f"\n💡 KEY INSIGHTS:")
    
    # 1. When is FCR dominant?
    fcr_dominant = df[df['pct_from_fcr'] > 40]
    if len(fcr_dominant) > 0:
        avg_btm = fcr_dominant['btm_ratio'].mean()
        avg_scaler = fcr_dominant['scaler_input'].mean()
        print(f"   • FCR dominates (>40%) when: BTM≈{avg_btm:.0%}, Scaler≈{avg_scaler:.1f}x")
    
    # 2. When is Peak Shaving dominant?
    peak_dominant = df[df['pct_from_peak'] > 60]
    if len(peak_dominant) > 0:
        avg_btm = peak_dominant['btm_ratio'].mean()
        avg_scaler = peak_dominant['scaler_input'].mean()
        print(f"   • Peak dominates (>60%) when: BTM≈{avg_btm:.0%}, Scaler≈{avg_scaler:.1f}x")
    
    # 3. Self-consumption trends
    high_self_cons = df[df['self_consumption_rate'] > 30]
    if len(high_self_cons) > 0:
        print(f"   • High self-consumption (>30%): Typically with BTM>{high_self_cons['btm_ratio'].min():.0%}")
    
    # 4. FCR participation
    no_fcr = df[df['fcr_participation_rate'] == 0]
    if len(no_fcr) > 0:
        print(f"   • No FCR participation when BTM≥{no_fcr['btm_ratio'].min():.0%} (insufficient FTM capacity)")
    
    print("\n" + "="*80)


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    
    print("="*70)
    print("MULTI-BATTERY VPP - SCENARIO ANALYSIS")
    print("="*70)
    
    # =========================================================================
    # DATE RANGE CONFIGURATION
    # =========================================================================
    # Specify the simulation period (YYYY-MM-DD format)
    
    START_DATE = "2024-01-01"   # Start date
    END_DATE = "2024-12-31"     # End date (1 week for testing)
    
    # For longer periods:
    # START_DATE = "2024-01-01"
    # END_DATE = "2024-01-31"    # Full month
    
    # For full year (warning: very slow):
    # START_DATE = None
    # END_DATE = None
    
    # =========================================================================
    # SCENARIO PARAMETERS  
    # =========================================================================
    
    # Reduced set for initial testing - avoiding edge cases (0.0 and 1.0 btm_ratio)
    BTM_RATIOS = [0.2,0.5,0.8]  # Skip 0.0 and 1.0 edge cases
    SCALER_INPUTS = [1.0]              # Single scaler for quick test
    
    # Full set (uncomment when ready):
    # BTM_RATIOS = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    # SCALER_INPUTS = [0.2, 0.5, 1.0, 1.5, 5.0]
    
    # Annual peak tariff (EUR/kW/year) - will be prorated for simulation period
    C_PEAK_ANNUAL = 192.66
    
    # Create template site configs (will be modified per scenario)
    load_columns = ['LG 01', 'LG 02', 'LG 03', 'LG 04', 'LG 05',
                    'LG 06', 'LG 07', 'LG 08', 'LG 09', 'LG 10',
                    'LG 11', 'LG 12', 'LG 13', 'LG 14', 'LG 15', 'LG 18',
                    'LG 19', 'LG 20', 'LG 21', 'LG 22', 'LG 23', 'LG 24',
                    'LG 25', 'LG 26', 'LG 27', 'LG 28', 'LG 29', 'LG 30']
    
    site_configs_template = []
    for i, col in enumerate(load_columns):
        site_configs_template.append(SiteConfig(
            site_id=f'Site_{i+1:02d}',
            load_column=col,
            battery=BatterySpec(
                name=f'Luna_{i+1}',
                E_max=215.0,
                P_max=108.0,
                eta_ch=0.974,
                eta_dis=0.974,
                I0=73000.0,
                V_bat=777.0
            ),
            btm_ratio=0.4,  # Will be overwritten
            P_buy_max=200.0,
            P_sell_max=200.0
        ))
    
    # Output directory
    output_dir = OUTPUTS_DIR / "scenario_outputs"
    
    # Run all scenarios (generates individual dashboards automatically)
    results_df, scenario_data = run_all_scenarios(
        btm_ratios=BTM_RATIOS,
        scaler_inputs=SCALER_INPUTS,
        site_configs_template=site_configs_template,
        start_date=START_DATE,
        end_date=END_DATE,
        C_peak_annual=C_PEAK_ANNUAL,
        verbose=True,
        output_dir=output_dir,
        generate_individual_dashboards=True
    )
    
    # Save results CSV
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(OUTPUTS_DIR / "scenario_results.csv", index=False)
    print(f"\nResults saved to outputs/scenario_results.csv")
    
    # =========================================================================
    # SAVE RESULTS FOR LATER DASHBOARD GENERATION
    # =========================================================================
    
    SAVE_RESULTS = True  # Set to True to save results for later use
    
    if SAVE_RESULTS:
        try:
            from lib.results_io import save_scenario_results
            
            # Build base_data info for saving
            base_data_info = {
                'time_index': scenario_data[0]['time_index'] if scenario_data else [],
                'num_steps': scenario_data[0]['num_steps'] if scenario_data else 0,
                'num_sites': scenario_data[0]['num_sites'] if scenario_data else 0,
                'site_names': scenario_data[0].get('site_names', []) if scenario_data else [],
            }
            
            save_scenario_results(
                results_df=results_df,
                all_results=scenario_data,
                base_data=base_data_info,
                metadata={
                    'start_date': START_DATE,
                    'end_date': END_DATE,
                    'btm_ratios': BTM_RATIOS,
                    'scaler_inputs': SCALER_INPUTS,
                    'C_peak_annual': C_PEAK_ANNUAL,
                }
            )
        except Exception as e:
            print(f"⚠️  Could not save results: {e}")
    
    # Print summary
    print_scenario_summary(results_df)
    
    # Check if any scenarios were feasible
    n_feasible = results_df['feasible'].sum() if 'feasible' in results_df.columns else 0
    
    # =========================================================================
    # DASHBOARD GENERATION (can be disabled)
    # =========================================================================
    
    GENERATE_DASHBOARDS = True  # Set to False to skip dashboard generation
    
    if n_feasible == 0:
        print("\n" + "="*70)
        print("⚠️  NO FEASIBLE SCENARIOS FOUND")
        print("="*70)
        print("\nPossible causes:")
        print("  1. Solver not installed or not working (check Gurobi/CPLEX/CBC)")
        print("  2. Model is infeasible (check constraints)")
        print("  3. Solver timeout (increase TimeLimit)")
        print("  4. Memory issues (try smaller date range)")
        print("\nCheck the terminal output above for solver error messages.")
    elif GENERATE_DASHBOARDS:
        # Create comparison visualizations
        print("\nGenerating comparison visualizations...")
        
        fig1 = create_scenario_comparison_dashboard(
            results_df,
            output_file=OUTPUTS_DIR / "scenario_comparison_dashboard.html"
        )
        
        fig2 = create_detailed_insights_report(
            results_df,
            output_file=OUTPUTS_DIR / "scenario_insights.html"
        )
        
        # Create master navigation page
        print("\nGenerating master navigation...")
        create_master_navigation(
            results_df,
            output_dir=output_dir,
            output_file=OUTPUTS_DIR / "scenario_master.html"
        )
        
        print("\n" + "="*70)
        print("✓ SCENARIO ANALYSIS COMPLETE!")
        print("="*70)
        print("\nGenerated files:")
        print(f"  📄 outputs/scenario_results.csv - Raw results data")
        print(f"  🌐 outputs/scenario_master.html - MASTER NAVIGATION (start here!)")
        print(f"  📊 outputs/scenario_comparison_dashboard.html - Comparison heatmaps")
        print(f"  📈 outputs/scenario_insights.html - Detailed insights")
        print(f"  📁 outputs/scenario_outputs/ - Individual scenario dashboards")
        print(f"\nOpen outputs/scenario_master.html in your browser to navigate all results!")
    else:
        print("\n⚠️  Dashboard generation skipped (GENERATE_DASHBOARDS = False)")
        print("   Run: python generate_dashboard.py --scenarios to create dashboards from saved results")

