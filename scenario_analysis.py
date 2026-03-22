"""
Scenario Analysis for Multi-Battery VPP
=========================================
Runs named scenarios (Baseline, No FCR, Full BTM, Full FTM,
Three cycles, No limit, Uncertain prices, Degradation parameters)
with configurable time windows, and generates comparison dashboards.
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

# Default degradation parameters used across most scenarios
DEFAULT_DEGRADATION = (1e-11, 1e-11, 1e-10)  # (a, b, c)
# Stressed degradation parameters for sensitivity analysis
STRESS_DEGRADATION = (5e-11, 5e-11, 5e-10)

@dataclass
class ScenarioConfig:
    """Definition of a single named scenario to run."""
    name: str
    btm_ratio: float
    ftm_ratio: float            # convenience; always 1 - btm_ratio
    scaler_input: float = 1.0
    enable_fcr: bool = True
    daily_cycle_limit: Optional[float] = 2.0   # None = no limit; 1 cycle = 1 round-trip (ch+dis)
    use_forecast_prices: bool = False
    degradation_params: Tuple[float, float, float] = DEFAULT_DEGRADATION

    @property
    def degradation_label(self) -> str:
        return "stress" if self.degradation_params != DEFAULT_DEGRADATION else "base"

    @property
    def scenario_id(self) -> str:
        slug = self.name.lower().replace(" ", "_")
        return f"{slug}_btm{self.btm_ratio:.1f}"


# Predefined scenarios matching the Excel table
# daily_cycle_limit: max round-trips per day (1 cycle = charge E + discharge E).
SCENARIO_DEFS: List[ScenarioConfig] = [
    ScenarioConfig(
        name="Baseline",
        btm_ratio=0.5, ftm_ratio=0.5,
        daily_cycle_limit=2.0,
    ),
    ScenarioConfig(
        name="No FCR",
        btm_ratio=0.5, ftm_ratio=0.5,
        enable_fcr=False,
        daily_cycle_limit=2.0,
    ),
    ScenarioConfig(
        name="Full BTM",
        btm_ratio=1.0, ftm_ratio=0.0,
        daily_cycle_limit=2.0,
    ),
    ScenarioConfig(
        name="Full FTM",
        btm_ratio=0.0, ftm_ratio=1.0,
        daily_cycle_limit=2.0,
    ),
    ScenarioConfig(
        name="Three cycles",
        btm_ratio=0.5, ftm_ratio=0.5,
        daily_cycle_limit=3.0,
    ),
    ScenarioConfig(
        name="No limit",
        btm_ratio=0.5, ftm_ratio=0.5,
        daily_cycle_limit=None,
    ),
    ScenarioConfig(
        name="Uncertain prices",
        btm_ratio=0.5, ftm_ratio=0.5,
        use_forecast_prices=True,
        daily_cycle_limit=2.0,
    ),
    ScenarioConfig(
        name="Degradation parameters",
        btm_ratio=0.5, ftm_ratio=0.5,
        daily_cycle_limit=2.0,
        degradation_params=STRESS_DEGRADATION,
    ),
]


@dataclass
class ScenarioResult:
    """Results from a single scenario run."""
    # Scenario identification
    scenario_name: str
    scenario_id: str
    btm_ratio: float
    scaler_input: float

    # Scenario flags (recorded for dashboards / CSV)
    enable_fcr: bool
    daily_cycle_limit: Optional[float]
    use_forecast_prices: bool
    degradation_label: str

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
    self_consumption_rate: float
    grid_dependency: float
    
    avg_soc_btm: float
    avg_soc_ftm: float
    avg_fcr_bid: float
    fcr_participation_rate: float
    
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
    avg_power_utilization: float


def _empty_result(scenario: ScenarioConfig, solver_status: str, solve_time: float) -> ScenarioResult:
    """Helper: build a zero-filled ScenarioResult for infeasible / failed runs."""
    return ScenarioResult(
        scenario_name=scenario.name, scenario_id=scenario.scenario_id,
        btm_ratio=scenario.btm_ratio, scaler_input=scenario.scaler_input,
        enable_fcr=scenario.enable_fcr, daily_cycle_limit=scenario.daily_cycle_limit,
        use_forecast_prices=scenario.use_forecast_prices,
        degradation_label=scenario.degradation_label,
        feasible=False, solver_status=solver_status, solve_time=solve_time,
        net_benefit=0, btm_savings=0, peak_savings=0, fcr_revenue=0, export_revenue=0,
        degradation_cost=0, baseline_cost=0, optimized_cost=0,
        total_demand_kwh=0, total_import_kwh=0, total_export_kwh=0,
        self_consumption_rate=0, grid_dependency=0,
        avg_soc_btm=0, avg_soc_ftm=0, avg_fcr_bid=0, fcr_participation_rate=0,
        peak_demand=0, peak_import=0, peak_reduction_pct=0,
        pct_from_btm=0, pct_from_peak=0, pct_from_fcr=0, pct_from_export=0,
        total_charge_kwh=0, total_discharge_kwh=0, equivalent_cycles=0, avg_power_utilization=0,
    )


def run_single_scenario(
    scenario: ScenarioConfig,
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
    Run a single named scenario.

    Args:
        scenario: Full scenario definition (name, btm_ratio, flags, etc.)
        site_configs_template: Template site configurations
        base_data: Pre-loaded data dictionary
        fcr_signal_up / fcr_signal_down: FCR activation profiles
        delta_t: Timestep in hours (0.25 = 15 min)
        C_peak: Peak tariff (already prorated for simulation period)
        start_date / end_date: Simulation window
        verbose: Print detailed output
    """
    btm_ratio = scenario.btm_ratio
    scaler_input = scenario.scaler_input
    scenario_id = scenario.scenario_id

    if verbose:
        print(f"\n{'='*60}")
        print(f"Running Scenario: {scenario.name} ({scenario_id})")
        print(f"  BTM Ratio: {btm_ratio:.1%}  |  FCR: {'On' if scenario.enable_fcr else 'Off'}")
        print(f"  Cycle Limit: {scenario.daily_cycle_limit}  |  Forecast prices: {scenario.use_forecast_prices}")
        print(f"  Degradation: {scenario.degradation_label}")
        print(f"{'='*60}")

    start_time = time.time()

    adjusted_P_max = max(500, 200 * scaler_input * 3)

    site_configs = []
    for template in site_configs_template:
        site_configs.append(SiteConfig(
            site_id=template.site_id,
            load_column=template.load_column,
            battery=template.battery,
            btm_ratio=btm_ratio,
            P_buy_max=adjusted_P_max,
            P_sell_max=adjusted_P_max,
        ))

    data = load_multi_site_data(
        data_dir=DATA_DIR,
        site_configs=site_configs,
        scale_loads_to_battery=True,
        scaler_input=scaler_input,
        start_date=start_date,
        end_date=end_date,
    )

    num_steps = data['num_steps']
    if len(fcr_signal_up) >= num_steps:
        fcr_signal_up_truncated = fcr_signal_up[:num_steps]
        fcr_signal_down_truncated = fcr_signal_down[:num_steps]
    else:
        pad_length = num_steps - len(fcr_signal_up)
        if verbose:
            print(f"  FCR data shorter by {pad_length} steps - padding with zeros")
        fcr_signal_up_truncated = np.concatenate([fcr_signal_up, np.zeros(pad_length)])
        fcr_signal_down_truncated = np.concatenate([fcr_signal_down, np.zeros(pad_length)])

    # Physical FCR eligibility (FTM capacity >= 1 MW)
    total_ftm_capacity = sum(cfg.battery.P_max * (1 - cfg.btm_ratio) for cfg in site_configs)
    fcr_physically_possible = total_ftm_capacity >= 1000

    # Combine physical and scenario-level FCR switch
    fcr_enabled = fcr_physically_possible and scenario.enable_fcr

    if not fcr_enabled:
        min_fcr_bid = 0
        if verbose:
            reason = "scenario disabled" if not scenario.enable_fcr else f"FTM {total_ftm_capacity:.0f} kW < 1 MW"
            print(f"  FCR DISABLED ({reason})")
    else:
        min_fcr_bid = 1000
        if verbose:
            print(f"  FCR enabled (FTM {total_ftm_capacity:.0f} kW)")

    try:
        data_adjusted = data.copy()
        if not fcr_enabled:
            data_adjusted['fcr_prices'] = np.zeros(len(data['fcr_prices']))

        a, b, c = scenario.degradation_params

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
            a=a, b=b, c=c,
            forecast_day_ahead=scenario.use_forecast_prices,
            daily_cycle_limit=scenario.daily_cycle_limit,
            force_disable_fcr=(not scenario.enable_fcr),
        )

        solver = SolverFactory('gurobi')

        num_timesteps = data['num_steps']
        num_sites = len(site_configs)
        problem_size = num_timesteps * num_sites
        if problem_size < 10000:
            time_limit = 300
        elif problem_size < 100000:
            time_limit = 1800
        else:
            time_limit = 7200

        solver.options['MIPGap'] = 0.05
        solver.options['TimeLimit'] = time_limit
        solver.options['Threads'] = 0
        solver.options['Method'] = 2
        solver.options['Presolve'] = 2
        solver.options['MIPFocus'] = 1
        solver.options['OutputFlag'] = 1

        if verbose:
            print(f"    Problem: {num_timesteps:,} timesteps x {num_sites} sites")
            print(f"    TimeLimit: {time_limit // 60} min, MIPGap: 5%")

        results = solver.solve(model, tee=verbose)
        solve_time = time.time() - start_time

        from pyomo.opt import TerminationCondition
        term_cond = results.solver.termination_condition
        if term_cond not in [TerminationCondition.optimal, TerminationCondition.feasible]:
            if term_cond == TerminationCondition.maxTimeLimit:
                print(f"    TIMEOUT after {solve_time:.0f}s")
            elif term_cond == TerminationCondition.infeasible:
                print(f"    INFEASIBLE - constraints cannot be satisfied")
            else:
                print(f"    FAILED: {term_cond}")
            return _empty_result(scenario, str(term_cond), solve_time), None, None, None, None

        df = extract_results(model, site_configs, data_adjusted)
        financials = calculate_financials(df, site_configs, C_peak, delta_t)

        if not fcr_enabled:
            for site_id in financials['sites']:
                financials['sites'][site_id]['fcr_revenue'] = 0
            financials['portfolio']['fcr_revenue'] = 0
        port = financials['portfolio']

        agg = df.groupby('t').agg({
            'demand': 'sum', 'P_buy': 'sum', 'P_sell': 'sum',
            'P_ch_BTM': 'sum', 'P_dis_BTM': 'sum',
            'P_ch_FTM': 'sum', 'P_dis_FTM': 'sum',
            'P_FCR_bid': 'sum', 'SOC_BTM': 'sum', 'SOC_FTM': 'sum',
        })

        total_E_max = sum(cfg.battery.E_max for cfg in site_configs)
        total_E_btm = sum(cfg.battery.E_max * cfg.btm_ratio for cfg in site_configs)
        total_E_ftm = sum(cfg.battery.E_max * (1 - cfg.btm_ratio) for cfg in site_configs)
        total_P_max = sum(cfg.battery.P_max for cfg in site_configs)

        total_demand_kwh = agg['demand'].sum() * delta_t
        total_import_kwh = agg['P_buy'].sum() * delta_t
        total_export_kwh = agg['P_sell'].sum() * delta_t
        total_charge_kwh = (agg['P_ch_BTM'].sum() + agg['P_ch_FTM'].sum()) * delta_t
        total_discharge_kwh = (agg['P_dis_BTM'].sum() + agg['P_dis_FTM'].sum()) * delta_t

        battery_to_load = min(total_discharge_kwh, total_demand_kwh - total_export_kwh)
        self_consumption_rate = battery_to_load / max(total_demand_kwh, 1) * 100
        grid_dependency = total_import_kwh / max(total_demand_kwh, 1) * 100

        avg_soc_btm = (agg['SOC_BTM'].mean() / total_E_btm * 100) if total_E_btm > 10 else 50
        avg_soc_ftm = (agg['SOC_FTM'].mean() / total_E_ftm * 100) if total_E_ftm > 10 else 50

        avg_fcr_bid = agg['P_FCR_bid'].mean()
        fcr_participation_rate = (agg['P_FCR_bid'] > 0).sum() / len(agg) * 100

        peak_demand = agg['demand'].max()
        peak_import = agg['P_buy'].max()
        peak_reduction_pct = (1 - peak_import / max(peak_demand, 1)) * 100

        export_revenue = (agg['P_sell'] * data['day_ahead'][:len(agg)] * delta_t).sum()

        total_value = abs(port['btm_savings']) + abs(port['peak_savings']) + abs(port['fcr_revenue'])
        if total_value > 0:
            pct_from_btm = abs(port['btm_savings']) / total_value * 100
            pct_from_peak = abs(port['peak_savings']) / total_value * 100
            pct_from_fcr = abs(port['fcr_revenue']) / total_value * 100
        else:
            pct_from_btm = pct_from_peak = pct_from_fcr = 0
        pct_from_export = export_revenue / max(total_value, 1) * 100 if total_value > 0 else 0

        equivalent_cycles = total_discharge_kwh / max(total_E_max, 1)
        avg_power = (agg['P_ch_BTM'] + agg['P_ch_FTM'] + agg['P_dis_BTM'] + agg['P_dis_FTM']).mean()
        avg_power_utilization = avg_power / max(total_P_max, 1) * 100

        if verbose:
            print(f"  Solved in {solve_time:.1f}s | Net Benefit: {port['net_benefit']:,.0f}")

        scenario_result = ScenarioResult(
            scenario_name=scenario.name,
            scenario_id=scenario_id,
            btm_ratio=btm_ratio,
            scaler_input=scaler_input,
            enable_fcr=scenario.enable_fcr,
            daily_cycle_limit=scenario.daily_cycle_limit,
            use_forecast_prices=scenario.use_forecast_prices,
            degradation_label=scenario.degradation_label,
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
            avg_power_utilization=avg_power_utilization,
        )
        return scenario_result, df, financials, site_configs, data

    except Exception as e:
        if verbose:
            print(f"  Error: {e}")
        return _empty_result(scenario, str(e), time.time() - start_time), None, None, None, None


def run_all_scenarios(
    scenarios: List[ScenarioConfig],
    site_configs_template: List[SiteConfig],
    start_date: str = None,
    end_date: str = None,
    C_peak_annual: float = 192.66,
    verbose: bool = True,
    output_dir: Path = None,
    generate_individual_dashboards: bool = True
) -> Tuple[pd.DataFrame, Dict]:
    """
    Run a list of named scenarios, generate dashboards, and return results.

    Args:
        scenarios: List of ScenarioConfig definitions to run
        site_configs_template: Template site configurations
        start_date / end_date: Simulation window (YYYY-MM-DD)
        C_peak_annual: Annual peak tariff EUR/kW (prorated automatically)
        verbose: Print progress
        output_dir: Directory for HTML dashboards
        generate_individual_dashboards: Generate per-scenario dashboards
    """

    from lib.dashboard_multi_battery import create_multi_battery_dashboard

    print("\nPre-loading base data...")
    base_configs = [SiteConfig(
        site_id=cfg.site_id,
        load_column=cfg.load_column,
        battery=cfg.battery,
        btm_ratio=0.4,
        P_buy_max=cfg.P_buy_max,
        P_sell_max=cfg.P_sell_max,
    ) for cfg in site_configs_template]

    base_data = load_multi_site_data(
        data_dir=DATA_DIR,
        site_configs=base_configs,
        scale_loads_to_battery=True,
        scaler_input=1.0,
        start_date=start_date,
        end_date=end_date,
    )

    simulation_days = base_data['num_steps'] / 96
    C_peak = C_peak_annual * (simulation_days / 365.0)

    print("\n" + "=" * 70)
    print("SCENARIO ANALYSIS")
    print("=" * 70)
    print(f"Scenarios: {len(scenarios)}")
    for sc in scenarios:
        print(f"  - {sc.name}  (BTM {sc.btm_ratio:.0%}, FCR={'On' if sc.enable_fcr else 'Off'}, "
              f"cycles={sc.daily_cycle_limit}, forecast={sc.use_forecast_prices}, deg={sc.degradation_label})")
    print(f"Date Range: {base_data['time_index'][0]} to {base_data['time_index'][-1]}")
    print(f"Duration: {simulation_days:.1f} days ({base_data['num_steps']} timesteps)")
    print(f"Peak Tariff: {C_peak:.2f} (prorated from {C_peak_annual:.2f}/year)")
    print("=" * 70)

    if output_dir is None:
        output_dir = OUTPUTS_DIR / "scenario_outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    delta_t = 0.25
    fcr_signal_up, fcr_signal_down = load_fcr_activation_profile(
        data_dir=DATA_DIR,
        start_date=start_date,
        end_date=end_date,
    )

    results = []
    scenario_data = {}
    total = len(scenarios)

    for idx, scenario in enumerate(scenarios, 1):
        print(f"\n[{idx}/{total}] {scenario.name}  btm={scenario.btm_ratio:.1f}")

        result, df_detail, financials, site_configs, data = run_single_scenario(
            scenario=scenario,
            site_configs_template=site_configs_template,
            base_data=base_data,
            fcr_signal_up=fcr_signal_up,
            fcr_signal_down=fcr_signal_down,
            delta_t=delta_t,
            C_peak=C_peak,
            start_date=start_date,
            end_date=end_date,
            verbose=verbose,
        )

        results.append(asdict(result))

        if result.feasible:
            print(f"    Net Benefit: {result.net_benefit:,.0f} | "
                  f"Peak: {result.peak_reduction_pct:.0f}% | "
                  f"FCR: {result.fcr_revenue:,.0f}")

            scenario_data[result.scenario_id] = {
                'result': result,
                'df': df_detail,
                'financials': financials,
                'site_configs': site_configs,
                'data': data,
            }

            if generate_individual_dashboards and df_detail is not None:
                dashboard_file = output_dir / f"dashboard_{result.scenario_id}.html"
                try:
                    create_multi_battery_dashboard(
                        df_detail, financials, site_configs, data,
                        output_file=dashboard_file,
                        delta_t=delta_t,
                        C_peak=C_peak,
                    )
                except Exception as e:
                    print(f"    Dashboard generation failed: {e}")

    results_df = pd.DataFrame(results)

    print("\n" + "=" * 70)
    print("SCENARIO ANALYSIS COMPLETE")
    print(f"Feasible: {results_df['feasible'].sum()} / {len(results_df)}")
    print("=" * 70)

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
        
        scenario_name = row.get('scenario_name', scenario_id)
        fcr_label = 'On' if row.get('enable_fcr', True) else 'Off'

        html += f"""
            <div class="nav-card">
                <h3>{badge}{scenario_name} &mdash; BTM {row['btm_ratio']:.0%} | FCR {fcr_label}</h3>
                <div class="metric">
                    <span class="metric-label">Net Benefit</span>
                    <span class="metric-value {benefit_class}">&euro;{row['net_benefit']:,.0f}</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Peak Reduction</span>
                    <span class="metric-value {peak_class}">{row['peak_reduction_pct']:.1f}%</span>
                </div>
                <div class="metric">
                    <span class="metric-label">FCR Revenue</span>
                    <span class="metric-value">&euro;{row['fcr_revenue']:,.0f}</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Degradation Cost</span>
                    <span class="metric-value">&euro;{row['degradation_cost']:,.0f}</span>
                </div>
                <a href="scenario_outputs/dashboard_{scenario_id}.html">View Dashboard &rarr;</a>
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
    Create a comparison dashboard for named scenarios.
    Uses bar charts keyed by scenario_name (not heatmaps).
    """

    df = results_df[results_df['feasible']].copy()
    if len(df) == 0:
        print("No feasible scenarios to visualize!")
        return None

    name_col = 'scenario_name' if 'scenario_name' in df.columns else 'scenario_id'
    names = df[name_col].tolist()

    fig = make_subplots(
        rows=3, cols=2,
        subplot_titles=(
            '<b>Net Benefit</b>',
            '<b>Value Attribution (%)</b>',
            '<b>Peak Reduction (%)</b>',
            '<b>FCR Participation Rate (%)</b>',
            '<b>Equivalent Battery Cycles</b>',
            '<b>Degradation Cost</b>',
        ),
        vertical_spacing=0.12,
        horizontal_spacing=0.12,
    )

    # 1 — Net benefit
    fig.add_trace(go.Bar(
        x=names, y=df['net_benefit'],
        marker_color=['#27ae60' if v >= 0 else '#e74c3c' for v in df['net_benefit']],
        hovertemplate='%{x}<br>Net Benefit: %{y:,.0f}<extra></extra>',
        showlegend=False,
    ), row=1, col=1)

    # 2 — Value attribution stacked bar
    fig.add_trace(go.Bar(
        x=names, y=df['pct_from_btm'], name='BTM',
        marker_color='#27ae60',
    ), row=1, col=2)
    fig.add_trace(go.Bar(
        x=names, y=df['pct_from_peak'], name='Peak',
        marker_color='#e67e22',
    ), row=1, col=2)
    fig.add_trace(go.Bar(
        x=names, y=df['pct_from_fcr'], name='FCR',
        marker_color='#8e44ad',
    ), row=1, col=2)

    # 3 — Peak reduction
    fig.add_trace(go.Bar(
        x=names, y=df['peak_reduction_pct'],
        marker_color='#e67e22',
        hovertemplate='%{x}<br>Peak Reduction: %{y:.1f}%<extra></extra>',
        showlegend=False,
    ), row=2, col=1)

    # 4 — FCR participation
    fig.add_trace(go.Bar(
        x=names, y=df['fcr_participation_rate'],
        marker_color='#8e44ad',
        hovertemplate='%{x}<br>FCR Participation: %{y:.1f}%<extra></extra>',
        showlegend=False,
    ), row=2, col=2)

    # 5 — Battery cycles
    fig.add_trace(go.Bar(
        x=names, y=df['equivalent_cycles'],
        marker_color='#2980b9',
        hovertemplate='%{x}<br>Cycles: %{y:.1f}<extra></extra>',
        showlegend=False,
    ), row=3, col=1)

    # 6 — Degradation cost
    fig.add_trace(go.Bar(
        x=names, y=df['degradation_cost'],
        marker_color='#e74c3c',
        hovertemplate='%{x}<br>Degradation: %{y:,.0f}<extra></extra>',
        showlegend=False,
    ), row=3, col=2)

    fig.update_layout(
        title=dict(text='<b>Scenario Comparison Dashboard</b>', font=dict(size=18)),
        height=1200,
        template='plotly_white',
        barmode='stack',
        legend=dict(orientation='h', yanchor='bottom', y=-0.08, xanchor='center', x=0.5),
    )

    if output_file:
        fig.write_html(str(output_file), include_plotlyjs='cdn')
        print(f"Dashboard saved to {output_file}")

    return fig


def create_detailed_insights_report(
    results_df: pd.DataFrame,
    output_file: Path = None
) -> go.Figure:
    """
    Create detailed insights for named scenarios using grouped bar / scatter charts.
    """

    df = results_df[results_df['feasible']].copy()
    if len(df) == 0:
        print("No feasible scenarios to create insights report!")
        return None

    name_col = 'scenario_name' if 'scenario_name' in df.columns else 'scenario_id'
    names = df[name_col].tolist()

    df['value_per_cycle'] = df['net_benefit'] / df['equivalent_cycles'].replace(0, np.nan)

    fig = make_subplots(
        rows=3, cols=2,
        subplot_titles=(
            '<b>Financial Breakdown</b>',
            '<b>Self-Consumption vs Grid Dependency</b>',
            '<b>FCR Revenue</b>',
            '<b>Value per Cycle</b>',
            '<b>Average SOC (BTM / FTM)</b>',
            '<b>Power Utilization (%)</b>',
        ),
        vertical_spacing=0.12,
        horizontal_spacing=0.12,
    )

    # 1 — Financials stacked
    fig.add_trace(go.Bar(x=names, y=df['btm_savings'], name='BTM Savings', marker_color='#27ae60'), row=1, col=1)
    fig.add_trace(go.Bar(x=names, y=df['peak_savings'], name='Peak Savings', marker_color='#e67e22'), row=1, col=1)
    fig.add_trace(go.Bar(x=names, y=df['fcr_revenue'], name='FCR Revenue', marker_color='#8e44ad'), row=1, col=1)
    fig.add_trace(go.Bar(x=names, y=-df['degradation_cost'], name='Degradation', marker_color='#e74c3c'), row=1, col=1)

    # 2 — Self-consumption and grid dependency
    fig.add_trace(go.Bar(x=names, y=df['self_consumption_rate'], name='Self-consumption %', marker_color='#2ecc71', showlegend=False), row=1, col=2)
    fig.add_trace(go.Bar(x=names, y=df['grid_dependency'], name='Grid dependency %', marker_color='#3498db', showlegend=False), row=1, col=2)

    # 3 — FCR revenue bar
    fig.add_trace(go.Bar(x=names, y=df['fcr_revenue'], marker_color='#8e44ad', showlegend=False), row=2, col=1)

    # 4 — Value per cycle
    fig.add_trace(go.Bar(x=names, y=df['value_per_cycle'], marker_color='#16a085', showlegend=False), row=2, col=2)

    # 5 — SOC grouped bar
    fig.add_trace(go.Bar(x=names, y=df['avg_soc_btm'], name='BTM SOC', marker_color='#2ecc71', showlegend=False), row=3, col=1)
    fig.add_trace(go.Bar(x=names, y=df['avg_soc_ftm'], name='FTM SOC', marker_color='#9b59b6', showlegend=False), row=3, col=1)

    # 6 — Power utilization
    fig.add_trace(go.Bar(x=names, y=df['avg_power_utilization'], marker_color='#2980b9', showlegend=False), row=3, col=2)

    fig.update_layout(
        title=dict(text='<b>Detailed Scenario Insights</b>', font=dict(size=18)),
        height=1200,
        template='plotly_white',
        barmode='relative',
        legend=dict(orientation='h', yanchor='bottom', y=-0.08, xanchor='center', x=0.5),
    )

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
    
    # All scenarios ranked by net benefit
    has_name = 'scenario_name' in df.columns
    header = f"   {'Rank':<5}{'Scenario':<25}{'BTM':<7}{'Net Benefit':<14}{'Peak':<9}{'FCR Rev':<11}{'Strategy'}"
    print(f"\n  ALL SCENARIOS (ranked by Net Benefit):")
    print(header)
    print("   " + "-" * len(header))

    ranked = df.sort_values('net_benefit', ascending=False)
    for rank, (_, row) in enumerate(ranked.iterrows(), 1):
        if row['pct_from_peak'] > max(row['pct_from_btm'], row['pct_from_fcr']):
            strategy = 'Peak'
        elif row['pct_from_fcr'] > max(row['pct_from_btm'], row['pct_from_peak']):
            strategy = 'FCR'
        else:
            strategy = 'BTM'
        name = row['scenario_name'] if has_name else row['scenario_id']
        print(f"   {rank:<5}{name:<25}{row['btm_ratio']:.0%}{'':<4}"
              f"{row['net_benefit']:>11,.0f}{'':<3}{row['peak_reduction_pct']:>5.1f}%{'':<3}"
              f"{row['fcr_revenue']:>9,.0f}{'':<2}{strategy}")

    best = df.loc[df['net_benefit'].idxmax()]
    best_name = best['scenario_name'] if has_name else best['scenario_id']
    print(f"\n  BEST SCENARIO: {best_name}")
    print(f"   BTM Ratio: {best['btm_ratio']:.0%}")
    print(f"   Net Benefit: {best['net_benefit']:,.0f}")
    print(f"   Peak Reduction: {best['peak_reduction_pct']:.1f}%")
    print(f"   FCR Revenue: {best['fcr_revenue']:,.0f}")

    print("\n" + "=" * 80)


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":

    print("=" * 70)
    print("MULTI-BATTERY VPP - SCENARIO ANALYSIS")
    print("=" * 70)

    # =========================================================================
    # DATE RANGE — change these to run for a week, month, or full year
    # =========================================================================
    START_DATE = "2024-01-01"
    END_DATE   = "2024-01-07"   # 1-week test; change to e.g. "2024-01-31" or "2024-12-31"

    # Annual peak tariff (EUR/kW/year) — prorated automatically
    C_PEAK_ANNUAL = 192.66

    # =========================================================================
    # SCENARIOS — uses SCENARIO_DEFS defined at the top of the file.
    # Comment out entries you don't want to run.
    # =========================================================================
    SCENARIOS_TO_RUN = SCENARIO_DEFS   # all 8 scenarios from the Excel table

    # =========================================================================
    # SITE TEMPLATE (28 batteries, will be modified per scenario)
    # =========================================================================
    load_columns = [
        'LG 01', 'LG 02', 'LG 03', 'LG 04', 'LG 05',
        'LG 06', 'LG 07', 'LG 08', 'LG 09', 'LG 10',
        'LG 11', 'LG 12', 'LG 13', 'LG 14', 'LG 15', 'LG 18',
        'LG 19', 'LG 20', 'LG 21', 'LG 22', 'LG 23', 'LG 24',
        'LG 25', 'LG 26', 'LG 27', 'LG 28', 'LG 29', 'LG 30',
    ]

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
                V_bat=777.0,
            ),
            btm_ratio=0.4,
            P_buy_max=200.0,
            P_sell_max=200.0,
        ))

    output_dir = OUTPUTS_DIR / "scenario_outputs"

    # =========================================================================
    # RUN
    # =========================================================================
    results_df, scenario_data = run_all_scenarios(
        scenarios=SCENARIOS_TO_RUN,
        site_configs_template=site_configs_template,
        start_date=START_DATE,
        end_date=END_DATE,
        C_peak_annual=C_PEAK_ANNUAL,
        verbose=True,
        output_dir=output_dir,
        generate_individual_dashboards=True,
    )

    # Save CSV
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(OUTPUTS_DIR / "scenario_results.csv", index=False)
    print(f"\nResults saved to outputs/scenario_results.csv")

    # =========================================================================
    # OPTIONAL: save results for later dashboard regeneration
    # =========================================================================
    try:
        from lib.results_io import save_scenario_results

        first_key = next(iter(scenario_data), None)
        first_data = scenario_data[first_key]['data'] if first_key else {}
        base_data_info = {
            'time_index': first_data.get('time_index', []),
            'num_steps': first_data.get('num_steps', 0),
            'num_sites': first_data.get('num_sites', 0),
            'site_names': first_data.get('site_names', []),
        }
        save_scenario_results(
            results_df=results_df,
            all_results=scenario_data,
            base_data=base_data_info,
            metadata={
                'start_date': START_DATE,
                'end_date': END_DATE,
                'scenarios': [s.name for s in SCENARIOS_TO_RUN],
                'C_peak_annual': C_PEAK_ANNUAL,
            },
        )
    except Exception as e:
        print(f"Could not save results: {e}")

    # Summary
    print_scenario_summary(results_df)

    n_feasible = results_df['feasible'].sum() if 'feasible' in results_df.columns else 0

    # =========================================================================
    # DASHBOARDS
    # =========================================================================
    GENERATE_DASHBOARDS = True

    if n_feasible == 0:
        print("\n" + "=" * 70)
        print("NO FEASIBLE SCENARIOS FOUND")
        print("=" * 70)
        print("\nPossible causes:")
        print("  1. Solver not installed (check Gurobi/CPLEX/CBC)")
        print("  2. Model is infeasible (check constraints)")
        print("  3. Solver timeout (increase TimeLimit)")
        print("  4. Memory issues (try shorter date range)")
    elif GENERATE_DASHBOARDS:
        print("\nGenerating comparison visualizations...")

        fig1 = create_scenario_comparison_dashboard(
            results_df,
            output_file=OUTPUTS_DIR / "scenario_comparison_dashboard.html",
        )
        fig2 = create_detailed_insights_report(
            results_df,
            output_file=OUTPUTS_DIR / "scenario_insights.html",
        )

        print("\nGenerating master navigation...")
        create_master_navigation(
            results_df,
            output_dir=output_dir,
            output_file=OUTPUTS_DIR / "scenario_master.html",
        )

        print("\n" + "=" * 70)
        print("SCENARIO ANALYSIS COMPLETE!")
        print("=" * 70)
        print("\nGenerated files:")
        print("  outputs/scenario_results.csv            - Raw results data")
        print("  outputs/scenario_master.html            - MASTER NAVIGATION (start here!)")
        print("  outputs/scenario_comparison_dashboard.html - Comparison charts")
        print("  outputs/scenario_insights.html          - Detailed insights")
        print("  outputs/scenario_outputs/               - Individual scenario dashboards")
    else:
        print("\nDashboard generation skipped (GENERATE_DASHBOARDS = False)")

