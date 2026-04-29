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
        name="80% BTM",
        btm_ratio=0.8, ftm_ratio=0.2,
        daily_cycle_limit=2.0,
    ),
    ScenarioConfig(
        name="80% FTM",
        btm_ratio=0.2, ftm_ratio=0.8,
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
    pct_from_da: float
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
        pct_from_btm=0, pct_from_peak=0, pct_from_da=0, pct_from_fcr=0, pct_from_export=0,
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
        day_ahead_arbitrage = (
            port['net_benefit']
            - port['btm_savings']
            - port['peak_savings']
            - port['fcr_revenue']
            + port['degradation_cost']
        )

        total_value = (
            abs(port['btm_savings'])
            + abs(port['peak_savings'])
            + abs(day_ahead_arbitrage)
            + abs(port['fcr_revenue'])
        )
        if total_value > 0:
            pct_from_btm = abs(port['btm_savings']) / total_value * 100
            pct_from_peak = abs(port['peak_savings']) / total_value * 100
            pct_from_da = abs(day_ahead_arbitrage) / total_value * 100
            pct_from_fcr = abs(port['fcr_revenue']) / total_value * 100
        else:
            pct_from_btm = pct_from_peak = pct_from_da = pct_from_fcr = 0
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
            pct_from_da=pct_from_da,
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
            --primary: #23364d;
            --secondary: #2f80ed;
            --success: #1f9d6b;
            --warning: #f2a43a;
            --danger: #d9534f;
            --light: #f3f6fb;
            --dark: #15212f;
            --card-bg: rgba(255,255,255,0.10);
            --card-border: rgba(255,255,255,0.16);
        }
        
        * { box-sizing: border-box; margin: 0; padding: 0; }
        
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: radial-gradient(1200px 400px at 20% -5%, rgba(47,128,237,0.30), transparent 50%),
                        linear-gradient(140deg, var(--dark) 0%, var(--primary) 100%);
            min-height: 100vh;
            color: white;
            padding: 26px;
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
            background: var(--card-bg);
            border-radius: 14px;
            padding: 22px;
            transition: all 0.3s ease;
            border: 1px solid var(--card-border);
            backdrop-filter: blur(4px);
        }
        
        .nav-card:hover {
            transform: translateY(-4px);
            background: rgba(255,255,255,0.14);
            box-shadow: 0 12px 32px rgba(0,0,0,0.25);
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
            padding: 14px 26px;
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
            <a href="scenario_overview_dashboard.html" class="main-link highlight">📊 Scenario Overview</a>
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

def create_scenario_overview_dashboard(
    results_df: pd.DataFrame,
    output_file: Path = None,
    scenario_data: Optional[Dict] = None,
) -> go.Figure:
    """
    Create one merged scenario overview dashboard (comparison + insights).
    """

    df = results_df[results_df['feasible']].copy()
    if len(df) == 0:
        print("No feasible scenarios to visualize!")
        return None

    name_col = 'scenario_name' if 'scenario_name' in df.columns else 'scenario_id'
    names = df[name_col].tolist()
    df['value_per_cycle'] = df['net_benefit'] / df['equivalent_cycles'].replace(0, np.nan)
    df['day_ahead_arbitrage'] = (
        df['net_benefit']
        - df['btm_savings']
        - df['peak_savings']
        - df['fcr_revenue']
        + df['degradation_cost']
    )
    if 'pct_from_da' not in df.columns:
        total_value = (
            df['btm_savings'].abs()
            + df['peak_savings'].abs()
            + df['day_ahead_arbitrage'].abs()
            + df['fcr_revenue'].abs()
        )
        total_value_safe = total_value.replace(0, np.nan)
        df['pct_from_btm'] = 100 * df['btm_savings'].abs() / total_value_safe
        df['pct_from_peak'] = 100 * df['peak_savings'].abs() / total_value_safe
        df['pct_from_da'] = 100 * df['day_ahead_arbitrage'].abs() / total_value_safe
        df['pct_from_fcr'] = 100 * df['fcr_revenue'].abs() / total_value_safe
        df[['pct_from_btm', 'pct_from_peak', 'pct_from_da', 'pct_from_fcr']] = (
            df[['pct_from_btm', 'pct_from_peak', 'pct_from_da', 'pct_from_fcr']].fillna(0.0)
        )

    target_btm = [0.2, 0.5, 0.8]  # Intended BTM ratios to compare
    target_btm_title = "/".join([str(int(round(r * 100))) for r in target_btm])

    fig = make_subplots(
        rows=6, cols=2,
        subplot_titles=(
            '<b>Net Benefit</b>',
            '<b>Value Attribution (%)</b>',
            '<b>Peak Reduction (%)</b>',
            '<b>FCR Participation (%)</b>',
            '<b>Financial Breakdown</b>',
            '<b>Equivalent Cycles / Value per Cycle</b>',
            '<b>SOC Mix (BTM/FTM) & Utilization</b>',
            '<b>Degradation Cost</b>',
            '<b>Customers Savings / Month</b>',
            f'<b>BTM Ratio Comparison ({target_btm_title})</b>',
            '<b>Uncertain Prices vs Baseline (Cash Components)</b>',
            '<b>No FCR vs Baseline (Cash Components)</b>',
        ),
        vertical_spacing=0.16,
        horizontal_spacing=0.12,
        specs=[
            [{"secondary_y": False}, {"secondary_y": False}],
            [{"secondary_y": False}, {"secondary_y": False}],
            [{"secondary_y": False}, {"secondary_y": True}],
            [{"secondary_y": True}, {"secondary_y": False}],
            [{"secondary_y": False}, {"secondary_y": False}],
            [{"secondary_y": False}, {"secondary_y": False}],
        ],
    )

    def _cell_domains_paper(row: int, col: int):
        """
        Get (x_domain, y_domain) in paper coordinates for a subplot cell.
        Relies on Plotly's internal _grid_ref mapping created by make_subplots.
        """
        grid_cell = fig._grid_ref[row - 1][col - 1]
        # In this Plotly version, grid_cell is a 1-tuple containing a SubplotRef.
        subplot_ref = grid_cell[0]
        xaxis_name = subplot_ref.layout_keys[0]
        yaxis_name = subplot_ref.layout_keys[1]
        x_domain = getattr(fig.layout, xaxis_name).domain
        y_domain = getattr(fig.layout, yaxis_name).domain
        return x_domain, y_domain

    def _add_local_legend(
        row: int,
        col: int,
        items: List[Tuple[str, str]],
        x_frac: float = 0.02,
        y_frac_from_top: float = 0.04,
    ):
        # Place inside the subplot cell domain (paper coordinates).
        x_domain, y_domain = _cell_domains_paper(row, col)
        x0, x1 = x_domain
        y0, y1 = y_domain
        x = x0 + x_frac * (x1 - x0)
        y_top = y1 - y_frac_from_top * (y1 - y0)

        # Build simple HTML with colored bullets.
        lines = []
        for label, color in items:
            lines.append(f'<span style="color:{color}; font-size:14px;">&#9679;</span> {label}')
        text = "<br>".join(lines)

        fig.add_annotation(
            x=x,
            y=y_top,
            xref='paper',
            yref='paper',
            text=text,
            showarrow=False,
            align='left',
            bgcolor='rgba(255,255,255,0.75)',
            bordercolor='rgba(0,0,0,0.12)',
            borderwidth=1,
            borderpad=6,
            font=dict(size=11, color='rgba(44,62,80,0.95)'),
        )

    # Row 1
    fig.add_trace(go.Bar(
        x=names, y=df['net_benefit'],
        marker_color=['#27ae60' if v >= 0 else '#e74c3c' for v in df['net_benefit']],
        hovertemplate='%{x}<br>Net Benefit: %{y:,.0f}<extra></extra>',
        showlegend=False,
    ), row=1, col=1)
    fig.add_trace(go.Bar(x=names, y=df['pct_from_btm'], marker_color='#27ae60', showlegend=False, hovertemplate='%{x}<br>BTM: %{y:.1f}%<extra></extra>'), row=1, col=2)
    fig.add_trace(go.Bar(x=names, y=df['pct_from_peak'], marker_color='#f39c12', showlegend=False, hovertemplate='%{x}<br>Peak: %{y:.1f}%<extra></extra>'), row=1, col=2)
    fig.add_trace(go.Bar(x=names, y=df['pct_from_da'], marker_color='#3498db', showlegend=False, hovertemplate='%{x}<br>DA: %{y:.1f}%<extra></extra>'), row=1, col=2)
    fig.add_trace(go.Bar(x=names, y=df['pct_from_fcr'], marker_color='#8e44ad', showlegend=False, hovertemplate='%{x}<br>FCR: %{y:.1f}%<extra></extra>'), row=1, col=2)

    # Row 2
    fig.add_trace(go.Bar(
        x=names, y=df['peak_reduction_pct'],
        marker_color='#e67e22',
        hovertemplate='%{x}<br>Peak Reduction: %{y:.1f}%<extra></extra>',
        showlegend=False,
    ), row=2, col=1)
    fig.add_trace(go.Bar(
        x=names, y=df['fcr_participation_rate'],
        marker_color='#8e44ad',
        hovertemplate='%{x}<br>FCR Participation: %{y:.1f}%<extra></extra>',
        showlegend=False,
    ), row=2, col=2)

    # Row 3
    fig.add_trace(go.Bar(x=names, y=df['btm_savings'], marker_color='#27ae60', showlegend=False, hovertemplate='%{x}<br>BTM Savings: %{y:,.0f}<extra></extra>'), row=3, col=1)
    fig.add_trace(go.Bar(x=names, y=df['peak_savings'], marker_color='#f39c12', showlegend=False, hovertemplate='%{x}<br>Peak Savings: %{y:,.0f}<extra></extra>'), row=3, col=1)
    fig.add_trace(go.Bar(x=names, y=df['day_ahead_arbitrage'], marker_color='#3498db', showlegend=False, hovertemplate='%{x}<br>DA Arbitrage: %{y:,.0f}<extra></extra>'), row=3, col=1)
    fig.add_trace(go.Bar(x=names, y=df['fcr_revenue'], marker_color='#8e44ad', showlegend=False, hovertemplate='%{x}<br>FCR Revenue: %{y:,.0f}<extra></extra>'), row=3, col=1)
    fig.add_trace(go.Bar(x=names, y=-df['degradation_cost'], marker_color='#e74c3c', showlegend=False, hovertemplate='%{x}<br>Degradation: %{y:,.0f}<extra></extra>'), row=3, col=1)

    # Subset plot: focus on baseline vs a small set of cycle-strategy scenarios
    eq_scenarios = ['Baseline', 'Three cycles', 'No limit']
    if 'scenario_name' in df.columns:
        df_eq = df[df['scenario_name'].isin(eq_scenarios)].copy()
        order_map = {n: i for i, n in enumerate(eq_scenarios)}
        df_eq['__order'] = df_eq['scenario_name'].map(order_map)
        df_eq = df_eq.sort_values('__order')
        names_eq = df_eq['scenario_name'].tolist()
    else:
        # Fallback if scenario_name isn't present
        df_eq = df.copy()
        names_eq = names

    if len(df_eq) >= 1:
        # Compute per-customer value per cycle when scenario_data is available.
        # This shows the variation across customers (sites).
        delta_t = 0.25  # hours; matches scenario_analysis.run_all_scenarios
        if scenario_data and isinstance(scenario_data, dict):
            vpc_mean = []
            vpc_lo = []
            vpc_hi = []
            cycles_mean = []
            cycles_lo = []
            cycles_hi = []

            def _compute_customer_cycles_and_vpc(payload: Dict) -> Tuple[np.ndarray, np.ndarray]:
                df_detail = payload.get('df', None)
                fin_sites = (payload.get('financials', {}) or {}).get('sites', {})
                site_configs = payload.get('site_configs', [])

                if df_detail is None or not isinstance(fin_sites, dict) or len(site_configs) == 0:
                    return np.array([]), np.array([])

                emax_map = {cfg.site_id: float(cfg.battery.E_max) for cfg in site_configs}

                # Discharge energy per site (kWh)
                if 'P_dis_BTM' in df_detail.columns and 'P_dis_FTM' in df_detail.columns:
                    grouped = df_detail.groupby('site')[['P_dis_BTM', 'P_dis_FTM']].sum()
                    discharge_kwh_by_site = grouped.sum(axis=1).to_dict()
                else:
                    return np.array([]), np.array([])

                cycles_by_site = {}
                vpc_by_site = {}
                for sid, fin in fin_sites.items():
                    e_max = emax_map.get(sid, None)
                    if e_max is None or e_max <= 0:
                        continue
                    discharge_kwh = float(discharge_kwh_by_site.get(sid, 0.0))
                    cycles = discharge_kwh / max(e_max, 1e-12)
                    if cycles <= 0:
                        continue
                    cycles_by_site[sid] = cycles
                    vpc_by_site[sid] = float(fin.get('net_benefit', 0.0)) / cycles

                if len(cycles_by_site) == 0 or len(vpc_by_site) == 0:
                    return np.array([]), np.array([])

                # Use shared keys to ensure aligned arrays
                common_sites = sorted(set(cycles_by_site.keys()).intersection(vpc_by_site.keys()))
                cycles_vals = np.array([cycles_by_site[s] for s in common_sites], dtype=float)
                vpc_vals = np.array([vpc_by_site[s] for s in common_sites], dtype=float)
                return cycles_vals, vpc_vals

            for _, row in df_eq.iterrows():
                sid = row['scenario_id']
                payload = scenario_data.get(sid, None) if isinstance(scenario_data, dict) else None
                if payload is None:
                    # Fallback to portfolio-level metric if payload is missing
                    cycles_vals = np.array([float(row.get('equivalent_cycles', 0.0))], dtype=float)
                    vpc_vals = np.array([float(row.get('value_per_cycle', 0.0))], dtype=float)
                else:
                    cycles_vals, vpc_vals = _compute_customer_cycles_and_vpc(payload)
                    if cycles_vals.size == 0 or vpc_vals.size == 0:
                        cycles_vals = np.array([float(row.get('equivalent_cycles', 0.0))], dtype=float)
                        vpc_vals = np.array([float(row.get('value_per_cycle', 0.0))], dtype=float)

                # Use robust uncertainty band: 10th–90th percentiles
                vpc_mean.append(float(np.mean(vpc_vals)))
                vpc_lo.append(float(np.quantile(vpc_vals, 0.10)))
                vpc_hi.append(float(np.quantile(vpc_vals, 0.90)))
                cycles_mean.append(float(np.mean(cycles_vals)))
                cycles_lo.append(float(np.quantile(cycles_vals, 0.10)))
                cycles_hi.append(float(np.quantile(cycles_vals, 0.90)))

            fig.add_trace(go.Bar(
                x=names_eq,
                y=cycles_mean,
                marker_color='#2980b9',
                showlegend=False,
                error_y=dict(
                    type='data',
                    symmetric=False,
                    array=[hi - m for hi, m in zip(cycles_hi, cycles_mean)],
                    arrayminus=[m - lo for lo, m in zip(cycles_lo, cycles_mean)],
                    visible=True,
                ),
                hovertemplate='%{x}<br>Cycles (avg): %{y:.2f}<extra></extra>',
            ), row=3, col=2)

            fig.add_trace(go.Scatter(
                x=names_eq,
                y=vpc_mean,
                mode='lines+markers',
                line=dict(color='#16a085', width=2),
                marker=dict(size=7),
                error_y=dict(
                    type='data',
                    symmetric=False,
                    array=[hi - m for hi, m in zip(vpc_hi, vpc_mean)],
                    arrayminus=[m - lo for lo, m in zip(vpc_lo, vpc_mean)],
                    visible=True,
                ),
                hovertemplate='%{x}<br>Value / Cycle (avg): %{y:,.0f}<extra></extra>',
                showlegend=False,
            ), row=3, col=2, secondary_y=True)
        else:
            # Fallback if scenario_data isn't provided
            fig.add_trace(go.Bar(
                x=names_eq,
                y=df_eq['equivalent_cycles'],
                marker_color='#2980b9',
                showlegend=False,
                hovertemplate='%{x}<br>Cycles: %{y:.2f}<extra></extra>',
            ), row=3, col=2)
            fig.add_trace(go.Scatter(
                x=names_eq,
                y=df_eq['value_per_cycle'],
                mode='lines+markers',
                line=dict(color='#16a085', width=2),
                marker=dict(size=7),
                hovertemplate='%{x}<br>Value / Cycle: %{y:,.0f}<extra></extra>',
                showlegend=False,
            ), row=3, col=2, secondary_y=True)

    # Row 4
    fig.add_trace(go.Bar(x=names, y=df['avg_soc_btm'], marker_color='#2ecc71', showlegend=False, hovertemplate='%{x}<br>Avg BTM SOC: %{y:.1f}%<extra></extra>'), row=4, col=1)
    fig.add_trace(go.Bar(x=names, y=df['avg_soc_ftm'], marker_color='#9b59b6', showlegend=False, hovertemplate='%{x}<br>Avg FTM SOC: %{y:.1f}%<extra></extra>'), row=4, col=1)
    fig.add_trace(go.Scatter(
        x=names, y=df['avg_power_utilization'],
        mode='lines+markers',
        line=dict(color='#34495e', width=2),
        marker=dict(size=7),
        hovertemplate='%{x}<br>Power Utilization: %{y:.1f}%<extra></extra>',
        showlegend=False,
    ), row=4, col=1, secondary_y=True)
    fig.add_trace(go.Bar(
        x=names, y=df['degradation_cost'],
        marker_color='#e74c3c',
        hovertemplate='%{x}<br>Degradation Cost: %{y:,.0f}<extra></extra>',
        showlegend=False,
    ), row=4, col=2)

    # ==========================================================================
    # Row 5: Per-customer plot + BTM ratio comparison
    # ==========================================================================
    def _count_unique_months(time_index) -> int:
        if time_index is None:
            return 1
        try:
            if len(time_index) == 0:
                return 1
        except Exception:
            # Some iterables (e.g., scalars) might not define len().
            pass
        try:
            t = pd.to_datetime(list(time_index))
            return max(int(t.to_period('M').nunique()), 1)
        except Exception:
            return 1

    # --- 5A) Per customer plot (left): best vs worst (semi-transparent, grouped so they don't stack) ---
    if scenario_data and isinstance(scenario_data, dict) and len(scenario_data) > 0:
        try:
            best_idx = df['net_benefit'].idxmax()
            worst_idx = df['net_benefit'].idxmin()

            best_scenario_id = df.loc[best_idx, 'scenario_id']
            worst_scenario_id = df.loc[worst_idx, 'scenario_id']

            best_payload = scenario_data.get(best_scenario_id, None)
            worst_payload = scenario_data.get(worst_scenario_id, None)

            if best_payload and isinstance(best_payload, dict):
                best_fin_sites = best_payload.get('financials', {}).get('sites', {})
                site_ids = sorted(best_fin_sites.keys())
                best_months = _count_unique_months(best_payload.get('data', {}).get('time_index', []))

                best_net_benefit_month = [
                    float(best_fin_sites[sid].get('net_benefit', 0.0)) / best_months
                    for sid in site_ids
                ]

                fig.add_trace(go.Bar(
                    x=site_ids,
                    y=best_net_benefit_month,
                    name='Best scenario',
                    marker_color='#27ae60',
                    opacity=0.55,
                    width=0.8,
                    base=0,
                    offset=0,
                    hovertemplate='Customer: %{x}<br>Best profit: %{y:,.2f} €/month<extra></extra>',
                    showlegend=False,
                ), row=5, col=1)

            if worst_payload and isinstance(worst_payload, dict):
                worst_fin_sites = worst_payload.get('financials', {}).get('sites', {})
                worst_months = _count_unique_months(worst_payload.get('data', {}).get('time_index', []))

                # Align with the best scenario customer list
                if 'site_ids' not in locals():
                    site_ids = sorted(worst_fin_sites.keys())

                worst_net_benefit_month = [
                    float(worst_fin_sites.get(sid, {}).get('net_benefit', 0.0)) / worst_months
                    for sid in site_ids
                ]

                fig.add_trace(go.Bar(
                    x=site_ids,
                    y=worst_net_benefit_month,
                    name='Worst scenario',
                    marker_color='#e74c3c',
                    opacity=0.55,
                    width=0.8,
                    base=0,
                    offset=0,
                    hovertemplate='Customer: %{x}<br>Worst profit: %{y:,.2f} €/month<extra></extra>',
                    showlegend=False,
                ), row=5, col=1)
        except Exception as e:
            print(f"Warning: could not build per-customer plot: {e}")

    # --- 5B) Comparison of the 3 scenarios (right): BTM 20/50/80 ---
    # We filter to "base" settings when possible, to ensure it is mostly only BTM-ratio varying.
    df_feas = df.copy()
    for col in ['enable_fcr', 'use_forecast_prices', 'degradation_label', 'daily_cycle_limit']:
        if col not in df_feas.columns:
            df_feas[col] = None

    candidates = df_feas
    try:
        candidates = candidates[
            (candidates['enable_fcr'] == True)
            & (candidates['use_forecast_prices'] == False)
            & (candidates['degradation_label'] == 'base')
            & (candidates['daily_cycle_limit'] == 2.0)
        ]
    except Exception:
        candidates = df_feas
    if len(candidates) < 3:
        candidates = df_feas

    def _pick_closest_btm(target: float):
        if len(candidates) == 0:
            return None
        idx = (candidates['btm_ratio'] - target).abs().idxmin()
        return candidates.loc[idx] if idx in candidates.index else None

    chosen_rows = []
    chosen_ids = set()
    for t in target_btm:
        r = _pick_closest_btm(t)
        if r is not None:
            sid = r['scenario_id']
            if sid not in chosen_ids:
                chosen_rows.append(r)
                chosen_ids.add(sid)

    chosen_df = pd.DataFrame(chosen_rows)
    labels = [f"BTM {r['btm_ratio']*100:.0f}%" for _, r in chosen_df.iterrows()]
    load_shifting = chosen_df['btm_savings'].to_numpy() if 'btm_savings' in chosen_df.columns else np.zeros(len(labels))
    peak_shaving = chosen_df['peak_savings'].to_numpy() if 'peak_savings' in chosen_df.columns else np.zeros(len(labels))
    da_arbitrage = chosen_df['day_ahead_arbitrage'].to_numpy() if 'day_ahead_arbitrage' in chosen_df.columns else np.zeros(len(labels))
    fcr_revenue = chosen_df['fcr_revenue'].to_numpy() if 'fcr_revenue' in chosen_df.columns else np.zeros(len(labels))
    degradation_cost = chosen_df['degradation_cost'].to_numpy() if 'degradation_cost' in chosen_df.columns else np.zeros(len(labels))

    fig.add_trace(go.Bar(
        x=labels, y=load_shifting,
        name='Load Shifting',
        marker_color='#27ae60',
        hovertemplate='%{x}<br>Load Shifting: %{y:,.0f}<extra></extra>',
        showlegend=False,
    ), row=5, col=2)
    fig.add_trace(go.Bar(
        x=labels, y=peak_shaving,
        name='Peak Shaving',
        marker_color='#f39c12',
        hovertemplate='%{x}<br>Peak Shaving: %{y:,.0f}<extra></extra>',
        showlegend=False,
    ), row=5, col=2)
    fig.add_trace(go.Bar(
        x=labels, y=da_arbitrage,
        name='DA Arbitrage',
        marker_color='#3498db',
        hovertemplate='%{x}<br>DA Arbitrage: %{y:,.0f}<extra></extra>',
        showlegend=False,
    ), row=5, col=2)
    fig.add_trace(go.Bar(
        x=labels, y=fcr_revenue,
        name='FCR Revenue',
        marker_color='#8e44ad',
        hovertemplate='%{x}<br>FCR Revenue: %{y:,.0f}<extra></extra>',
        showlegend=False,
    ), row=5, col=2)
    fig.add_trace(go.Bar(
        x=labels, y=-degradation_cost,
        name='Degradation Cost',
        marker_color='#e74c3c',
        hovertemplate='%{x}<br>Degradation Cost: %{y:,.0f}<extra></extra>',
        showlegend=False,
    ), row=5, col=2)

    # Ensure subplot title matches the ACTUALLY chosen BTM ratios (may differ from targets
    # if the strict candidate filter leaves fewer than 3 scenarios).
    try:
        actual_btm_title = "/".join([str(int(round(r * 100))) for r in chosen_df['btm_ratio'].tolist()])
        for ann in fig.layout.annotations or []:
            if isinstance(ann.text, str) and 'BTM Ratio Comparison' in ann.text:
                ann.text = f'<b>BTM Ratio Comparison ({actual_btm_title})</b>'
    except Exception:
        # If title override fails, keep the originally set subplot title.
        pass

    # Local legends (one per plot cell), positioned inside each subplot's domain
    # to avoid overlapping other cells.
    _add_local_legend(
        row=1, col=1,
        items=[
            ('Positive Net Benefit', '#27ae60'),
            ('Negative Net Benefit', '#e74c3c'),
        ],
    )
    _add_local_legend(
        row=1, col=2,
        items=[
            ('Load Shifting', '#27ae60'),
            ('Peak Shaving', '#f39c12'),
            ('DA Arbitrage', '#3498db'),
            ('FCR Revenue', '#8e44ad'),
        ],
    )
    _add_local_legend(
        row=2, col=1,
        items=[
            ('Peak Reduction', '#e67e22'),
        ],
    )
    _add_local_legend(
        row=2, col=2,
        items=[
            ('FCR Participation', '#8e44ad'),
        ],
    )
    _add_local_legend(
        row=3, col=1,
        items=[
            ('Load Shifting', '#27ae60'),
            ('Peak Shaving', '#f39c12'),
            ('DA Arbitrage', '#3498db'),
            ('FCR Revenue', '#8e44ad'),
            ('Degradation Cost', '#e74c3c'),
        ],
    )
    _add_local_legend(
        row=3, col=2,
        items=[
            ('Equivalent Cycles', '#2980b9'),
            ('Value / Cycle', '#16a085'),
        ],
    )
    _add_local_legend(
        row=4, col=1,
        items=[
            ('Avg BTM SOC', '#2ecc71'),
            ('Avg FTM SOC', '#9b59b6'),
            ('Power Utilization', '#34495e'),
        ],
    )
    _add_local_legend(
        row=4, col=2,
        items=[
            ('Degradation Cost', '#e74c3c'),
        ],
    )
    _add_local_legend(
        row=5, col=1,
        items=[
            ('Best scenario', '#27ae60'),
            ('Worst scenario', '#e74c3c'),
        ],
    )
    _add_local_legend(
        row=5, col=2,
        items=[
            ('Load Shifting', '#27ae60'),
            ('Peak Shaving', '#f39c12'),
            ('DA Arbitrage', '#3498db'),
            ('FCR Revenue', '#8e44ad'),
            ('Degradation Cost', '#e74c3c'),
        ],
    )

    # ==========================================================================
    # Row 6: Breakdown (two stacked bars per comparison)
    # ==========================================================================
    def _format_scenario_attributes_hover(row: pd.Series) -> str:
        scenario_name = str(row.get('scenario_name', row.get('scenario_id', 'Scenario')))
        btm_pct = float(row.get('btm_ratio', 0.0)) * 100.0
        fcr_on = bool(row.get('enable_fcr', True))
        forecast_on = bool(row.get('use_forecast_prices', False))
        daily_cycle_limit = row.get('daily_cycle_limit', None)
        degradation_label = str(row.get('degradation_label', 'base'))

        fcr_txt = 'On' if fcr_on else 'Off'
        forecast_txt = 'On' if forecast_on else 'Off'
        if daily_cycle_limit is None or (isinstance(daily_cycle_limit, float) and np.isnan(daily_cycle_limit)):
            cycles_txt = 'No limit'
        else:
            cycles_txt = f"{float(daily_cycle_limit):g}"

        return (
            f"BTM {btm_pct:.0f}% | FCR {fcr_txt} | Forecast {forecast_txt} | Cycles/day {cycles_txt} | Deg {degradation_label}"
        )

    def _get_row_by_scenario_name(df_in: pd.DataFrame, scenario_name: str) -> Optional[pd.Series]:
        if 'scenario_name' not in df_in.columns:
            return None
        matches = df_in[df_in['scenario_name'] == scenario_name]
        if len(matches) == 0:
            return None
        return matches.iloc[0]

    def _component_stack_values(row: pd.Series) -> Dict[str, float]:
        # Note: request uses "Load Shifting" instead of "BTM savings".
        load_shifting = float(row.get('btm_savings', 0.0))
        peak_shaving = float(row.get('peak_savings', 0.0))
        da_arbitrage = float(row.get('day_ahead_arbitrage', 0.0))
        fcr_revenue = float(row.get('fcr_revenue', 0.0))
        degradation_cost = float(row.get('degradation_cost', 0.0))  # positive cost
        return {
            'Load Shifting': load_shifting,
            'Peak Shaving': peak_shaving,
            'DA Arbitrage': da_arbitrage,
            'FCR Revenue': fcr_revenue,
            'Degradation Cost': -degradation_cost,  # plot as negative
        }

    def _add_two_scenario_stacked_breakdown(
        row_col: Tuple[int, int],
        scenario_left: str,
        scenario_right: str,
        colors: Dict[str, str],
    ):
        r, c = row_col
        left_row = _get_row_by_scenario_name(df, scenario_left)
        right_row = _get_row_by_scenario_name(df, scenario_right)
        if left_row is None or right_row is None:
            return

        x_scenarios = [scenario_left, scenario_right]
        attrs = [_format_scenario_attributes_hover(left_row), _format_scenario_attributes_hover(right_row)]

        stack = {
            'Load Shifting': ('Load Shifting', colors['Load Shifting'], lambda v: v),
            'Peak Shaving': ('Peak Shaving', colors['Peak Shaving'], lambda v: v),
            'DA Arbitrage': ('DA Arbitrage', colors['DA Arbitrage'], lambda v: v),
            'FCR Revenue': ('FCR Revenue', colors['FCR Revenue'], lambda v: v),
            'Degradation Cost': ('Degradation Cost', colors['Degradation Cost'], lambda v: v),
        }

        # Plot one trace per component so Plotly stacks them within the subplot.
        for key, (label, color, _) in stack.items():
            left_val = _component_stack_values(left_row)[key]
            right_val = _component_stack_values(right_row)[key]

            # For degradation (negative in plot), show positive cost in hover.
            left_abs_for_hover = abs(left_val)
            right_abs_for_hover = abs(right_val)

            customdata = [
                [left_abs_for_hover, attrs[0]],
                [right_abs_for_hover, attrs[1]],
            ]

            fig.add_trace(
                go.Bar(
                    x=x_scenarios,
                    y=[left_val, right_val],
                    customdata=customdata,
                    marker_color=color,
                    name=label,
                    showlegend=False,
                    hovertemplate=f"%{{x}}<br>{label}: %{{customdata[0]:,.0f}} €<br>%{{customdata[1]}}<extra></extra>",
                ),
                row=r,
                col=c,
            )

    stack_colors = {
        'Load Shifting': '#27ae60',
        'Peak Shaving': '#f39c12',
        'DA Arbitrage': '#3498db',
        'FCR Revenue': '#8e44ad',
        'Degradation Cost': '#e74c3c',
    }

    _add_two_scenario_stacked_breakdown(
        row_col=(6, 1),
        scenario_left='Baseline',
        scenario_right='Uncertain prices',
        colors=stack_colors,
    )
    _add_two_scenario_stacked_breakdown(
        row_col=(6, 2),
        scenario_left='Baseline',
        scenario_right='No FCR',
        colors=stack_colors,
    )

    fig.update_layout(
        title=dict(text='<b>Scenario Overview Dashboard</b>', font=dict(size=18)),
        height=3400,
        template='plotly_white',
        barmode='relative',
        showlegend=False,
        margin=dict(t=90, b=70, l=60, r=60),
    )

    fig.update_yaxes(title_text='€', row=1, col=1)
    fig.update_yaxes(title_text='Attribution (%)', row=1, col=2)
    fig.update_yaxes(title_text='Peak Reduction (%)', row=2, col=1)
    fig.update_yaxes(title_text='FCR Participation (%)', row=2, col=2)
    fig.update_yaxes(title_text='€', row=3, col=1)
    fig.update_yaxes(title_text='Cycles', row=3, col=2)
    fig.update_yaxes(title_text='€/Cycle', row=3, col=2, secondary_y=True)
    fig.update_yaxes(title_text='SOC (%)', row=4, col=1)
    fig.update_yaxes(title_text='Utilization (%)', row=4, col=1, secondary_y=True)
    fig.update_yaxes(title_text='€', row=4, col=2)
    fig.update_yaxes(title_text='€/month', row=5, col=1)
    fig.update_yaxes(title_text='€', row=5, col=2)
    fig.update_yaxes(title_text='€', row=6, col=1)
    fig.update_yaxes(title_text='€', row=6, col=2)
    fig.update_xaxes(tickangle=-20)
    fig.update_xaxes(tickangle=-90, row=5, col=1)

    if output_file:
        fig.write_html(str(output_file), include_plotlyjs='cdn')
        print(f"Overview dashboard saved to {output_file}")

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
        pct_da = row.get('pct_from_da', 0.0)
        if row['pct_from_peak'] > max(row['pct_from_btm'], row['pct_from_fcr'], pct_da):
            strategy = 'Peak'
        elif row['pct_from_fcr'] > max(row['pct_from_btm'], row['pct_from_peak'], pct_da):
            strategy = 'FCR'
        elif pct_da > max(row['pct_from_btm'], row['pct_from_peak'], row['pct_from_fcr']):
            strategy = 'DA'
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
    START_DATE = "2024-06-01"
    END_DATE   = "2024-08-31"   # 1-week test; change to e.g. "2024-01-31" or "2024-12-31"

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
                V_bat=0.777,
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

        fig_overview = create_scenario_overview_dashboard(
            results_df,
            output_file=OUTPUTS_DIR / "scenario_overview_dashboard.html",
            scenario_data=scenario_data,
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
        print("  outputs/scenario_overview_dashboard.html - Merged comparison + insights")
        print("  outputs/scenario_outputs/               - Individual scenario dashboards")
    else:
        print("\nDashboard generation skipped (GENERATE_DASHBOARDS = False)")

