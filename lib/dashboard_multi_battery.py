"""
Multi-Battery VPP Dashboard - Enhanced Version
=================================================
Interactive Plotly dashboard with:
- Proper subplot layout with aligned axes
- Action reasoning: WHY is the optimizer doing what it's doing?
- Clear BTM vs FTM visualization
- Economic drivers visible at each timestep
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, List, Tuple
from pathlib import Path

# Color palette - professional energy dashboard
COLORS = {
    'demand': '#7f8c8d',           # Gray for baseline demand
    'grid_import': '#3498db',      # Blue for grid import
    'grid_export': '#e74c3c',      # Red for grid export
    'btm_charge': '#e67e22',       # Orange for BTM charging
    'btm_discharge': '#27ae60',    # Green for BTM discharge
    'ftm_charge': '#9b59b6',       # Purple for FTM charging  
    'ftm_discharge': '#1abc9c',    # Teal for FTM discharge
    'fcr': '#2563eb',              # Royal Blue for FCR
    'price_high': '#c0392b',       # Dark red for high prices
    'price_low': '#16a085',        # Teal for low prices
    'soc': '#2ecc71',              # Bright green for SOC
    'peak': '#e74c3c',             # Red for peak
    'savings': '#27ae60',          # Green for savings
}

# =============================================================================
# ACTION CLASSIFICATION
# =============================================================================

def classify_actions(df: pd.DataFrame, price_threshold_high: float = 0.08, 
                     price_threshold_low: float = 0.02) -> pd.DataFrame:
    """
    Classify what action the optimizer is taking and WHY at each timestep.
    
    Returns DataFrame with action labels for BTM and FTM partitions.
    """
    df = df.copy()
    
    # Calculate net flows
    df['BTM_net'] = df['P_ch_BTM'] - df['P_dis_BTM']
    df['FTM_net'] = df['P_ch_FTM'] - df['P_dis_FTM']
    df['total_charge'] = df['P_ch_BTM'] + df['P_ch_FTM']
    df['total_discharge'] = df['P_dis_BTM'] + df['P_dis_FTM']
    
    # Identify price conditions
    df['price_high'] = df['Price'] > price_threshold_high
    df['price_low'] = df['Price'] < price_threshold_low
    df['price_negative'] = df['Price'] < 0
    
    # --- BTM Action Classification ---
    def classify_btm(row):
        reasons = []
        
        # Check if BTM is active
        if abs(row['BTM_net']) < 0.1:
            return 'Idle', ''
        
        if row['BTM_net'] > 0:  # Charging
            if row['price_negative']:
                reasons.append('Negative price arbitrage')
            elif row['price_low']:
                reasons.append('Low price charging')
            if row['demand'] < row['P_dis_BTM'] + row['P_dis_FTM']:
                reasons.append('Excess capacity storage')
            action = 'Charging'
        else:  # Discharging
            if row['price_high']:
                reasons.append('High price discharge')
            if row['Grid_Net'] > 0 and row['demand'] > 50:
                reasons.append('Load shifting')
            if row.get('is_peak_hour', False):
                reasons.append('Peak shaving')
            action = 'Discharging'
        
        return action, '; '.join(reasons) if reasons else 'Optimization'
    
    # --- FTM Action Classification ---
    def classify_ftm(row):
        reasons = []
        
        # Check FCR participation
        if row['P_FCR_bid'] > 0:
            if abs(row['FCR_signal']) > 0:
                if row['FCR_signal'] > 0:
                    return 'FCR Activation (Charge)', 'Grid frequency support - absorbing power'
                else:
                    return 'FCR Activation (Discharge)', 'Grid frequency support - injecting power'
            else:
                return 'FCR Reserved', f"Capacity reserved: {row['P_FCR_bid']:.0f} kW"
        
        # Check arbitrage
        if abs(row['FTM_net']) < 0.1:
            return 'Idle', ''
        
        if row['FTM_net'] > 0:  # Charging
            if row['price_negative']:
                reasons.append('Negative price - paid to consume')
            elif row['price_low']:
                reasons.append('Low price arbitrage buy')
            action = 'Arbitrage Charge'
        else:  # Discharging
            if row['price_high']:
                reasons.append('High price arbitrage sell')
            if row['P_sell'] > 0:
                reasons.append('Grid export')
            action = 'Arbitrage Discharge'
        
        return action, '; '.join(reasons) if reasons else 'DA Arbitrage'
    
    # Apply classifications
    btm_results = df.apply(classify_btm, axis=1)
    df['BTM_action'] = [r[0] for r in btm_results]
    df['BTM_reason'] = [r[1] for r in btm_results]
    
    ftm_results = df.apply(classify_ftm, axis=1)
    df['FTM_action'] = [r[0] for r in ftm_results]
    df['FTM_reason'] = [r[1] for r in ftm_results]
    
    # Combined action summary
    def combined_action(row):
        actions = []
        if 'Discharg' in row['BTM_action']:
            actions.append('BTM↓')
        elif 'Charg' in row['BTM_action']:
            actions.append('BTM↑')
        
        if 'FCR' in row['FTM_action']:
            actions.append('FCR')
        elif 'Discharg' in row['FTM_action'] or row['FTM_action'] == 'Arbitrage Discharge':
            actions.append('FTM↓')
        elif 'Charg' in row['FTM_action'] or row['FTM_action'] == 'Arbitrage Charge':
            actions.append('FTM↑')
        
        return ' + '.join(actions) if actions else 'Idle'
    
    df['action_summary'] = df.apply(combined_action, axis=1)
    
    return df


# =============================================================================
# MAIN PORTFOLIO DASHBOARD (SUBPLOTS VERSION)
# =============================================================================

def create_multi_battery_dashboard(
    df: pd.DataFrame,
    financials: Dict,
    site_configs: List,
    data: Dict,
    output_file: Path = None,
    delta_t: float = 0.25,
    C_peak: float = None,
    reconcile_debug: bool = False,
) -> go.Figure:
    """
    Create a clean single-page portfolio dashboard with dedicated sections:
    BTM operations, FTM/FCR operations, and financial/degradation summaries.
    """

    site_ids = df['site'].unique().tolist()

    # Aggregate data across all sites
    agg_df = df.groupby('t').agg({
        'datetime': 'first',
        'demand': 'sum',
        'P_buy': 'sum',
        'P_buy_BTM': 'sum',
        'P_buy_FTM': 'sum',
        'P_sell': 'sum',
        'Grid_Net': 'sum',
        'P_ch_BTM': 'sum',
        'P_dis_BTM': 'sum',
        'P_ch_FTM': 'sum',
        'P_dis_FTM': 'sum',
        'P_FCR_bid': 'sum',
        'SOC_BTM': 'sum',
        'SOC_FTM': 'sum',
        'SOC_total': 'sum',
        'Price': 'first',
        'Price industrial': 'first',
        'FCR_price': 'first',
        'FCR_total': 'first',
        'u_FCR': 'first',
        'FCR_signal': 'first',
        'SOH': 'mean',
    }).reset_index()

    total_E_max = sum(cfg.battery.E_max for cfg in site_configs)
    total_E_btm = sum(cfg.battery.E_max * cfg.btm_ratio for cfg in site_configs)
    total_E_ftm = sum(cfg.battery.E_max * (1 - cfg.btm_ratio) for cfg in site_configs)

    agg_df['SOC_pct'] = 100 * agg_df['SOC_total'] / max(total_E_max, 1e-9)
    agg_df['SOC_BTM_pct'] = np.where(
        total_E_btm > 1e-9, 100 * agg_df['SOC_BTM'] / total_E_btm, 0.0
    )
    agg_df['SOC_FTM_pct'] = np.where(
        total_E_ftm > 1e-9, 100 * agg_df['SOC_FTM'] / total_E_ftm, 0.0
    )

    # Classify actions
    agg_df = classify_actions(agg_df)

    # Calculate running peak
    # Portfolio running peak = running maximum of the portfolio net grid import (P_buy).
    agg_df['running_peak'] = agg_df['P_buy'].expanding().max()
    agg_df['running_peak_btm'] = agg_df['P_buy_BTM'].expanding().max()

    # FCR: reserved vs activated average power in each step
    agg_df['fcr_activation_fraction'] = agg_df['FCR_signal'] / 3600.0
    agg_df['FCR_activated_power'] = agg_df['FCR_total'] * agg_df['fcr_activation_fraction']
    agg_df['FCR_activation_ratio'] = np.where(
        agg_df['FCR_total'] > 1e-6,
        100 * np.abs(agg_df['FCR_activated_power']) / agg_df['FCR_total'],
        0.0,
    )

    # --- Exact per-step realized cashflows (consistent with calculate_financials) ---
    df_cash = df.copy()
    df_cash['datetime_idx'] = pd.to_datetime(df_cash['datetime'])
    df_cash['cash_btm'] = (df_cash['demand'] - df_cash['P_buy_BTM']) * df_cash['Price industrial'] * delta_t
    df_cash['cash_da'] = (df_cash['P_sell'] - df_cash['P_buy_FTM']) * df_cash['Price'] * delta_t
    df_cash['cash_fcr'] = df_cash['FCR_price'] * df_cash['P_FCR_bid']

    cash_by_t = df_cash.groupby('t', as_index=False)[['cash_btm', 'cash_da', 'cash_fcr']].sum()
    agg_df = agg_df.merge(cash_by_t, on='t', how='left')
    agg_df[['cash_btm', 'cash_da', 'cash_fcr']] = agg_df[['cash_btm', 'cash_da', 'cash_fcr']].fillna(0.0)

    # SOH and cumulative degradation cost trajectory (per-site exact mapping, then portfolio sum)
    site_i0_map = {cfg.site_id: cfg.battery.I0 for cfg in site_configs}
    df_cash['I0_site'] = df_cash['site'].map(site_i0_map).astype(float)
    soh_start_map = df_cash.sort_values(['site', 't']).groupby('site')['SOH'].first().to_dict()
    df_cash['SOH0_site'] = df_cash['site'].map(soh_start_map).astype(float)
    soh_denom = (df_cash['SOH0_site'] - 0.6).replace(0, np.nan)
    df_cash['deg_cost_to_t'] = df_cash['I0_site'] * (df_cash['SOH0_site'] - df_cash['SOH']) / soh_denom
    df_cash['deg_cost_to_t'] = df_cash['deg_cost_to_t'].replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(lower=0.0)
    deg_by_t = df_cash.groupby('t', as_index=False)['deg_cost_to_t'].sum()
    agg_df = agg_df.merge(deg_by_t, on='t', how='left')
    agg_df['deg_cost_to_t'] = agg_df['deg_cost_to_t'].fillna(0.0)

    # Causal monthly series
    agg_df['datetime_idx'] = pd.to_datetime(agg_df['datetime'])
    monthly_energy = agg_df.set_index('datetime_idx')[['cash_btm', 'cash_da', 'cash_fcr']].resample('ME').sum()
    monthly_energy = monthly_energy.rename(columns={
        'cash_btm': 'val_load_shifting',
        'cash_da': 'val_da_arb',
        'cash_fcr': 'val_fcr',
    })

    # Peak savings monthly allocation by actual peak event timing per site
    c_peak_effective = C_peak
    if c_peak_effective is None:
        c_peak_candidates = []
        for site_id, site_fin in financials.get('sites', {}).items():
            peak_kw_old = float(site_fin.get('peak_kw_old', 0.0))
            baseline_peak = float(site_fin.get('baseline_peak', 0.0))
            if peak_kw_old > 0:
                c_peak_candidates.append(baseline_peak / peak_kw_old)
        c_peak_effective = float(np.mean(c_peak_candidates)) if c_peak_candidates else 0.0

    monthly_baseline_peak = {}
    monthly_optimized_peak = {}
    for site_id, site_df in df_cash.groupby('site'):
        site_df_sorted = site_df.sort_values('t')
        if site_df_sorted.empty:
            continue
        idx_old = site_df_sorted['demand'].idxmax()
        month_old = site_df_sorted.loc[idx_old, 'datetime_idx'].to_period('M').to_timestamp('M')
        baseline_peak_cost = site_df_sorted.loc[idx_old, 'demand'] * c_peak_effective

        p_peak_new = float(site_df_sorted['P_peak'].iloc[0])
        idx_new = (site_df_sorted['P_buy_BTM'] + site_df_sorted['P_buy_FTM']).idxmax()
        month_new = site_df_sorted.loc[idx_new, 'datetime_idx'].to_period('M').to_timestamp('M')
        peak_cost_new = p_peak_new * c_peak_effective

        monthly_baseline_peak[month_old] = monthly_baseline_peak.get(month_old, 0.0) + baseline_peak_cost
        monthly_optimized_peak[month_new] = monthly_optimized_peak.get(month_new, 0.0) + peak_cost_new

    monthly_peak_df = pd.DataFrame(index=monthly_energy.index)
    monthly_peak_df['baseline_peak_cost'] = [monthly_baseline_peak.get(m, 0.0) for m in monthly_peak_df.index]
    monthly_peak_df['optimized_peak_cost'] = [monthly_optimized_peak.get(m, 0.0) for m in monthly_peak_df.index]
    monthly_peak_df['val_peak_savings'] = monthly_peak_df['baseline_peak_cost'] - monthly_peak_df['optimized_peak_cost']

    monthly_df = pd.concat([monthly_energy, monthly_peak_df[['val_peak_savings']]], axis=1).fillna(0.0)
    monthly_df['month_name'] = monthly_df.index.strftime('%B %Y')

    # Optional strict reconciliation checks
    if reconcile_debug:
        port = financials['portfolio']
        checks = [
            ('btm_savings', monthly_df['val_load_shifting'].sum(), port.get('btm_savings', 0.0)),
            ('day_ahead_arbitrage', monthly_df['val_da_arb'].sum(), port.get('day_ahead_arbitrage', 0.0)),
            ('peak_savings', monthly_df['val_peak_savings'].sum(), port.get('peak_savings', 0.0)),
            ('fcr_revenue', monthly_df['val_fcr'].sum(), port.get('fcr_revenue', 0.0)),
            ('degradation_cost', agg_df['deg_cost_to_t'].iloc[-1] if len(agg_df) else 0.0, port.get('degradation_cost', 0.0)),
        ]
        for name, observed, expected in checks:
            tol = max(1.0, 0.01 * abs(expected))
            if abs(observed - expected) > tol:
                raise AssertionError(
                    f"Reconciliation failed for {name}: observed={observed:.4f}, expected={expected:.4f}, tol={tol:.4f}"
                )

        net_observed = (
            monthly_df['val_load_shifting'].sum()
            + monthly_df['val_da_arb'].sum()
            + monthly_df['val_peak_savings'].sum()
            + monthly_df['val_fcr'].sum()
            - (agg_df['deg_cost_to_t'].iloc[-1] if len(agg_df) else 0.0)
        )
        net_expected = port.get('net_benefit', 0.0)
        tol = max(1.0, 0.01 * abs(net_expected))
        if abs(net_observed - net_expected) > tol:
            raise AssertionError(
                f"Reconciliation failed for net_benefit: observed={net_observed:.4f}, expected={net_expected:.4f}, tol={tol:.4f}"
            )

    # ==========================================================================
    # Secondary-axis ranges (robust, data-driven)
    # ==========================================================================
    def _robust_axis_range(values, q_low: float = 0.01, q_high: float = 0.99, pad_frac: float = 0.05,
                            include_zero: bool = False):
        """Compute a robust [min,max] for Plotly `range` using quantiles."""
        arr = np.asarray(pd.to_numeric(values, errors='coerce'), dtype=float)
        arr = arr[~np.isnan(arr)]
        if arr.size == 0:
            return [0.0, 1.0]

        lo = float(np.quantile(arr, q_low))
        hi = float(np.quantile(arr, q_high))

        if include_zero:
            lo = min(lo, 0.0)
            hi = max(hi, 0.0)

        if np.isclose(lo, hi):
            pad = abs(hi) * pad_frac if not np.isclose(hi, 0.0) else 1.0
            lo -= pad
            hi += pad
        else:
            span = hi - lo
            lo -= span * pad_frac
            hi += span * pad_frac
        return [lo, hi]

    # Prices are plotted on secondary axes; make the scale robust so outliers
    # don't flatten the curve.
    retail_range = _robust_axis_range(agg_df['Price industrial'] * 1000, include_zero=True)
    da_range = _robust_axis_range(agg_df['Price'] * 1000, include_zero=True)
    row5_price_range = _robust_axis_range(
        np.concatenate([(agg_df['Price'] * 1000).to_numpy(), (agg_df['FCR_price'] * 1000).to_numpy()]),
        include_zero=True
    )

    # Utilization / allocation on FCR utilization secondary axis.
    sec_row6_alloc_vals = np.where(agg_df['u_FCR'] > 0.5, 100.0, 0.0)
    sec_row6_activation_range = _robust_axis_range(
        np.concatenate([sec_row6_alloc_vals, agg_df['FCR_activation_ratio'].to_numpy()]),
        include_zero=True
    )

    # Degradation cost on SOH/degradation secondary axis.
    sec_row8_degradation_range = _robust_axis_range(agg_df['deg_cost_to_t'], include_zero=True)

    # ==========================================================================
    # CREATE SUBPLOT FIGURE
    # ==========================================================================

    fig = make_subplots(
        rows=7, cols=1,
        shared_xaxes=False,
        vertical_spacing=0.035,
        row_heights=[0.13, 0.14, 0.11, 0.11, 0.12, 0.18, 0.21],
        subplot_titles=(
            '<b>1. GRID EXCHANGE</b> — Import (+) / Export (-) vs Original Demand',
            '<b>2. BATTERY OPERATIONS (BTM + FTM)</b> — Charge (+) / Discharge (-), colored by domain',
            '<b>3. BTM SOC</b> — BTM SOC and retail tariff',
            '<b>4. FTM SOC & PRICES</b> — FTM SOC with DA/FCR prices',
            '<b>5. FCR UTILIZATION</b> — Reserved capacity vs activated power',
            '<b>6. MONTHLY FINANCIAL STACK</b> — Value attribution by month',
            '<b>7. SOH & DEGRADATION</b> — State of health and cumulative degradation cost',
        ),
        specs=[
            [{"secondary_y": True}],
            [{"secondary_y": True}],
            [{"secondary_y": True}],
            [{"secondary_y": True}],
            [{"secondary_y": True}],
            [{"secondary_y": False}],
            [{"secondary_y": True}],
        ],
    )

    # ==========================================================================
    # ROW 1: GRID EXCHANGE
    # ==========================================================================

    fig.add_trace(go.Scatter(
        x=agg_df['datetime'], y=agg_df['demand'],
        name='Original Demand',
        fill='tozeroy',
        line=dict(width=0),
        fillcolor='rgba(127,140,141,0.3)',
        hovertemplate='Demand: %{y:.0f} kW<extra></extra>'
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=agg_df['datetime'], y=agg_df['P_buy'],
        name='Grid Import',
        line=dict(color=COLORS['grid_import'], width=2),
        hovertemplate='Import: %{y:.0f} kW<extra></extra>'
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=agg_df['datetime'], y=-agg_df['P_sell'],
        name='Grid Export',
        line=dict(color=COLORS['grid_export'], width=2),
        fill='tozeroy',
        fillcolor='rgba(231,76,60,0.2)',
        hovertemplate='Export: %{y:.0f} kW<extra></extra>'
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=agg_df['datetime'], y=agg_df['running_peak'],
        name='Peak (Running Max)',
        line=dict(color=COLORS['peak'], width=1, dash='dot'),
        hovertemplate='Peak: %{y:.0f} kW<extra></extra>'
    ), row=1, col=1)

    fig.add_hline(y=0, line_dash='dash', line_color='black', line_width=1, row=1, col=1)

    fig.add_trace(go.Scatter(
        x=agg_df['datetime'], y=agg_df['Price'] * 1000,
        name='DA Price',
        line=dict(color='rgba(192,57,43,0.4)', width=1),
        hovertemplate='%{y:.1f} €/MWh<extra></extra>'
    ), row=1, col=1, secondary_y=True)

    # ==========================================================================
    # ROW 2: Combined BTM + FTM operations
    # ==========================================================================

    fig.add_trace(go.Bar(
        x=agg_df['datetime'], y=agg_df['P_ch_BTM'],
        name='BTM Charge',
        marker_color='rgba(230,126,34,0.85)',
        hovertemplate='BTM Charge: %{y:.0f} kW<extra></extra>'
    ), row=2, col=1)

    fig.add_trace(go.Bar(
        x=agg_df['datetime'], y=-agg_df['P_dis_BTM'],
        name='BTM Discharge',
        marker_color='rgba(230,126,34,0.65)',
        hovertemplate='BTM Discharge: %{y:.0f} kW<extra></extra>'
    ), row=2, col=1)

    fig.add_trace(go.Bar(
        x=agg_df['datetime'], y=agg_df['P_ch_FTM'],
        name='FTM Charge',
        marker_color='rgba(26,188,156,0.85)',
        hovertemplate='FTM Charge: %{y:.0f} kW<extra></extra>'
    ), row=2, col=1)

    fig.add_trace(go.Bar(
        x=agg_df['datetime'], y=-agg_df['P_dis_FTM'],
        name='FTM Discharge',
        marker_color='rgba(26,188,156,0.65)',
        hovertemplate='FTM Discharge: %{y:.0f} kW<extra></extra>'
    ), row=2, col=1)

    fig.add_trace(go.Scatter(
        x=agg_df['datetime'], y=agg_df['Price'] * 1000,
        name='DA Price',
        line=dict(color='rgba(192,57,43,0.55)', width=1),
        hovertemplate='DA Price: %{y:.1f} €/MWh<extra></extra>'
    ), row=2, col=1, secondary_y=True)

    fig.add_hline(y=0, line_dash='solid', line_color='black', line_width=1, row=2, col=1)

    # ==========================================================================
    # ROW 3: BTM SOC + retail
    # ==========================================================================

    fig.add_trace(go.Scatter(
        x=agg_df['datetime'], y=agg_df['SOC_BTM_pct'],
        name='BTM SOC',
        line=dict(color=COLORS['btm_discharge'], width=2),
        fill='tozeroy',
        fillcolor='rgba(39,174,96,0.20)',
        hovertemplate='BTM SOC: %{y:.1f}%<extra></extra>'
    ), row=3, col=1)

    fig.add_trace(go.Scatter(
        x=agg_df['datetime'], y=agg_df['Price industrial'] * 1000,
        name='Retail Price',
        line=dict(color='rgba(44,62,80,0.55)', width=1.2, dash='dot'),
        hovertemplate='Retail: %{y:.1f} €/MWh<extra></extra>'
    ), row=3, col=1, secondary_y=True)

    # ==========================================================================
    # ROW 4: FTM SOC + prices
    # ==========================================================================
    fig.add_trace(go.Scatter(
        x=agg_df['datetime'], y=agg_df['SOC_FTM_pct'],
        name='FTM SOC',
        line=dict(color=COLORS['soc'], width=2),
        fill='tozeroy',
        fillcolor='rgba(46,204,113,0.18)',
        hovertemplate='FTM SOC: %{y:.1f}%<extra></extra>'
    ), row=4, col=1)

    fig.add_trace(go.Scatter(
        x=agg_df['datetime'], y=agg_df['Price'] * 1000,
        name='DA Price',
        line=dict(color=COLORS['price_high'], width=1.2),
        hovertemplate='DA: %{y:.1f} €/MWh<extra></extra>'
    ), row=4, col=1, secondary_y=True)

    fig.add_trace(go.Scatter(
        x=agg_df['datetime'], y=agg_df['FCR_price'] * 1000,
        name='FCR Price',
        line=dict(color=COLORS['fcr'], width=1.2, dash='dash'),
        hovertemplate='FCR: %{y:.1f} €/MW<extra></extra>'
    ), row=4, col=1, secondary_y=True)

    # ==========================================================================
    # ROW 5: FCR reserved vs activated
    # ==========================================================================

    fig.add_trace(go.Scatter(
        x=agg_df['datetime'], y=agg_df['FCR_total'],
        name='FCR Reserved',
        line=dict(color=COLORS['fcr'], width=2),
        fill='tozeroy',
        fillcolor='rgba(37,99,235,0.18)',
        hovertemplate='Reserved FCR: %{y:.0f} kW<extra></extra>'
    ), row=5, col=1)

    fig.add_trace(go.Scatter(
        x=agg_df['datetime'], y=agg_df['FCR_activated_power'],
        name='Activated FCR Avg Power',
        line=dict(color=COLORS['ftm_discharge'], width=1.7),
        hovertemplate='Activated Avg Power: %{y:.1f} kW<extra></extra>'
    ), row=5, col=1)

    fig.add_trace(go.Scatter(
        x=agg_df['datetime'],
        y=np.where(agg_df['u_FCR'] > 0.5, 100, 0),
        name='FCR Allocated Flag',
        line=dict(color='rgba(52,73,94,0.85)', width=1, dash='dot'),
        hovertemplate='FCR Allocated: %{y:.0f}%<extra></extra>'
    ), row=5, col=1, secondary_y=True)

    fig.add_trace(go.Scatter(
        x=agg_df['datetime'], y=agg_df['FCR_activation_ratio'],
        name='Activation Ratio',
        line=dict(color='rgba(231,76,60,0.80)', width=1),
        hovertemplate='Activation Ratio: %{y:.1f}%<extra></extra>'
    ), row=5, col=1, secondary_y=True)

    fig.add_hline(y=1000, line_dash='dot', line_color='red', line_width=1.5, row=5, col=1,
                  annotation_text='1 MW min bid', annotation_position='right')
    fig.add_hline(y=0, line_dash='dash', line_color='black', line_width=1, row=5, col=1)

    # ==========================================================================
    # ROW 6: Monthly revenue attribution
    # ==========================================================================
    _peak_label = 'Peak Shaving (causal month attribution)'
    if C_peak is not None:
        _peak_label += f' — C_peak €{C_peak:.2f}/kW'
    fig.add_trace(go.Bar(
        x=monthly_df['month_name'], y=monthly_df['val_peak_savings'],
        name=_peak_label,
        marker_color=COLORS['peak'],
        hovertemplate='Peak Shaving: €%{y:,.2f}<extra>Attributed to month of peak event</extra>'
    ), row=6, col=1)

    fig.add_trace(go.Bar(
        x=monthly_df['month_name'], y=monthly_df['val_load_shifting'],
        name='Load Shifting (Net)',
        marker_color=COLORS['btm_charge'],
        hovertemplate='Load Shifting: €%{y:,.0f}<extra>Retail price arbitrage</extra>'
    ), row=6, col=1)

    fig.add_trace(go.Bar(
        x=monthly_df['month_name'], y=monthly_df['val_da_arb'],
        name='DA Arbitrage',
        marker_color=COLORS['ftm_charge'],
        hovertemplate='DA Arbitrage: €%{y:,.0f}<extra>Wholesale trading</extra>'
    ), row=6, col=1)

    fig.add_trace(go.Bar(
        x=monthly_df['month_name'], y=monthly_df['val_fcr'],
        name='FCR Revenue',
        marker_color=COLORS['fcr'],
        hovertemplate='FCR Revenue: €%{y:,.0f}<extra>Frequency support</extra>'
    ), row=6, col=1)

    # ==========================================================================
    # ROW 7: SOH and cumulative degradation
    # ==========================================================================
    fig.add_trace(go.Scatter(
        x=agg_df['datetime'], y=100 * agg_df['SOH'],
        name='Average SOH',
        line=dict(color='rgba(39,174,96,0.95)', width=2),
        hovertemplate='SOH: %{y:.3f}%<extra></extra>'
    ), row=7, col=1)

    fig.add_trace(go.Scatter(
        x=agg_df['datetime'], y=agg_df['deg_cost_to_t'],
        name='Cumulative Degradation Cost',
        line=dict(color='rgba(231,76,60,0.90)', width=2),
        fill='tozeroy',
        fillcolor='rgba(231,76,60,0.12)',
        hovertemplate='Cum. Degradation (model-consistent): €%{y:,.2f}<extra></extra>'
    ), row=7, col=1, secondary_y=True)

    # ==========================================================================
    # LAYOUT
    # ==========================================================================

    annotation_text = (
        "Battery operations: BTM (orange) and FTM (teal) shown together."
        "<br>SOC/market section: BTM and FTM SOC with retail/DA/FCR prices."
        "<br>Financial section: monthly value stack + SOH/degradation trend."
    )
    fig.update_layout(
        title=dict(
            text=f'<b>VPP Portfolio Dashboard</b> — {len(site_ids)} Sites, '
                 f'{total_E_max:.0f} kWh Total Capacity',
            font=dict(size=20)
        ),
        height=2050,
        template='plotly_white',
        hovermode='x unified',
        barmode='relative',
        showlegend=False,
        margin=dict(t=120, b=80, l=70, r=70),
        annotations=[
            dict(
                text=annotation_text,
                xref='paper',
                yref='paper',
                x=0.99,
                y=1.08,
                xanchor='right',
                yanchor='top',
                showarrow=False,
                align='left',
                font=dict(size=11, color='rgba(44,62,80,0.95)'),
                bgcolor='rgba(255,255,255,0.88)',
                bordercolor='rgba(52,73,94,0.25)',
                borderwidth=1,
                borderpad=6,
            )
        ],
    )

    # Axis labels
    fig.update_yaxes(title_text='Power (kW)', row=1, col=1)
    fig.update_yaxes(title_text='€/MWh', row=1, col=1, secondary_y=True)
    fig.update_yaxes(title_text='Power (kW)', row=2, col=1)
    fig.update_yaxes(title_text='DA (€/MWh)', row=2, col=1, secondary_y=True, range=da_range)
    fig.update_yaxes(title_text='SOC (%)', row=3, col=1, range=[0, 100])
    fig.update_yaxes(title_text='Retail (€/MWh)', row=3, col=1, secondary_y=True, range=retail_range)
    fig.update_yaxes(title_text='SOC (%)', row=4, col=1, range=[0, 100])
    fig.update_yaxes(title_text='Price (€/MWh or €/MW)', row=4, col=1, secondary_y=True, range=row5_price_range)
    fig.update_yaxes(title_text='Power (kW)', row=5, col=1)
    fig.update_yaxes(title_text='Activation / Allocation (%)', row=5, col=1, secondary_y=True, range=sec_row6_activation_range)
    fig.update_yaxes(title_text='Value (€)', row=6, col=1)
    fig.update_yaxes(title_text='SOH (%)', row=7, col=1, range=[60, 101])
    fig.update_yaxes(title_text='Cum. Degradation (€)', row=7, col=1, secondary_y=True, range=sec_row8_degradation_range)

    # Link time axes for rows 1-5 and row 7
    fig.update_xaxes(matches='x', row=1, col=1)
    fig.update_xaxes(matches='x', row=2, col=1)
    fig.update_xaxes(matches='x', row=3, col=1)
    fig.update_xaxes(matches='x', row=4, col=1)
    fig.update_xaxes(matches='x', row=5, col=1)
    fig.update_xaxes(matches='x', row=7, col=1)
    fig.update_xaxes(title_text='Month', row=6, col=1)
    fig.update_xaxes(title_text='Time', row=7, col=1)
    fig.update_xaxes(rangeslider=dict(visible=True, thickness=0.03), row=7, col=1)

    # Save
    if output_file:
        fig.write_html(str(output_file), include_plotlyjs='cdn')
        print(f"Dashboard saved to {output_file}")

    return fig


# =============================================================================
# DETAILED SITE DASHBOARD
# =============================================================================

def create_detailed_site_dashboard(
    df: pd.DataFrame,
    site_id: str,
    financials: Dict,
    site_config,
    output_file: Path = None,
    delta_t: float = 0.25
) -> go.Figure:
    """
    Create detailed dashboard for a single site with action reasoning.
    """
    
    site_df = df[df['site'] == site_id].copy()
    site_fin = financials['sites'][site_id]
    
    # Classify actions
    site_df = classify_actions(site_df)
    
    E_max = site_config.battery.E_max
    E_btm = E_max * site_config.btm_ratio
    E_ftm = E_max * (1 - site_config.btm_ratio)
    
    fig = make_subplots(
        rows=5, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.04,
        row_heights=[0.22, 0.20, 0.18, 0.20, 0.20],
        subplot_titles=(
            f'<b>1. GRID & LOAD</b> — {site_id}',
            '<b>2. BATTERY OPERATIONS</b> — Charge (+) / Discharge (-)',
            '<b>3. STATE OF CHARGE</b> — BTM & FTM',
            '<b>4. MARKET PRICES</b> — Decision Drivers',
            '<b>5. ACTION TIMELINE</b> — What & Why'
        ),
        specs=[
            [{"secondary_y": False}],
            [{"secondary_y": False}],
            [{"secondary_y": False}],
            [{"secondary_y": True}],
            [{"secondary_y": False}]
        ]
    )
    
    # ==========================================================================
    # ROW 1: GRID & LOAD
    # ==========================================================================
    
    # Demand
    fig.add_trace(go.Scatter(
        x=site_df['datetime'], y=site_df['demand'],
        name='Demand',
        fill='tozeroy',
        fillcolor='rgba(127,140,141,0.3)',
        line=dict(color=COLORS['demand'], width=1),
        hovertemplate='Demand: %{y:.0f} kW<extra></extra>'
    ), row=1, col=1)
    
    # Grid import
    fig.add_trace(go.Scatter(
        x=site_df['datetime'], y=site_df['P_buy'],
        name='Grid Import',
        line=dict(color=COLORS['grid_import'], width=2),
        hovertemplate='Import: %{y:.0f} kW<extra></extra>'
    ), row=1, col=1)
    
    # Grid export (negative)
    fig.add_trace(go.Scatter(
        x=site_df['datetime'], y=-site_df['P_sell'],
        name='Grid Export',
        line=dict(color=COLORS['grid_export'], width=2),
        fill='tozeroy',
        fillcolor='rgba(231,76,60,0.2)',
        hovertemplate='Export: %{y:.0f} kW<extra></extra>'
    ), row=1, col=1)
    
    # Peak line
    fig.add_hline(y=site_fin['peak_kw_new'], line_dash="dot", line_color="red", 
                  line_width=2, row=1, col=1, annotation_text="Optimized Peak")
    fig.add_hline(y=0, line_dash="solid", line_color="black", line_width=1, row=1, col=1)
    
    # ==========================================================================
    # ROW 2: BATTERY OPERATIONS
    # ==========================================================================
    
    # BTM
    fig.add_trace(go.Bar(
        x=site_df['datetime'], y=site_df['P_ch_BTM'],
        name='BTM Charge',
        marker_color=COLORS['btm_charge'],
        hovertemplate='BTM↑: %{y:.0f} kW<extra></extra>'
    ), row=2, col=1)
    
    fig.add_trace(go.Bar(
        x=site_df['datetime'], y=-site_df['P_dis_BTM'],
        name='BTM Discharge',
        marker_color=COLORS['btm_discharge'],
        hovertemplate='BTM↓: %{y:.0f} kW<extra></extra>'
    ), row=2, col=1)
    
    # FTM
    fig.add_trace(go.Bar(
        x=site_df['datetime'], y=site_df['P_ch_FTM'],
        name='FTM Charge',
        marker_color=COLORS['ftm_charge'],
        hovertemplate='FTM↑: %{y:.0f} kW<extra></extra>'
    ), row=2, col=1)
    
    fig.add_trace(go.Bar(
        x=site_df['datetime'], y=-site_df['P_dis_FTM'],
        name='FTM Discharge',
        marker_color=COLORS['ftm_discharge'],
        hovertemplate='FTM↓: %{y:.0f} kW<extra></extra>'
    ), row=2, col=1)
    
    fig.add_hline(y=0, line_dash="solid", line_color="black", line_width=1, row=2, col=1)
    
    # ==========================================================================
    # ROW 3: SOC
    # ==========================================================================
    
    site_df['SOC_BTM_pct'] = 100 * site_df['SOC_BTM'] / E_btm if E_btm > 0 else 0
    site_df['SOC_FTM_pct'] = 100 * site_df['SOC_FTM'] / E_ftm if E_ftm > 0 else 0
    
    fig.add_trace(go.Scatter(
        x=site_df['datetime'], y=site_df['SOC_BTM_pct'],
        name='BTM SOC',
        line=dict(color=COLORS['btm_discharge'], width=2),
        fill='tozeroy',
        fillcolor='rgba(39,174,96,0.2)',
        hovertemplate='BTM: %{y:.1f}%<extra></extra>'
    ), row=3, col=1)
    
    fig.add_trace(go.Scatter(
        x=site_df['datetime'], y=site_df['SOC_FTM_pct'],
        name='FTM SOC',
        line=dict(color=COLORS['ftm_discharge'], width=2),
        fill='tozeroy',
        fillcolor='rgba(26,188,156,0.2)',
        hovertemplate='FTM: %{y:.1f}%<extra></extra>'
    ), row=3, col=1)
    
    # ==========================================================================
    # ROW 4: MARKET PRICES
    # ==========================================================================
    
    price_mwh = site_df['Price'] * 1000
    
    fig.add_trace(go.Scatter(
        x=site_df['datetime'], y=price_mwh,
        name='DA Price',
        line=dict(color=COLORS['price_high'], width=2),
        fill='tozeroy',
        fillcolor='rgba(192,57,43,0.1)',
        hovertemplate='DA: %{y:.1f} €/MWh<extra></extra>'
    ), row=4, col=1)
    
    # Highlight negative prices
    neg_mask = price_mwh < 0
    if neg_mask.any():
        fig.add_trace(go.Scatter(
            x=site_df.loc[neg_mask, 'datetime'],
            y=price_mwh[neg_mask],
            name='Negative Price',
            mode='markers',
            marker=dict(color=COLORS['price_low'], size=8, symbol='diamond'),
            hovertemplate='NEG: %{y:.1f} €/MWh<extra></extra>'
        ), row=4, col=1)
    
    fig.add_hline(y=0, line_dash="dash", line_color="black", line_width=1, row=4, col=1)
    
    # FCR price
    fig.add_trace(go.Scatter(
        x=site_df['datetime'], y=site_df['FCR_price'] * 1000,
        name='FCR Price',
        line=dict(color=COLORS['fcr'], width=2, dash='dash'),
        hovertemplate='FCR: %{y:.1f} €/MW<extra></extra>'
    ), row=4, col=1, secondary_y=True)
    
    # ==========================================================================
    # ROW 5: ACTION TIMELINE
    # ==========================================================================
    
    # Create action bars with hover showing reason
    action_colors = {
        'Idle': 'rgba(200,200,200,0.3)',
        'BTM↑': COLORS['btm_charge'],
        'BTM↓': COLORS['btm_discharge'],
        'FTM↑': COLORS['ftm_charge'],
        'FTM↓': COLORS['ftm_discharge'],
        'FCR': COLORS['fcr'],
    }
    
    # Plot action summary as colored bars
    action_vals = []
    action_colors_list = []
    hover_texts = []
    
    for _, row in site_df.iterrows():
        summary = row['action_summary']
        btm_reason = row['BTM_reason']
        ftm_reason = row['FTM_reason']
        
        # Determine color and value
        if 'FCR' in summary:
            val = 3
            color = COLORS['fcr']
        elif 'FTM↓' in summary:
            val = -2
            color = COLORS['ftm_discharge']
        elif 'FTM↑' in summary:
            val = 2
            color = COLORS['ftm_charge']
        elif 'BTM↓' in summary:
            val = -1
            color = COLORS['btm_discharge']
        elif 'BTM↑' in summary:
            val = 1
            color = COLORS['btm_charge']
        else:
            val = 0
            color = 'rgba(200,200,200,0.3)'
        
        action_vals.append(val)
        action_colors_list.append(color)
        
        # Build hover text
        hover = f"<b>{summary}</b>"
        if btm_reason:
            hover += f"<br>BTM: {btm_reason}"
        if ftm_reason:
            hover += f"<br>FTM: {ftm_reason}"
        hover_texts.append(hover)
    
    fig.add_trace(go.Bar(
        x=site_df['datetime'],
        y=action_vals,
        name='Action',
        marker_color=action_colors_list,
        hovertext=hover_texts,
        hovertemplate='%{hovertext}<extra></extra>'
    ), row=5, col=1)
    
    # Add action legend annotations
    fig.add_annotation(
        xref='paper', yref='y5',
        x=0.02, y=3, text='FCR', showarrow=False, font=dict(size=9)
    )
    fig.add_annotation(
        xref='paper', yref='y5',
        x=0.02, y=2, text='FTM↑', showarrow=False, font=dict(size=9)
    )
    fig.add_annotation(
        xref='paper', yref='y5',
        x=0.02, y=1, text='BTM↑', showarrow=False, font=dict(size=9)
    )
    fig.add_annotation(
        xref='paper', yref='y5',
        x=0.02, y=-1, text='BTM↓', showarrow=False, font=dict(size=9)
    )
    fig.add_annotation(
        xref='paper', yref='y5',
        x=0.02, y=-2, text='FTM↓', showarrow=False, font=dict(size=9)
    )
    
    fig.add_hline(y=0, line_dash="solid", line_color="black", line_width=1, row=5, col=1)
    
    # ==========================================================================
    # LAYOUT
    # ==========================================================================
    
    fig.update_layout(
        title=dict(
            text=f'<b>{site_id} Detailed Dashboard</b>',
            font=dict(size=18)
        ),
        height=1200,
        template='plotly_white',
        hovermode='x unified',
        barmode='relative',
        showlegend=True,
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=-0.1,
            xanchor='center',
            x=0.5,
            font=dict(size=9)
        ),
        
        # KPI box
        annotations=[
            dict(
                text=f"<b>{site_id}</b><br>"
                     f"───────────<br>"
                     f"Battery: {site_config.battery.E_max:.0f} kWh<br>"
                     f"Power: {site_config.battery.P_max:.0f} kW<br>"
                     f"BTM: {site_config.btm_ratio*100:.0f}% | FTM: {(1-site_config.btm_ratio)*100:.0f}%<br>"
                     f"───────────<br>"
                     f"Peak: {site_fin['peak_kw_old']:.0f}→{site_fin['peak_kw_new']:.0f} kW<br>"
                     f"BTM Sav: €{site_fin['btm_savings']:,.0f}<br>"
                     f"Peak Sav: €{site_fin['peak_savings']:,.0f}<br>"
                     f"FCR Rev: €{site_fin['fcr_revenue']:,.0f}<br>"
                     f"───────────<br>"
                     f"<b>Net: €{site_fin['net_benefit']:,.0f}</b>",
                xref='paper', yref='paper',
                x=1.02, y=0.98,
                showarrow=False,
                font=dict(size=10, family='monospace'),
                bgcolor='rgba(255,255,255,0.95)',
                bordercolor='#34495e',
                borderwidth=2,
                borderpad=8,
                align='left'
            )
        ]
    )
    
    # Y-axis labels
    fig.update_yaxes(title_text='Power (kW)', row=1, col=1)
    fig.update_yaxes(title_text='Power (kW)', row=2, col=1)
    fig.update_yaxes(title_text='SOC (%)', row=3, col=1, range=[0, 100])
    fig.update_yaxes(title_text='DA (€/MWh)', row=4, col=1)
    fig.update_yaxes(title_text='FCR (€/MW)', row=4, col=1, secondary_y=True)
    fig.update_yaxes(title_text='Action', row=5, col=1, range=[-3, 4])
    
    fig.update_xaxes(title_text='Time', row=5, col=1)
    fig.update_xaxes(rangeslider=dict(visible=True, thickness=0.03), row=5, col=1)
    
    if output_file:
        fig.write_html(str(output_file), include_plotlyjs='cdn')
        print(f"Site dashboard saved to {output_file}")
    
    return fig


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def print_financial_summary(financials: Dict):
    """Print formatted financial summary to console with clear explanations."""
    
    port = financials['portfolio']
    
    print("\n" + "=" * 90)
    print("💰 FINANCIAL SUMMARY - VALUE ATTRIBUTION")
    print("=" * 90)
    
    # ==========================================================================
    # COST COMPARISON
    # ==========================================================================
    print("\n📊 COST COMPARISON (Energy + Peak Charges)")
    print("-" * 90)
    
    baseline_total = port['baseline_energy'] + port['baseline_peak']
    optimized_total = port['energy_cost'] + port['peak_cost']
    gross_savings = baseline_total - optimized_total
    
    print(f"  WITHOUT BATTERY (Baseline):")
    print(f"    Energy Cost:         €{port['baseline_energy']:>12,.0f}  (demand × spot price)")
    print(f"    Peak Charge:         €{port['baseline_peak']:>12,.0f}  (max demand × peak tariff)")
    print(f"    ─────────────────────────────────────")
    print(f"    TOTAL:               €{baseline_total:>12,.0f}")
    
    print(f"\n  WITH BATTERY (Optimized):")
    print(f"    Energy Cost:         €{port['energy_cost']:>12,.0f}  (net grid × spot price)")
    print(f"    Peak Charge:         €{port['peak_cost']:>12,.0f}  (reduced peak × peak tariff)")
    print(f"    ─────────────────────────────────────")
    print(f"    TOTAL:               €{optimized_total:>12,.0f}")
    
    print(f"\n  GROSS SAVINGS:         €{gross_savings:>12,.0f}  (baseline - optimized)")
    
    # ==========================================================================
    # VALUE ATTRIBUTION
    # ==========================================================================
    print("\n📈 VALUE ATTRIBUTION (Where does the benefit come from?)")
    print("-" * 90)
    
    # BTM: Load shifting vs retail tariffs
    print(f"  🟢 BTM LOAD SHIFTING:  €{port['btm_savings']:>+12,.0f}")
    print(f"      = Retail load served from battery instead of grid")
    print(f"      (Shift consumption from high to low retail tariffs)")
    
    # FTM: Day-ahead arbitrage on wholesale prices
    da_value = port.get('day_ahead_arbitrage', 0.0)
    print(f"\n  🔵 FTM DA ARBITRAGE:   €{da_value:>+12,.0f}")
    print(f"      (Wholesale day-ahead price arbitrage on FTM leg)")
    
    print(f"\n  🟡 PEAK REDUCTION:     €{port['peak_savings']:>+12,.0f}")
    print(f"      = Baseline peak charge - Optimized peak charge")
    print(f"      = €{port['baseline_peak']:,.0f} - €{port['peak_cost']:,.0f} = €{port['peak_savings']:,.0f}")
    if port['peak_savings'] < 0:
        print(f"      ⚠️  NEGATIVE: Battery charging/FCR increased grid peak above demand.")
        print(f"         This is optimal when FCR revenue > peak penalty.")
    else:
        print(f"      (Value from reducing maximum grid import)")
    
    print(f"\n  🟣 FCR REVENUE:        €{port['fcr_revenue']:>+12,.0f}")
    print(f"      (Capacity payments for frequency regulation)")
    
    print(f"\n  🔴 DEGRADATION COST:   €{-port['degradation_cost']:>+12,.0f}")
    print(f"      (Battery wear from cycling + calendar aging)")
    
    # ==========================================================================
    # NET BENEFIT CALCULATION
    # ==========================================================================
    print("\n" + "-" * 90)
    print("✅ NET BENEFIT CALCULATION:")
    print(f"   BTM Load Shifting {port['btm_savings']:>+12,.0f}")
    print(f"   FTM DA Arbitrage  {da_value:>+12,.0f}")
    print(f"   Peak Reduction    {port['peak_savings']:>+12,.0f}")
    print(f"   FCR Revenue       {port['fcr_revenue']:>+12,.0f}")
    print(f"   Degradation       {-port['degradation_cost']:>+12,.0f}")
    print(f"   ─────────────────────────────────────")
    calculated_net = (
        port['btm_savings']
        + da_value
        + port['peak_savings']
        + port['fcr_revenue']
        - port['degradation_cost']
    )
    print(f"   NET BENEFIT       €{calculated_net:>+11,.0f}")
    
    if abs(calculated_net - port['net_benefit']) > 1:
        print(f"   ⚠️  MISMATCH: Stored value = €{port['net_benefit']:,.0f}")
    
    print("=" * 90)
    
    # ==========================================================================
    # PER-SITE BREAKDOWN (condensed)
    # ==========================================================================
    print("\n📋 PER-SITE BREAKDOWN")
    print("-" * 90)
    print(f"{'Site':<12} {'BTM Shift':>12} {'Peak Red':>12} {'FCR Rev':>12} {'Degrad':>10} {'Net':>14}")
    print("-" * 90)
    
    for site_id, data in financials['sites'].items():
        print(f"{site_id:<12} "
              f"€{data['btm_savings']:>+11,.0f} "
              f"€{data['peak_savings']:>+11,.0f} "
              f"€{data['fcr_revenue']:>+11,.0f} "
              f"€{-data['degradation_cost']:>+9,.0f} "
              f"€{data['net_benefit']:>+13,.0f}")
    
    print("-" * 90)
    print(f"{'TOTAL':<12} "
          f"€{port['btm_savings']:>+11,.0f} "
          f"€{port['peak_savings']:>+11,.0f} "
          f"€{port['fcr_revenue']:>+11,.0f} "
          f"€{-port['degradation_cost']:>+9,.0f} "
          f"€{port['net_benefit']:>+13,.0f}")
    print("=" * 90)


def generate_comparison_report(df: pd.DataFrame, financials: Dict, output_file: Path = None):
    """Generate a comparison report across all sites."""
    
    site_ids = df['site'].unique()
    
    report_data = []
    for site_id in site_ids:
        site_df = df[df['site'] == site_id]
        site_fin = financials['sites'][site_id]
        
        report_data.append({
            'Site': site_id,
            'Avg Demand (kW)': site_df['demand'].mean(),
            'Peak Old (kW)': site_fin['peak_kw_old'],
            'Peak New (kW)': site_fin['peak_kw_new'],
            'Peak Reduction (%)': 100 * (1 - site_fin['peak_kw_new'] / max(site_fin['peak_kw_old'], 1)),
            'Avg SOC (%)': site_df['SOC_total_pct'].mean(),
            'BTM Savings (€)': site_fin['btm_savings'],
            'Peak Savings (€)': site_fin['peak_savings'],
            'FCR Revenue (€)': site_fin['fcr_revenue'],
            'Net Benefit (€)': site_fin['net_benefit'],
        })
    
    report_df = pd.DataFrame(report_data)
    
    if output_file:
        report_df.to_csv(output_file, index=False)
        print(f"Comparison report saved to {output_file}")
    
    return report_df


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("Dashboard module - import and call create_multi_battery_dashboard()")
    print("or run model_multi_battery.py for full workflow")
