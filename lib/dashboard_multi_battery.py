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
    delta_t: float = 0.25
) -> go.Figure:
    """
    Create comprehensive multi-battery VPP dashboard with proper subplots.
    """
    
    site_ids = df['site'].unique().tolist()
    
    # Aggregate data across all sites
    agg_df = df.groupby('t').agg({
        'datetime': 'first',
        'demand': 'sum',
        'P_buy': 'sum',
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
    }).reset_index()
    
    total_E_max = sum(cfg.battery.E_max for cfg in site_configs)
    total_E_btm = sum(cfg.battery.E_max * cfg.btm_ratio for cfg in site_configs)
    total_E_ftm = sum(cfg.battery.E_max * (1 - cfg.btm_ratio) for cfg in site_configs)
    
    agg_df['SOC_pct'] = 100 * agg_df['SOC_total'] / total_E_max
    agg_df['SOC_BTM_pct'] = 100 * agg_df['SOC_BTM'] / total_E_btm
    agg_df['SOC_FTM_pct'] = 100 * agg_df['SOC_FTM'] / total_E_ftm
    
    # Classify actions
    agg_df = classify_actions(agg_df)
    
    # Calculate running peak
    agg_df['running_peak'] = agg_df['P_buy'].expanding().max()

    # --- Pre-calculate Revenue Streams for Attribution ---
    # 1. BTM Savings: Avoided cost (Demand - P_buy_BTM) * Retail Price
    #    Note: This is an approximation using aggregated values. Accuracy depends on site-level details,
    #    but for dashboard visualization it is sufficient.
    #    Using P_dis_BTM approximation for visualization impact
    agg_df['val_btm_savings'] = agg_df['P_dis_BTM'] * agg_df['Price industrial'] * delta_t
    
    # 2. DA Arbitrage: Revenue Sell - Cost Buy FTM
    agg_df['val_da_arb'] = (agg_df['P_sell'] * agg_df['Price'] - agg_df['P_ch_FTM'] * agg_df['Price']) * delta_t
    
    # 3. FCR Revenue
    agg_df['val_fcr'] = agg_df['FCR_total'] * agg_df['FCR_price'] * delta_t

    # ==========================================================================
    # CREATE SUBPLOT FIGURE
    # ==========================================================================
    
    fig = make_subplots(
        rows=6, cols=1,
        shared_xaxes=False, # We will manually link rows 1-5, Row 6 is monthly
        vertical_spacing=0.03,
        row_heights=[0.16, 0.16, 0.16, 0.16, 0.16, 0.20],
        subplot_titles=(
            '<b>1. GRID EXCHANGE</b> — Import (+) / Export (-) vs Original Demand',
            '<b>2. BATTERY OPERATIONS</b> — BTM (Local Load) vs FTM (FCR Market)',
            '<b>3. FTM OVERVIEW</b> — SOC, DA Price & FCR Price',
            '<b>4. BTM OVERVIEW</b> — SOC & Retail Price',
            '<b>5. FCR PARTICIPATION</b> — Aggregated Bid (≥1MW required)',
            '<b>6. MONTHLY REVENUE ATTRIBUTION</b> — Stacked Value Creation'
        ),
        specs=[
            [{"secondary_y": True}],
            [{"secondary_y": False}],
            [{"secondary_y": True}], # FTM: SOC + Prices
            [{"secondary_y": True}], # BTM: SOC + Retail Price
            [{"secondary_y": False}],
            [{"secondary_y": False}]
        ]
    )
    
    # ==========================================================================
    # ROW 1: GRID EXCHANGE
    # ==========================================================================
    
    # Original demand (background)
    fig.add_trace(go.Scatter(
        x=agg_df['datetime'], y=agg_df['demand'],
        name='Original Demand',
        fill='tozeroy',
        line=dict(width=0),
        fillcolor='rgba(127,140,141,0.3)',
        hovertemplate='Demand: %{y:.0f} kW<extra></extra>'
    ), row=1, col=1)
    
    # Grid import (positive)
    fig.add_trace(go.Scatter(
        x=agg_df['datetime'], y=agg_df['P_buy'],
        name='Grid Import',
        line=dict(color=COLORS['grid_import'], width=2),
        hovertemplate='Import: %{y:.0f} kW<extra></extra>'
    ), row=1, col=1)
    
    # Grid export (negative)
    fig.add_trace(go.Scatter(
        x=agg_df['datetime'], y=-agg_df['P_sell'],
        name='Grid Export',
        line=dict(color=COLORS['grid_export'], width=2),
        fill='tozeroy',
        fillcolor='rgba(231,76,60,0.2)',
        hovertemplate='Export: %{y:.0f} kW<extra></extra>'
    ), row=1, col=1)
    
    # Running peak line
    fig.add_trace(go.Scatter(
        x=agg_df['datetime'], y=agg_df['running_peak'],
        name='Peak (Running Max)',
        line=dict(color=COLORS['peak'], width=1, dash='dot'),
        hovertemplate='Peak: %{y:.0f} kW<extra></extra>'
    ), row=1, col=1)
    
    # Zero line
    fig.add_hline(y=0, line_dash="dash", line_color="black", line_width=1, row=1, col=1)
    
    # Price on secondary axis (faded)
    fig.add_trace(go.Scatter(
        x=agg_df['datetime'], y=agg_df['Price'] * 1000,
        name='DA Price',
        line=dict(color='rgba(192,57,43,0.4)', width=1),
        hovertemplate='%{y:.1f} €/MWh<extra></extra>'
    ), row=1, col=1, secondary_y=True)
    
    # ==========================================================================
    # ROW 2: BATTERY OPERATIONS (BTM vs FTM)
    # ==========================================================================
    
    # BTM charging (positive = charging)
    fig.add_trace(go.Bar(
        x=agg_df['datetime'], y=agg_df['P_ch_BTM'],
        name='BTM Charge',
        marker_color=COLORS['btm_charge'],
        opacity=0.8,
        hovertemplate='BTM Charge: %{y:.0f} kW<extra></extra>'
    ), row=2, col=1)
    
    # BTM discharging (negative for visual)
    fig.add_trace(go.Bar(
        x=agg_df['datetime'], y=-agg_df['P_dis_BTM'],
        name='BTM Discharge',
        marker_color=COLORS['btm_discharge'],
        opacity=0.8,
        hovertemplate='BTM Discharge: %{y:.0f} kW<extra></extra>'
    ), row=2, col=1)
    
    # FTM charging
    fig.add_trace(go.Bar(
        x=agg_df['datetime'], y=agg_df['P_ch_FTM'],
        name='FTM Charge',
        marker_color=COLORS['ftm_charge'],
        opacity=0.8,
        hovertemplate='FTM Charge: %{y:.0f} kW<extra></extra>'
    ), row=2, col=1)
    
    # FTM discharging
    fig.add_trace(go.Bar(
        x=agg_df['datetime'], y=-agg_df['P_dis_FTM'],
        name='FTM Discharge',
        marker_color=COLORS['ftm_discharge'],
        opacity=0.8,
        hovertemplate='FTM Discharge: %{y:.0f} kW<extra></extra>'
    ), row=2, col=1)
    
    # Zero line
    fig.add_hline(y=0, line_dash="solid", line_color="black", line_width=1, row=2, col=1)
    
    # ==========================================================================
    # ROW 3: FTM OVERVIEW (SOC + DA Price + FCR Price)
    # ==========================================================================
    
    # FTM SOC (Area)
    fig.add_trace(go.Scatter(
        x=agg_df['datetime'], y=agg_df['SOC_FTM_pct'],
        name='FTM SOC',
        line=dict(color=COLORS['soc'], width=2),
        fill='tozeroy',
        fillcolor='rgba(46, 204, 113, 0.2)',
        hovertemplate='FTM SOC: %{y:.1f}%<extra></extra>'
    ), row=3, col=1)

    # DA Price (Line, Secondary Y)
    fig.add_trace(go.Scatter(
        x=agg_df['datetime'], y=agg_df['Price'] * 1000,
        name='DA Price',
        line=dict(color=COLORS['price_high'], width=1.5),
        hovertemplate='DA: %{y:.1f} €/MWh<extra></extra>'
    ), row=3, col=1, secondary_y=True)

    # FCR Price (Dashed Line, Secondary Y)
    fig.add_trace(go.Scatter(
        x=agg_df['datetime'], y=agg_df['FCR_price'] * 1000,
        name='FCR Price',
        line=dict(color=COLORS['fcr'], width=1.5, dash='dash'),
        hovertemplate='FCR: %{y:.1f} €/MW<extra></extra>'
    ), row=3, col=1, secondary_y=True)

    # ==========================================================================
    # ROW 4: BTM OVERVIEW (SOC + Retail Price)
    # ==========================================================================
    
    # BTM SOC (Area)
    fig.add_trace(go.Scatter(
        x=agg_df['datetime'], y=agg_df['SOC_BTM_pct'],
        name='BTM SOC',
        line=dict(color=COLORS['btm_charge'], width=2),
        fill='tozeroy',
        fillcolor='rgba(230, 126, 34, 0.2)',
        hovertemplate='BTM SOC: %{y:.1f}%<extra></extra>'
    ), row=4, col=1)

    # Retail Price (Line, Secondary Y)
    fig.add_trace(go.Scatter(
        x=agg_df['datetime'], y=agg_df['Price industrial'] * 1000,
        name='Retail Price',
        line=dict(color='black', width=1.5, dash='dot'),
        hovertemplate='Retail: %{y:.1f} €/MWh<extra></extra>'
    ), row=4, col=1, secondary_y=True)
    
    # ==========================================================================
    # ROW 5: FCR PARTICIPATION
    # ==========================================================================
    
    # FCR bid (aggregated)
    fig.add_trace(go.Scatter(
        x=agg_df['datetime'], y=agg_df['FCR_total'],
        name='FCR Bid',
        line=dict(color=COLORS['fcr'], width=2),
        fill='tozeroy',
        fillcolor='rgba(142,68,173,0.3)',
        hovertemplate='FCR Bid: %{y:.0f} kW<extra></extra>'
    ), row=5, col=1)
    
    # 1 MW minimum threshold
    fig.add_hline(y=1000, line_dash="dot", line_color="red", line_width=2, row=5, col=1,
                  annotation_text="1 MW Minimum", annotation_position="right")
    
    # FCR activation markers
    fcr_activations = agg_df[agg_df['FCR_signal'] != 0].copy()
    if len(fcr_activations) > 0:
        # Positive activations (charging)
        pos_act = fcr_activations[fcr_activations['FCR_signal'] > 0]
        if len(pos_act) > 0:
            fig.add_trace(go.Scatter(
                x=pos_act['datetime'], y=pos_act['FCR_total'],
                name='FCR Charge Event',
                mode='markers',
                marker=dict(color=COLORS['btm_charge'], size=12, symbol='triangle-up',
                           line=dict(color='black', width=1)),
                hovertemplate='FCR CHARGE: %{y:.0f} kW<extra></extra>'
            ), row=5, col=1)
        
        # Negative activations (discharging)
        neg_act = fcr_activations[fcr_activations['FCR_signal'] < 0]
        if len(neg_act) > 0:
            fig.add_trace(go.Scatter(
                x=neg_act['datetime'], y=neg_act['FCR_total'],
                name='FCR Discharge Event',
                mode='markers',
                marker=dict(color=COLORS['grid_export'], size=12, symbol='triangle-down',
                           line=dict(color='black', width=1)),
                hovertemplate='FCR DISCHARGE: %{y:.0f} kW<extra></extra>'
            ), row=5, col=1)
    
    # ==========================================================================
    # ROW 6: MONTHLY REVENUE ATTRIBUTION
    # ==========================================================================
    
    # Resample to Monthly sums
    # Ensure datetime is index
    agg_df['datetime_idx'] = pd.to_datetime(agg_df['datetime'])
    
    # Yearly Peak Tariff (approximate monthly attribution)
    C_PEAK_ANNUAL = 192.66 
    
    # Pre-calculate BTM Charging Cost for Net Load Shifting
    agg_df['val_btm_cost'] = agg_df['P_ch_BTM'] * agg_df['Price industrial'] * delta_t
    
    # 1. Resample sums for Energy Value
    monthly_energy = agg_df.set_index('datetime_idx')[['val_btm_savings', 'val_da_arb', 'val_fcr', 'val_btm_cost']].resample('ME').sum()
    
    # Calculate Net Load Shifting (Gross Savings - Charging Cost)
    # Note: val_btm_savings from agg_df was P_dis * Price * dt
    # We subtract val_btm_cost (P_ch * Price * dt)
    monthly_energy['val_load_shifting'] = monthly_energy['val_btm_savings'] - monthly_energy['val_btm_cost']
    
    # 2. Resample Max for Peak Savings
    monthly_peaks = agg_df.set_index('datetime_idx')[['demand', 'P_buy']].resample('ME').max()
    
    # Calculate Monthly Peak Savings
    # Assumption: Peak savings are valued at 1/12th of annual tariff per month for visualization
    monthly_peaks['val_peak_savings'] = (monthly_peaks['demand'] - monthly_peaks['P_buy']) * (C_PEAK_ANNUAL / 12.0)
    
    # Merge
    monthly_df = pd.concat([monthly_energy, monthly_peaks], axis=1)
    monthly_df['month_name'] = monthly_df.index.strftime('%B %Y')
    
    # Plot Stacked Bars - DETAILED REVENUE ATTRIBUTION
    
    # 1. Peak Shaving
    fig.add_trace(go.Bar(
        x=monthly_df['month_name'], y=monthly_df['val_peak_savings'],
        name='Peak Shaving',
        marker_color=COLORS['peak'],
        hovertemplate='Peak Shaving: €%{y:,.0f}<extra>Reduced max grid load</extra>'
    ), row=6, col=1)

    # 2. Load Shifting (Net)
    fig.add_trace(go.Bar(
        x=monthly_df['month_name'], y=monthly_df['val_load_shifting'],
        name='Load Shifting (Net)',
        marker_color=COLORS['btm_charge'],
        hovertemplate='Load Shifting: €%{y:,.0f}<extra>Retail price arbitrage</extra>'
    ), row=6, col=1)
    
    # 3. DA Arbitrage
    fig.add_trace(go.Bar(
        x=monthly_df['month_name'], y=monthly_df['val_da_arb'],
        name='DA Arbitrage',
        marker_color=COLORS['ftm_charge'],
        hovertemplate='DA Arbitrage: €%{y:,.0f}<extra>Wholesale trading</extra>'
    ), row=6, col=1)
    
    # 4. FCR Revenue
    fig.add_trace(go.Bar(
        x=monthly_df['month_name'], y=monthly_df['val_fcr'],
        name='FCR Revenue',
        marker_color=COLORS['fcr'],
        hovertemplate='FCR Revenue: €%{y:,.0f}<extra>Frequency support</extra>'
    ), row=6, col=1)
    
    # Note: Peak savings not shown as they are annual/difficult to attribute monthly
    
    # ==========================================================================
    # LAYOUT
    # ==========================================================================
    
    fig.update_layout(
        title=dict(
            text=f'<b>VPP Portfolio Dashboard</b> — {len(site_ids)} Sites, '
                 f'{total_E_max:.0f} kWh Total Capacity',
            font=dict(size=20)
        ),
        height=1400,
        template='plotly_white',
        hovermode='x unified',
        barmode='relative',
        showlegend=True,
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=-0.08,
            xanchor='center',
            x=0.5,
            font=dict(size=10)
        ),
        
        # Add KPI annotation box with clear value attribution
        # annotations=[
            # dict(
            #     text=f"<b>📊 PORTFOLIO SUMMARY</b><br>"
            #          f"━━━━━━━━━━━━━━━━━━━━━━━━<br>"
            #          f"Sites: {len(site_ids)} | Batteries: {total_E_max:,.0f} kWh<br>"
            #          f"FTM (Market): {sum(c.battery.P_max*(1-c.btm_ratio) for c in site_configs):,.0f} kW<br>"
            #          f"<br>"
            #          f"<b>💰 COSTS</b><br>"
            #          f"━━━━━━━━━━━━━━━━━━━━━━━━<br>"
            #          f"Baseline (no battery): €{port['baseline_energy'] + port['baseline_peak']:,.0f}<br>"
            #          f"Optimized cost:        €{port['energy_cost'] + port['peak_cost']:,.0f}<br>"
            #          f"<br>"
            #          f"<b>📈 VALUE BREAKDOWN</b><br>"
            #          f"━━━━━━━━━━━━━━━━━━━━━━━━<br>"
            #          f"🟢 BTM Load Shifting:  €{port['btm_savings']:>+10,.0f}<br>"
            #          f"   (shift retail load high→low price)<br>"
            #          f"🔵 FTM DA Arbitrage:   €{port.get('day_ahead_arbitrage', 0):>+10,.0f}<br>"
            #          f"   (wholesale day-ahead arbitrage)<br>"
            #          f"🟡 Peak Reduction:     €{port['peak_savings']:>+10,.0f}<br>"
            #          f"   ({baseline_peak_kw:.0f}→{optimized_peak_kw:.0f} kW)<br>"
            #          f"🟣 FCR Revenue:        €{port['fcr_revenue']:>+10,.0f}<br>"
            #          f"   (capacity payments, ≥1 MW bid)<br>"
            #          f"🔴 Degradation:        €{-port['degradation_cost']:>+10,.0f}<br>"
            #          f"━━━━━━━━━━━━━━━━━━━━━━━━<br>"
            #          f"<b>✅ NET BENEFIT:        €{port['net_benefit']:>+10,.0f}</b>",
            #     xref='paper', yref='paper',
            #     x=1.02, y=0.98,
            #     showarrow=False,
            #     font=dict(size=10, family='monospace'),
            #     bgcolor='rgba(255,255,255,0.95)',
            #     bordercolor='#34495e',
            #     borderwidth=2,
            #     borderpad=10,
            #     align='left'
            # )
        # ]
    )
    
    # Update y-axis labels
    fig.update_yaxes(title_text='Power (kW)', row=1, col=1)
    fig.update_yaxes(title_text='€/MWh', row=1, col=1, secondary_y=True)
    fig.update_yaxes(title_text='Power (kW)', row=2, col=1)
    
    # R3: FTM
    fig.update_yaxes(title_text='SOC (%)', row=3, col=1, range=[0, 100])
    fig.update_yaxes(title_text='Price', row=3, col=1, secondary_y=True, autorange=True) 
    
    # R4: BTM
    fig.update_yaxes(title_text='SOC (%)', row=4, col=1, range=[0, 100])
    fig.update_yaxes(title_text='Retail €/MWh', row=4, col=1, secondary_y=True, autorange=True)

    fig.update_yaxes(title_text='FCR Bid (kW)', row=5, col=1)
    fig.update_yaxes(title_text='Revenue (€)', row=6, col=1)
    
    # Update x-axis linkage for Rows 1-5 (Manual since shared_xaxes=False)
    fig.update_xaxes(matches='x', row=1, col=1)
    fig.update_xaxes(matches='x', row=2, col=1)
    fig.update_xaxes(matches='x', row=3, col=1)
    fig.update_xaxes(matches='x', row=4, col=1)
    fig.update_xaxes(matches='x', row=5, col=1)
    # Row 6 is independent
    fig.update_xaxes(title_text='Month', row=6, col=1)
    
    # Add range slider only to bottom plot
    fig.update_xaxes(rangeslider=dict(visible=True, thickness=0.03), row=6, col=1)
    
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
