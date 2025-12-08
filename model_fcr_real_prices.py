#%%
from pyomo.environ import *
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
from pathlib import Path
from data_utils import load_and_process_data, generate_fcr_activation_profile
from plotly.subplots import make_subplots
import pandas as pd
from pyomo.environ import value
this_file = Path(__file__).parent
#old plotting function
def plot_results(model):

    data = pd.DataFrame({
        'P_ch': [value(model.P_ch[t]) for t in model.T],
        'P_dis': [value(model.P_dis[t]) for t in model.T],
        'P_buy': [value(model.P_buy[t]) for t in model.T],
        'P_sell': [value(model.P_sell[t]) for t in model.T],
        'SOC': [value(model.SOC[t]) for t in model.T],
        'SOH': [value(model.SOH[t]) for t in model.T],
        'Price': [value(model.C_buy[t]) for t in model.T],
        'Demand': [value(model.D[t]) for t in model.T]
    }, index=model.T)

    data['P_buy_new'] = np.maximum(data['Demand'] + data['P_ch'] - data['P_dis'], 0)
    data['P_sell_new'] = np.maximum(-(data['Demand'] + data['P_ch'] - data['P_dis']), 0)

    delta_t = 0.25
    print("Cost without objective: ", (day_ahead * load * delta_t).sum() + load.max()*peak_tarif)
    objective_value = value(model.Obj)
    print("Cost optimized:", objective_value)

    savings_load_shift = sum(data["P_buy_new"]*delta_t*data["Price"])
    savings_peak_shave = (data["Demand"].max() - data["P_buy"].max())*peak_tarif
    trading_revenue = sum(data["P_sell_new"]*delta_t*data["Price"])

    print("Peak capacity: ", load.max())
    print("Peak capacity optimized: ", data["P_buy"].max())

    print(f"Cost savings load shifting: {savings_load_shift} Eur")
    print(f"Cost savings peak shave: {savings_peak_shave} Eur")
    print(f"Revenue day-ahead trading: {trading_revenue} Eur")



    # Compute net battery power: charge - discharge
    data['P_net'] = data['P_ch'] - data['P_dis']

    # Optional: create datetime index if 15-min intervals
    start = pd.Timestamp('2024-01-01 00:00')
    delta_t = pd.Timedelta(minutes=15)
    data.index = [start + t*delta_t for t in model.T]

    # --- Plotly figure ---
    fig = go.Figure()

    # Net battery power
    fig.add_trace(go.Scatter(
        x=data.index,
        y=data['P_net'],
        name='Net Battery Power (kW)',
        line=dict(color='blue'),
        yaxis='y1'
    ))

    fig.add_trace(go.Scatter(
        x=data.index,
        y=data['P_buy'],
        name='Optimized load (kW)',
        line=dict(color='black'),
        yaxis='y1'
    ))

    fig.add_trace(go.Scatter(
        x=data.index,
        y=data['P_sell'],
        name='Day ahead sell (kWh)',
        line=dict(color='purple'),
        yaxis='y1'
    ))

    fig.add_trace(go.Scatter(
        x=data.index,
        y=data['SOH'],
        name='State of health',
        line=dict(color='grey'),
        yaxis='y3'
    ))

    # Day-ahead price
    fig.add_trace(go.Scatter(
        x=data.index,
        y=data['Price'],
        name='Day-ahead Price ($/kWh)',
        line=dict(color='red'),
        yaxis='y2'
    ))

    # Demand
    fig.add_trace(go.Scatter(
        x=data.index,
        y=data['Demand'],
        name='Demand (kW)',
        line=dict(color='green'),
        yaxis='y1'
    ))

    fig.add_trace(go.Scatter(
        x=data.index,
        y=data['SOC'],
        name='State of charge (kWh)',
        line=dict(color='orange'),
        yaxis='y1'
    ))

    # --- Layout with 3 y-axes ---
    fig.update_layout(
        title='Battery Net Power vs Day-ahead Price and Demand',
        xaxis=dict(title='Time'),
        yaxis=dict(title='Net Battery Power (kW)', side='left', color='blue'),
        yaxis2=dict(title='Price ($/kWh)', overlaying='y', side='right', color='red'),
        yaxis3=dict(title='Demand (kW)', overlaying='y', side='right', position=0.95, color='green'),
        legend=dict(x=0.01, y=0.99)
    )

    fig.write_html("charge_plot_with_SOH.html")



def plot_comprehensive_results(model, delta_t=0.25, E_nom=None, c_peak=None):
    """
    Generates a 4-row Interactive Dashboard for Battery Optimization.
    
    Args:
        model: The solved Pyomo model
        delta_t: Time step duration (hours)
        E_nom: Total Nominal Energy Capacity in kWh (Required for SOC %)
        c_peak: (Optional) Peak Tariff price in EUR/kW. If None, tries to read model.C_peak.
    """
    print("Extracting and processing optimization results...")
    
    # --- 1. DATA PREPARATION ---
    
    # Validation: Ensure Capacity is provided
    if E_nom is None:
        if hasattr(model, 'E_bat_max'):
            E_nom = value(model.E_bat_max)
        else:
            raise ValueError("Error: Please provide E_nom (Total Battery Capacity in kWh).")

    # Validation: Peak Tariff Retrieval
    if c_peak is not None:
        c_peak_val = c_peak
    elif hasattr(model, 'C_peak'):
        try:
            c_peak_val = value(model.C_peak)
        except:
            c_peak_val = 0
            print("Warning: model.C_peak exists but has no value. Peak Savings will be 0.")
    else:
        c_peak_val = 0
        print("Warning: C_peak not found in model or arguments. Peak Savings will be 0.")

    records = []
    is_split = hasattr(model, 'P_ch_BTM')

    for t in model.T:
        # A. Basic Market & Grid Data
        price = value(model.C_buy[t])
        demand = value(model.D[t])
        p_buy = value(model.P_buy[t])
        p_sell = value(model.P_sell[t])
        fcr_price = value(model.C_FCR[t]) if hasattr(model, 'C_FCR') else 0
        
        # Grid Net: Positive = Import, Negative = Export
        grid_net = p_buy - p_sell
        
        # B. Battery Flows & Physics
        if is_split:
            # 1. BTM Partition
            p_btm_ch = value(model.P_ch_BTM[t])
            p_btm_dis = value(model.P_dis_BTM[t])
            p_btm_net = p_btm_ch - p_btm_dis
            
            # 2. FTM Partition
            p_ftm_ch = value(model.P_ch_FTM[t])
            p_ftm_dis = value(model.P_dis_FTM[t])
            
            # 3. FCR Logic (Signal already includes efficiency)
            sig = value(model.FCR_signal[t])
            p_fcr_bid = value(model.P_FCR_bid[t]) # Define variable for later use
            
            fcr_physical_flow = p_fcr_bid * sig 
            
            # Net FTM = Scheduled Arbitrage + FCR Activation
            p_ftm_net = (p_ftm_ch - p_ftm_dis) + fcr_physical_flow
            
            # State Variables
            soc_kwh = value(model.SOC_BTM[t]) + value(model.SOC_FTM[t])
            
        else:
            # Fallback for simple model
            p_btm_ch = value(model.P_ch[t])
            p_btm_dis = value(model.P_dis[t])
            p_btm_net = p_btm_ch - p_btm_dis
            p_ftm_net = 0
            soc_kwh = value(model.SOC[t])
            p_fcr_bid = 0 # Explicit definition

        # Total Battery Net Flow
        batt_total_net = p_btm_net + p_ftm_net
        
        # SOC Percentage
        current_cap_kwh = E_nom * value(model.SOH[t])
        soc_pct = (soc_kwh / current_cap_kwh * 100) if current_cap_kwh > 0 else 0

        # C. Financial Stream Calculations
        rev_fcr_t = p_fcr_bid * fcr_price * delta_t
        rev_export_t = p_sell * price * delta_t
        sav_btm_t = p_btm_dis * price * delta_t # Gross avoided cost
        
        # Net Cashflow
        cost_base_t = demand * delta_t * price
        cost_opt_t = (p_buy * delta_t * price) - (p_sell * delta_t * price) - rev_fcr_t

        records.append({
            't': t, 
            'Price': price, 
            'Demand': demand, 
            'Grid_Net': grid_net,
            'P_Buy': p_buy, # Stored explicitly for Peak Calculation
            'Batt_Net': batt_total_net, 
            'P_BTM_Net': p_btm_net, 
            'P_FTM_Net': p_ftm_net,
            'SOC_Pct': soc_pct, 
            'SOC_kWh': soc_kwh,
            'Cost_Base': cost_base_t, 
            'Cost_Opt': cost_opt_t,
            'Rev_FCR': rev_fcr_t,
            'Rev_Export': rev_export_t,
            'Sav_BTM': sav_btm_t,
            'P_peak': value(model.P_peak)
        })

    df = pd.DataFrame(records)
    
    # D. Index Alignment
    start_time = pd.Timestamp("2024-01-01 00:00")
    df.index = [start_time + pd.Timedelta(minutes=15*t) for t in df['t']]
    
    # E. Cumulative Financials
    df['Cum_FCR'] = df['Rev_FCR'].cumsum()
    df['Cum_Export'] = df['Rev_Export'].cumsum()
    df['Cum_BTM'] = df['Sav_BTM'].cumsum()
    df['Cum_Cost_Base'] = df['Cost_Base'].cumsum()
    df['Cum_Cost_Opt'] = df['Cost_Opt'].cumsum()
    
    # --- PEAK SAVINGS CALCULATION ---
    # Baseline Peak: Max raw demand (assuming no battery)
    baseline_peak_kw = df['Demand'].max()
    
    # Optimized Peak: Max Grid Import (P_Buy)
    optimized_peak_kw = df['P_Buy'].max()
    
    # Costs
    cost_peak_old = baseline_peak_kw * c_peak_val
    cost_peak_new = optimized_peak_kw * c_peak_val
    peak_savings_eur = cost_peak_old - cost_peak_new
    
    print("-" * 40)
    print(f"PEAK SHAVING REPORT (Price: {c_peak_val:.2f} EUR/kW):")
    print(f"  Old Peak: {baseline_peak_kw:.2f} kW -> Cost: {cost_peak_old:,.2f} EUR")
    print(f"  New Peak: {optimized_peak_kw:.2f} kW -> Cost: {cost_peak_new:,.2f} EUR")
    print(f"  SAVINGS:  {peak_savings_eur:,.2f} EUR")
    print("-" * 40)

    # --- 2. PLOTTING STRATEGY ---
    fig = make_subplots(
        rows=4, cols=1, 
        shared_xaxes=True, 
        vertical_spacing=0.08,
        subplot_titles=(
            "1. GRID PROFILE: Import (+) vs Export (-)",
            "2. BATTERY OPERATIONS: BTM (Self-Cons) vs FTM (FCR/Arb)",
            "3. STATE OF CHARGE (%) vs MARKET PRICE",
            "4. REVENUE STACK: Where is the value coming from?"
        ),
        specs=[[{"secondary_y": False}], [{"secondary_y": False}], [{"secondary_y": True}], [{"secondary_y": False}]]
    )

    # --- ROW 1: GRID ---
    fig.add_trace(go.Scatter(x=df.index, y=df['Demand'], name='Original Load',
                             fill='tozeroy', line=dict(color='lightgrey', width=0), hoverinfo='skip'), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['Grid_Net'], name='Net Grid Profile',
                             line=dict(color='black', width=2)), row=1, col=1)
    # Peak Line (Optimized)
    fig.add_trace(go.Scatter(x=df.index, y=[optimized_peak_kw]*len(df), name='New Peak Limit',
                             line=dict(color='red', dash='dot', width=1)), row=1, col=1)

    # --- ROW 2: BATTERY ---
    fig.add_trace(go.Bar(x=df.index, y=df['P_BTM_Net'], name='BTM Flow',
                         marker=dict(color=df['P_BTM_Net'].apply(lambda x: '#d62728' if x>=0 else '#2ca02c')),
                         marker_line_width=0), row=2, col=1)
    fig.add_trace(go.Bar(x=df.index, y=df['P_FTM_Net'], name='FTM Flow',
                         marker=dict(color=df['P_FTM_Net'].apply(lambda x: '#ff7f0e' if x>=0 else '#1f77b4')),
                         marker_line_width=0), row=2, col=1)

    # --- ROW 3: SOC ---
    fig.add_trace(go.Scatter(x=df.index, y=df['SOC_Pct'], name='SOC %',
                             line=dict(color='green', width=2)), row=3, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['Price'], name='Price',
                             line=dict(color='red', width=1), opacity=0.5), row=3, col=1, secondary_y=True)

    # --- ROW 4: REVENUE ---
    fig.add_trace(go.Scatter(x=df.index, y=df['Cum_BTM'], name='BTM Savings', stackgroup='one', line=dict(width=0, color='#2ca02c')), row=4, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['Cum_Export'], name='Export Rev', stackgroup='one', line=dict(width=0, color='#1f77b4')), row=4, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['Cum_FCR'], name='FCR Rev', stackgroup='one', line=dict(width=0, color='#9467bd')), row=4, col=1)
    
    net_savings = df['Cum_Cost_Base'] - df['Cum_Cost_Opt']
    fig.add_trace(go.Scatter(x=df.index, y=net_savings, name='Net Profit', line=dict(color='black', width=2, dash='dot')), row=4, col=1)

    # Peak Annotation
    fig.add_annotation(
        xref="paper", yref="y4", x=0.02, y=0.90,
        text=f"<b>Peak Savings:</b> {peak_savings_eur:,.0f} € (Lump Sum)",
        showarrow=False, font=dict(color="red", size=12),
        bgcolor="rgba(255,255,255,0.8)", bordercolor="red"
    )

    fig.update_layout(height=1300, template="plotly_white", barmode='relative', hovermode="x unified")
    fig.show()
    fig.write_html("optimization_dashboard_final.html")

    
#%%


def build_split_battery_model(T,
                              C_buy, C_sell, D,
                              E_bat_max, I0, C_peak,
                              P_ch_max, P_dis_max,
                              P_buy_max, P_sell_max,
                              C_FCR, 
                              FCR_signal, # Keeping original (pre-processed) signal
                              btm_ratio=0.4,  
                              eta_ch=0.95, eta_dis=0.95,
                              delta_t=0.25,
                              SOC0=0.5, SOH0=1.0,
                              # Literature SOH Params
                              a=0.0, b=0.0, c=0.0, V_bat=777,
                              # Min Bid Size
                              min_fcr_bid=0): 

    model = ConcreteModel()
    
    # --- Sets ---
    model.T = RangeSet(0, T)
    model.Tstep = RangeSet(0, T-1)
    
    # --- Parameters ---
    model.delta_t = Param(initialize=delta_t)
    model.eta_ch = Param(initialize=eta_ch)
    model.eta_dis = Param(initialize=eta_dis)
    
    model.btm_ratio = Param(initialize=btm_ratio)
    model.ftm_ratio = Param(initialize=1.0 - btm_ratio)
    
    # SOH Params (Attached to model for stability)
    model.a = Param(initialize=a)
    model.b = Param(initialize=b)
    model.c = Param(initialize=c)
    model.V_bat = Param(initialize=V_bat)

    model.min_fcr_bid = Param(initialize=min_fcr_bid)

    # Load and Prices
    model.D = Param(model.T, initialize=lambda m, t: D[t])
    model.C_buy = Param(model.T, initialize=lambda m, t: C_buy[t])
    model.C_sell = Param(model.T, initialize=lambda m, t: C_sell[t])
    model.C_FCR = Param(model.T, initialize=lambda m, t: C_FCR[t])
    model.FCR_signal = Param(model.T, initialize=lambda m, t: FCR_signal[t])
    
    # Calculate Max Power per Partition
    ftm_p_max = P_ch_max * (1.0 - btm_ratio)
    btm_p_max = P_ch_max * btm_ratio
    
    # --- Variables ---
    model.P_ch_BTM = Var(model.T, within=NonNegativeReals, bounds=(0, btm_p_max))
    model.P_dis_BTM = Var(model.T, within=NonNegativeReals, bounds=(0, btm_p_max))
    model.SOC_BTM = Var(model.T, within=NonNegativeReals, bounds=(0, E_bat_max * btm_ratio))
    
    model.P_ch_FTM = Var(model.T, within=NonNegativeReals, bounds=(0, ftm_p_max))
    model.P_dis_FTM = Var(model.T, within=NonNegativeReals, bounds=(0, ftm_p_max))
    model.SOC_FTM = Var(model.T, within=NonNegativeReals, bounds=(0, E_bat_max * (1-btm_ratio)))
    
    # Semi-Continuous FCR Bid
    model.P_FCR_bid = Var(model.T, within=NonNegativeReals, bounds=(0, ftm_p_max))
    model.u_FCR = Var(model.T, within=Binary)
    
    # Binary for Buy/Sell Exclusivity
    model.u_buy = Var(model.T, within=Binary)
    
    model.P_buy = Var(model.T, within=NonNegativeReals, bounds=(0, P_buy_max))
    model.P_sell = Var(model.T, within=NonNegativeReals, bounds=(0, P_sell_max))
    model.P_peak = Var(within=NonNegativeReals)
    
    model.SOH = Var(model.T, within=NonNegativeReals, bounds=(0, 1.0))
    model.E_bat_current = Var(model.T, within=NonNegativeReals)

    # --- Constraints ---

    # 1. Initialization
    def init_soc(m):
        return m.SOC_BTM[0] == SOC0 * E_bat_max * m.btm_ratio
    model.Init_SOC_BTM = Constraint(rule=init_soc)
    
    def init_soc_ftm(m):
        return m.SOC_FTM[0] == SOC0 * E_bat_max * m.ftm_ratio
    model.Init_SOC_FTM = Constraint(rule=init_soc_ftm)
    
    model.Init_SOH = Constraint(rule=lambda m: m.SOH[0] == SOH0)

    # 2. SOC Balances
    def soc_balance_btm(m, t):
        return m.SOC_BTM[t+1] == m.SOC_BTM[t] + (m.eta_ch * m.P_ch_BTM[t] - (1/m.eta_dis) * m.P_dis_BTM[t]) * m.delta_t
    model.SOC_Balance_BTM = Constraint(model.Tstep, rule=soc_balance_btm)
    
    def soc_balance_ftm(m, t):
        # CORRECTED: Back to simple multiplication because FCR_signal contains efficiency
        fcr_flow = m.P_FCR_bid[t] * m.FCR_signal[t]
        return m.SOC_FTM[t+1] == m.SOC_FTM[t] + (m.eta_ch * m.P_ch_FTM[t] - (1/m.eta_dis) * m.P_dis_FTM[t] + fcr_flow) * m.delta_t
    model.SOC_Balance_FTM = Constraint(model.Tstep, rule=soc_balance_ftm)

    # 3. Dynamic Capacity Limits
    def update_capacity(m, t):
        return m.E_bat_current[t] == E_bat_max * m.SOH[t]
    model.Update_Cap = Constraint(model.T, rule=update_capacity)

    def max_soc_btm(m, t):
        return m.SOC_BTM[t] <= m.E_bat_current[t] * m.btm_ratio
    model.Max_SOC_BTM = Constraint(model.T, rule=max_soc_btm)
    
    def max_soc_ftm(m, t):
        return m.SOC_FTM[t] <= (m.E_bat_current[t] * m.ftm_ratio) - (m.P_FCR_bid[t] * m.delta_t * m.eta_ch)
    model.Max_SOC_FTM = Constraint(model.T, rule=max_soc_ftm)

    def min_soc_ftm(m, t):
        return m.SOC_FTM[t] >= (m.P_FCR_bid[t] * m.delta_t * (1/m.eta_dis))
    model.Min_SOC_FTM = Constraint(model.T, rule=min_soc_ftm)

    # 4. Global Power Balance
    def global_balance(m, t):
        total_ch = m.P_ch_BTM[t] + m.P_ch_FTM[t]
        total_dis = m.P_dis_BTM[t] + m.P_dis_FTM[t]
        return m.D[t] + total_ch + m.P_sell[t] == m.P_buy[t] + total_dis 
    model.Global_Balance = Constraint(model.T, rule=global_balance)

    # FIX 1H: Simultaneous Buy/Sell Prevention
    def buy_bin_rule(m, t):
        return m.P_buy[t] <= m.u_buy[t] * P_buy_max
    model.Buy_binary = Constraint(model.T, rule=buy_bin_rule)

    def sell_bin_rule(m, t):
        return m.P_sell[t] <= (1 - m.u_buy[t]) * P_sell_max
    model.Sell_binary = Constraint(model.T, rule=sell_bin_rule)

    # 5. Peak Definition
    def peak_rule(m, t):
        return m.P_buy[t] <= m.P_peak
    model.Peak_Def = Constraint(model.T, rule=peak_rule)

    # 6. FCR Constraints
    def fcr_block_rule(m, t):
        if t == 0 or t % 16 == 0: return Constraint.Skip
        return m.u_FCR[t] == m.u_FCR[t-1]
    model.FCR_Block = Constraint(model.T, rule=fcr_block_rule)
    
    def fcr_bid_consistency(m, t):
        if t == 0 or t % 16 == 0: return Constraint.Skip
        return m.P_FCR_bid[t] == m.P_FCR_bid[t-1]
    model.FCR_Bid_Block = Constraint(model.T, rule=fcr_bid_consistency)

    # Max Bid Limit
    def fcr_bid_max(m, t):
        return m.P_FCR_bid[t] <= m.u_FCR[t] * ftm_p_max
    model.FCR_Bid_Max = Constraint(model.T, rule=fcr_bid_max)
    
    # NEW: Min Bid Limit (Semi-Continuous)
    def fcr_bid_min(m, t):
        return m.P_FCR_bid[t] >= m.u_FCR[t] * m.min_fcr_bid
    model.FCR_Bid_Min = Constraint(model.T, rule=fcr_bid_min)
    
    # 7. SOH Degradation (Literature Model, but with correct Param references)
    def soh_update(m, t):
        total_p_flow = (m.P_ch_BTM[t] + m.P_dis_BTM[t]) + (m.P_ch_FTM[t] + m.P_dis_FTM[t])
        # Using model params m.a, m.b, m.c, m.V_bat, m.delta_t
        return m.SOH[t+1] == m.SOH[t] - (m.a + m.b*(total_p_flow)/m.V_bat + m.c*m.SOH[t])*m.delta_t*3600
    model.SOH_Update = Constraint(model.Tstep, rule=soh_update)

    # 8. Daily Cycle Limit
    steps_per_day = int(24 / delta_t)
    num_days = int((T + 1) // steps_per_day)
    model.Days = RangeSet(0, num_days - 1)

    def daily_cycle_limit_rule(m, d):
        start_t = d * steps_per_day
        end_t = start_t + steps_per_day - 1
        daily_discharge = sum(
            (m.P_dis_BTM[t] + m.P_dis_FTM[t]) * m.delta_t 
            for t in range(start_t, end_t + 1) if t <= T
        )
        return daily_discharge <= 2.0 * E_bat_max
    model.Daily_Cycle_Limit = Constraint(model.Days, rule=daily_cycle_limit_rule)

    # Objective
    def objective_rule(m):
        energy_cost = sum((m.C_buy[t] * m.P_buy[t])*m.delta_t for t in m.T)
        peak_cost = m.P_peak * C_peak
        deg_cost = I0 * (SOH0 - m.SOH[T]) / (SOH0 - 0.8)
        
        energy_revenue = sum((m.C_sell[t] * m.P_sell[t])*m.delta_t for t in m.T)
        fcr_revenue = sum(m.C_FCR[t] * m.P_FCR_bid[t] * m.delta_t for t in m.T)
        
        return (energy_cost + peak_cost + deg_cost) - (energy_revenue + fcr_revenue)
        
    model.Obj = Objective(rule=objective_rule, sense=minimize)

    return model

def build_battery_model(T,
                        C_buy, # energy cost
                        C_sell,
                        D, # demand
                        E_bat_max, # energy capacity
                        I0, # investment cost
                        C_peak, # network tariff per kW
                        P_ch_max, # max charging
                        P_dis_max, # max discharging
                        P_buy_max, # max import
                        P_sell_max, # max export
                        C_FCR,  # FCR prices distributed per 15 minute slot
                        FCR_signal,
                        eta_ch=0.95, # charging efficiency
                        eta_dis=0.95, # discharging efficiency
                        delta_t=0.25, # quarter hour time steps
                        SOC0=0.5, # initial SOC (percentage of E_bat_max)
                        SOH0=1.0, # initial SOH 
                        a=0.0, b=0.0, c=0.0, # SOH degradation parameters
                        V_bat=777# Voltage of battery,

):  
    
    model = ConcreteModel()

    # Timeranges for constraints
    model.T = RangeSet(0, T)
    model.Tstep = RangeSet(0, T-1) # for variables with inital conditions
    block_size = int(round(4.0 / delta_t))
    num_blocks = int(np.ceil((T + 1) / block_size))
    model.B = RangeSet(0, num_blocks - 1)
    block_map = {t: t // block_size for t in range(T + 1)}
    fcr_upper_bound = min(P_ch_max, P_dis_max)

    # Variables
    model.P_ch = Var(model.T, within=NonNegativeReals, bounds=(0, P_ch_max))
    model.P_dis = Var(model.T, within=NonNegativeReals, bounds=(0, P_dis_max))
    model.P_buy = Var(model.T, within=NonNegativeReals)
    model.P_sell = Var(model.T, within=NonNegativeReals)
    model.P_FCR = Var(model.T, within=NonNegativeReals, bounds=(0, fcr_upper_bound))

    # State variables
    model.SOC = Var(model.T, within=NonNegativeReals, bounds=(0, E_bat_max))
    model.SOH = Var(model.T, within=NonNegativeReals, bounds=(0, 1.0))
    model.E_bat = Var(model.T, within=NonNegativeReals, bounds=(0, E_bat_max))

    # Peak auxiliary
    model.P_peak = Var(within=NonNegativeReals)

    # Binaries for linearizing charging and buying balances
    model.u_ch = Var(model.T, within=Binary)
    model.u_buy = Var(model.T, within=Binary)
    model.u_FCR = Var(model.T, within=Binary)
    
    # Parameters 
    model.C_buy = Param(model.T, initialize=lambda m, t: C_buy[t])
    model.C_sell = Param(model.T, initialize=lambda m, t: C_sell[t])
    model.D = Param(model.T, initialize=lambda m, t: D[t])
    model.E_bat_max = Param(initialize=E_bat_max)
    model.I0 = Param(initialize=I0)
    model.C_peak = Param(initialize=C_peak)
    model.P_ch_max = Param(initialize=P_ch_max)
    model.P_dis_max = Param(initialize=P_dis_max)
    model.eta_ch = Param(initialize=eta_ch)
    model.eta_dis = Param(initialize=eta_dis)
    model.delta_t = Param(initialize=delta_t)
    model.a = Param(initialize=a)
    model.b = Param(initialize=b)
    model.c = Param(initialize=c)
    model.FCR_signal = Param(model.T, initialize=lambda m, t: FCR_signal[t])
    model.FCR_active = Param(model.T, initialize=lambda m, t: 1 if abs(FCR_signal[t]) > 0 else 0, within=Binary)
    model.block_of_t = Param(model.T, initialize=lambda m, t: block_map[t], within=NonNegativeIntegers)

    # Define a param that stays at 1 after first FCR_active in block until block ends, then resets
    def fcr_active_cumulative_init(m, t):
        if t == 0:
            return m.FCR_active[0]
        block = m.block_of_t[t]
        prev_t = t - 1
        if m.block_of_t[prev_t] != block:
            # New block, reset cumulative
            return m.FCR_active[t]
        else:
            # In same block, cumulative OR
            return max(value(m.FCR_active[prev_t]), value(m.FCR_active[t]))
    model.FCR_active_cumulative = Param(
        model.T,
        initialize=fcr_active_cumulative_init,
        within=Binary
    )

    def fcr_block_rule(m, t):
        if t == 0:
            return Constraint.Skip

        if (t) % 16 != 0:
            return m.u_FCR[t] == m.u_FCR[t-1]
        else:
            # At t = 0, 16, 32, ... we allow changes → no constraint
            return Constraint.Skip

    model.fcr_block_rule = Constraint(model.T, rule=fcr_block_rule)
    
    def soh_init_rule(m):
        return m.SOH[0] == SOH0
    model.SOH_init = Constraint(rule=soh_init_rule)

    def soc_init_rule(m):
        return m.SOC[0] == SOC0*E_bat_max
    model.SOC_init = Constraint(rule=soc_init_rule)

    def soc_end_rule(m): # WHY?
        return m.SOC[T] == SOC0*E_bat_max
    model.SOC_end = Constraint(rule=soc_end_rule)

    # Charge balance: SOC_{t+1} = SOC_t + (eta_ch*P_ch - 1/eta_dis * P_dis) * dt / E0
    def soc_balance_rule(m, t):
        return m.SOC[t+1] == m.SOC[t] + (m.eta_ch*m.P_ch[t] - (1.0/m.eta_dis)*m.P_dis[t] + m.P_FCR[t] * m.FCR_signal[t]) * m.delta_t
    model.SOC_balance = Constraint(model.Tstep, rule=soc_balance_rule)

    # Limit SOC by current maximum capacity
    def soc_max_limit_rule(m, t):
        return m.SOC[t] <= m.E_bat[t] - eta_ch*fcr_upper_bound*delta_t*(1 - m.FCR_active_cumulative[t])
    model.SOC_max_limit = Constraint(model.T, rule=soc_max_limit_rule)

    # Limit SOC by current maximum capacity
    def soc_min_limit_rule(m, t):
        return m.SOC[t] >= fcr_upper_bound*delta_t*(1 - m.FCR_active_cumulative[t])/eta_dis
    model.SOC_min_limit = Constraint(model.T, rule=soc_min_limit_rule)

    # Power balance: P_dis + P_buy = D + P_ch
    def power_balance_rule(m, t):
        return m.P_dis[t] + m.P_buy[t] == m.D[t] + m.P_ch[t] + m.P_sell[t]
    model.Power_balance = Constraint(model.T, rule=power_balance_rule)

    # Upper capacity limit
    def upper_capacity_rule(m, t):
        return m.P_ch[t]*delta_t <= m.E_bat[t] - m.SOC[t]
    model.Upper_capacity = Constraint(model.T, rule=upper_capacity_rule)

    # Lower capacity limit
    def lower_capacity_rule(m, t):
        return m.P_dis[t]*delta_t <= m.SOC[t]
    model.Lower_capacity = Constraint(model.T, rule=lower_capacity_rule)

    # Charging limit
    def ch_bin_rule(m, t):
        return m.P_ch[t] <= m.u_ch[t] * P_ch_max * (1 - m.u_FCR[t]*m.FCR_active[t])
    model.Charge_binary = Constraint(model.T, rule=ch_bin_rule)

    # Discharging limit
    def dis_bin_rule(m, t):
        return m.P_dis[t] <= (1 - m.u_ch[t]) * P_dis_max *  (1 - m.u_FCR[t]*m.FCR_active[t])
    model.Distcharge_binary = Constraint(model.T, rule=dis_bin_rule)

    def fcr_power_limit(m,t):
        return m.P_FCR[t] == fcr_upper_bound*m.u_FCR[t]*m.FCR_active[t]
    model.FCR_power = Constraint(model.T, rule = fcr_power_limit)

    # Buying limit
    def buy_bin_rule(m, t):
        return m.P_buy[t] <= m.u_buy[t] * P_buy_max
    model.Buy_binary = Constraint(model.T, rule=buy_bin_rule)

    # Selling limit
    def sell_bin_rule(m, t):
        return m.P_sell[t] <= (1 - m.u_buy[t]) * P_sell_max
    model.Sell_binary = Constraint(model.T, rule=sell_bin_rule)

    # Peak definition: P_buy[t] <= P_peak
    def peak_rule(m, t):
        return m.P_buy[t] <= m.P_peak
    model.Peak_def = Constraint(model.T, rule=peak_rule)

    # SOH_{t+1} = SOH_t - (a + b*(P_dis + P_ch) + c*SOH_t)
    def soh_update_rule(m, t):
        return m.SOH[t+1] == m.SOH[t] - (m.a + m.b*(m.P_dis[t] + m.P_ch[t])/V_bat + m.c*m.SOH[t])*delta_t*3600 # To get seconds
    model.SOH_update = Constraint(model.Tstep, rule=soh_update_rule)

    # Update declining battery capacity
    def battery_capacity_update(m, t):
        return m.E_bat[t] == m.E_bat_max*m.SOH[t]
    model.battery_capacity_update = Constraint(model.T, rule=battery_capacity_update)

    # Objective: minimize sum_t C_t * P_buy_t + SOH_T * I0 + C_peak * max_t P_buy_t
    def objective_rule(m):
        cost =  sum((m.C_buy[t] * m.P_buy[t] - m.C_sell[t] * m.P_sell[t])*delta_t for t in m.T) + m.P_peak * m.C_peak + I0*(SOH0 - m.SOH[T])/(SOH0 - 0.8)
        revenue = sum(C_FCR[t] * m.u_FCR[t] for t in m.T)
        return cost - revenue
    model.Obj = Objective(rule=objective_rule, sense=minimize)

    return model
 
#laod data and define Parameters
fcr_prices, day_ahead, load = load_and_process_data(this_file, specific_load='LG 18')
peak_tarif = 192.66

#Ratio of BTM to FTM capacity
btm_ratio = 0.4
N = 7*24*4
T = len(day_ahead) - 1
num_steps = T+1
delta_t = 0.25
block_size = int(round(4.0 / delta_t))

#total max power feeding or drawing from grid
p_buy_max = 500 #kW
p_sell_max = 500 #kW
# Luna 2000-215 data:
E_2000 = 215
C_2000 = 108
I_2000 = 73000
eta_2000 = 0.974
N_bat = 1
min_fcr_bid = 20 #20kw parameter for later with multiple batteries 1MW
FCR_signal = generate_fcr_activation_profile(
        num_steps=num_steps,
        block_size=block_size,
        eta_ch=eta_2000,
        eta_dis=eta_2000,
        seed=42
    )


model_fixed_rates = build_split_battery_model(
    T=T,
    C_buy=day_ahead,
    C_sell=day_ahead,
    D=load,
    E_bat_max=E_2000*N_bat,
    I0=I_2000*N_bat,
    C_peak=peak_tarif,
    P_buy_max=p_buy_max,
    P_sell_max=p_sell_max,
    P_ch_max=C_2000*N_bat,
    P_dis_max=C_2000*N_bat,
    eta_ch=eta_2000,
    eta_dis=eta_2000,
    delta_t=0.25,
    SOC0=0.5,
    SOH0=1.0,
    a=10e-11,
    b=10e-11,
    c=10e-10,
    V_bat = 777,
    C_FCR=fcr_prices,
    FCR_signal=FCR_signal,
    btm_ratio=btm_ratio,
    min_fcr_bid=min_fcr_bid
)
# Battery model, all values in kWh or kW
model = build_battery_model(
    T=T,
    C_buy=day_ahead,
    C_sell=day_ahead,
    D=load,
    E_bat_max=E_2000*N_bat,
    I0=I_2000*N_bat,
    C_peak=peak_tarif,
    P_buy_max=100000,
    P_sell_max=100000,
    P_ch_max=C_2000*N_bat,
    P_dis_max=C_2000*N_bat,
    eta_ch=eta_2000,
    eta_dis=eta_2000,
    delta_t=0.25,
    SOC0=0.5,
    SOH0=1.0,
    a=10e-11,
    b=10e-11,
    c=10e-10,
    V_bat = 777,
    C_FCR=fcr_prices,
    FCR_signal=FCR_signal
)

solver = SolverFactory('gurobi')
results = solver.solve(model_fixed_rates, tee=True)
model_fixed_rates.display()

# Extract data from model
plot_comprehensive_results(model_fixed_rates, delta_t=0.25, E_nom=E_2000, c_peak=peak_tarif)

# %%
