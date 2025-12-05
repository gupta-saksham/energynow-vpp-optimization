from pyomo.environ import *
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
from pathlib import Path
this_file = Path(__file__).parent

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

def load_day_ahead_prices():
    day_ahead_df = pd.read_csv(this_file / "Day ahead.csv")
    day_ahead = day_ahead_df["Germany/Luxembourg [Eur/MWh]"]/1000 # to get EUR/KWh
    return day_ahead

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


day_ahead = load_day_ahead_prices() #EUR/kWh
fcr_prices = load_fcr_prices() #EUR/kW

load_df = pd.read_csv(this_file / "Load profile.csv")
load = load_df["LG 18"]*3

peak_tarif = 192.66

N = 7*24*4
T = len(day_ahead) - 1
num_steps = T+1
delta_t = 0.25
block_size = int(round(4.0 / delta_t))

# Luna 2000-215 data:
E_2000 = 215
C_2000 = 108
I_2000 = 73000
eta_2000 = 0.974
N_bat = 1

FCR_signal = generate_fcr_activation_profile(
        num_steps=num_steps,
        block_size=block_size,
        eta_ch=eta_2000,
        eta_dis=eta_2000,
        seed=42
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
results = solver.solve(model, tee=True)
model.display()

# Extract data from model

