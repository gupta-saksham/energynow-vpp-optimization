import requests
import pandas as pd
import time
import numpy as np
from datetime import date, timedelta

# --- KONFIGURATION ---
YEAR = 2024
FILE_FREQ = "frequency_2024_1min.csv"
FILE_MODEL = "FCR_Energy_2024_15min.csv"

# FCR Parameter
DEADBAND = 0.010   # 10 mHz
FULL_POWER = 0.200 # 200 mHz

def calculate_fcr_factors(freq_series):
    """Berechnet die FCR-Belastung (-1 bis 1) auf Sekundenbasis"""
    delta_f = freq_series - 50.0
    slope = FULL_POWER - DEADBAND
    
    conditions = [
        (delta_f > DEADBAND) & (delta_f < FULL_POWER),   # Positiv Regelbereich
        (delta_f >= FULL_POWER),                          # Positiv Sättigung
        (delta_f < -DEADBAND) & (delta_f > -FULL_POWER), # Negativ Regelbereich
        (delta_f <= -FULL_POWER),                         # Negativ Sättigung
        (delta_f.abs() <= DEADBAND)                       # Totband
    ]
    
    choices = [
        (delta_f - DEADBAND) / slope,
        1.0,
        (delta_f + DEADBAND) / slope,
        -1.0,
        0.0
    ]
    
    return pd.Series(np.select(conditions, choices, default=0.0), index=freq_series.index)

def fetch_and_process_all_2024_final():
    base_url = "https://api.energy-charts.info/frequency"
    
    # # 1. Header schreiben
    # with open(FILE_FREQ, 'w', newline='', encoding='utf-8') as f:
    #     f.write('Datum (MEZ),"Frequenz (Netzfrequenz)"\n')
        
    # with open(FILE_MODEL, 'w', newline='', encoding='utf-8') as f:
    #     f.write('Time_Slot_Start,FCR_Power_Factor_Up_Sum,FCR_Power_Factor_Down_Sum\n')

    start_date = date(2024, 9, 9)
    end_date = date(YEAR, 12, 31)
    current_date = start_date

    print(f"Starte Prozess für {YEAR}...")

    while current_date <= end_date:
        day_str = current_date.strftime("%Y-%m-%d")
        
        params = {
            "country": "de",
            "start": day_str,
            "end": day_str 
        }

        try:
            print(f"Verarbeite: {day_str} ...", end="\r")
            
            response = requests.get(base_url, params=params)
            
            if response.status_code == 200:
                json_resp = response.json()
                
                if 'unix_seconds' in json_resp and 'data' in json_resp:
                    
                    # --- BASIS DATAFRAME ---
                    df = pd.DataFrame({
                        'timestamp': pd.to_datetime(json_resp['unix_seconds'], unit='s'),
                        'frequency': json_resp['data']
                    })
                    
                    df['timestamp'] = df['timestamp'].dt.tz_localize('UTC').dt.tz_convert('Europe/Berlin')
                    df.set_index('timestamp', inplace=True)
                    
                    # --- TEIL 1: FREQUENZ (Minutenwerte) ---
                    df_freq_1min = df[['frequency']].resample('1min').mean()
                    
                    save_freq = pd.DataFrame()
                    save_freq['Datum (MEZ)'] = df_freq_1min.index.map(lambda x: x.isoformat())
                    # WICHTIG: .values verwenden!
                    save_freq['Frequenz (Netzfrequenz)'] = df_freq_1min['frequency'].round(4).values
                    
                    if not save_freq.empty:
                        save_freq.dropna().to_csv(FILE_FREQ, mode='a', header=False, index=False)
                    
                    # --- TEIL 2: MODELL (15 Min Summen) ---
                    df['p_factor'] = calculate_fcr_factors(df['frequency'])
                    df['p_up'] = df['p_factor'].apply(lambda x: x if x > 0 else 0)
                    df['p_down'] = df['p_factor'].apply(lambda x: abs(x) if x < 0 else 0)
                    
                    df_model_15min = df[['p_up', 'p_down']].resample('15min').sum()
                    
                    save_model = pd.DataFrame()
                    save_model['Time_Slot_Start'] = df_model_15min.index.map(lambda x: x.isoformat())
                    
                    # --- HIER WAR DER FEHLER ---
                    # Wir nutzen .values, um Index-Konflikte zu vermeiden
                    save_model['FCR_Power_Factor_Up_Sum'] = df_model_15min['p_up'].round(4).values
                    save_model['FCR_Power_Factor_Down_Sum'] = df_model_15min['p_down'].round(4).values
                    
                    if not save_model.empty:
                        save_model.to_csv(FILE_MODEL, mode='a', header=False, index=False)
                
            time.sleep(0.3) 
            
        except Exception as e:
            print(f"\nFehler bei {day_str}: {e}")

        current_date += timedelta(days=1)

    print(f"\n\nFertig! Bitte prüfen Sie jetzt die CSV Dateien.")

if __name__ == "__main__":
    fetch_and_process_all_2024_final()