# src/engineering_model.py

from src.config import AIR_DENSITY, ROTOR_AREA, POWER_COEFFICIENT

def calculate_power(v):
    return 0.5 * AIR_DENSITY * ROTOR_AREA * (v ** 3) * POWER_COEFFICIENT

def apply_physical_model(df):
    df["theoretical_power"] = df["wind_speed"].apply(calculate_power)
    return df