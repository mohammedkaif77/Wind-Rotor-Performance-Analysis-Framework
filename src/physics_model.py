import numpy as np

def calculate_theoretical_power(rho, radius, wind_speed):
    """
    Calculates theoretical wind turbine power.
    P = 0.5 * rho * A * V^3
    """
    area = np.pi * radius**2
    power = 0.5 * rho * area * wind_speed**3
    return power