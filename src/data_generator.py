import numpy as np
import pandas as pd
import os

def generate_wind_dataset(samples=1000):
    """
    Generates synthetic wind turbine dataset
    using aerodynamic power equation with noise.
    """

    np.random.seed(42)

    wind_speed = np.random.uniform(3, 25, samples)  # m/s
    air_density = np.random.uniform(1.1, 1.3, samples)  # kg/m^3
    rotor_radius = np.random.uniform(20, 60, samples)  # meters
    temperature = np.random.uniform(-10, 40, samples)  # °C

    # Physics-based power calculation
    area = np.pi * rotor_radius**2
    theoretical_power = 0.5 * air_density * area * wind_speed**3

    # Add real-world noise (turbulence, inefficiency)
    noise = np.random.normal(0, theoretical_power * 0.1)
    power_output = theoretical_power + noise

    df = pd.DataFrame({
        "wind_speed": wind_speed,
        "air_density": air_density,
        "rotor_radius": rotor_radius,
        "temperature": temperature,
        "power_output": power_output
    })

    os.makedirs("data", exist_ok=True)
    df.to_csv("data/wind_data.csv", index=False)

    print("Synthetic wind dataset generated successfully.")