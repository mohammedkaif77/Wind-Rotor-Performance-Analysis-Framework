import os
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from src.data_generator import generate_wind_dataset


def main():

    # Ensure reports directory exists
    os.makedirs("reports", exist_ok=True)

    # Generate dataset
    df = generate_wind_dataset()

    # Features & Target
    X = df[["wind_speed", "air_density", "rotor_radius", "temperature"]]
    y = df["power_output"]

    # Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Train Model
    model = RandomForestRegressor(random_state=42)
    model.fit(X_train, y_train)

    # Predictions
    y_pred = model.predict(X_test)

    # Evaluation
    r2 = r2_score(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred) ** 0.5  # Manually compute RMSE

    print(f"R2 Score: {r2}")
    print(f"RMSE: {rmse}")

    # 🚨 Performance Gate
    if r2 < 0.95:
        raise ValueError("Model performance below acceptable threshold (0.95)")

    # Save Model
    joblib.dump(model, "reports/model.pkl")

    print("Model saved successfully.")


if __name__ == "__main__":
    main()
