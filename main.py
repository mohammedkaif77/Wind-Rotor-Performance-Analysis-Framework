import os
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib

from src.data_generator import generate_wind_dataset


def main():
    print("Starting Wind Rotor ML Pipeline...")

    # 1️⃣ Generate Dataset
    generate_wind_dataset()

    # 2️⃣ Load Dataset
    data_path = "data/wind_data.csv"
    df = pd.read_csv(data_path)

    X = df.drop("power_output", axis=1)
    y = df["power_output"]

    # 3️⃣ Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 4️⃣ Train Model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # 5️⃣ Predictions
    y_pred = model.predict(X_test)

    # 6️⃣ Version-Safe RMSE Calculation
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    print(f"RMSE: {rmse:.4f}")
    print(f"R2 Score: {r2:.4f}")

    # 7️⃣ Performance Gate
    if r2 < 0.75:
        raise ValueError("Model performance below acceptable threshold!")

    # 8️⃣ Save Model
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, "models/wind_model.pkl")

    # 9️⃣ Save Report
    os.makedirs("reports", exist_ok=True)
    with open("reports/model_report.txt", "w") as f:
        f.write("Wind Rotor Model Performance Report\n")
        f.write("-----------------------------------\n")
        f.write(f"RMSE: {rmse:.4f}\n")
        f.write(f"R2 Score: {r2:.4f}\n")

    print("Pipeline completed successfully.")


if __name__ == "__main__":
    main()
