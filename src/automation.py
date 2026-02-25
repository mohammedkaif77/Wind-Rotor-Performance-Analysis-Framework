import os
import numpy as np

from src.data_loader import load_data
from src.preprocessing import preprocess_data
from src.ml_model import train_model, predict
from src.evaluation import evaluate_model
from src.validation import cross_validate_model
from src.visualization import plot_actual_vs_predicted, plot_residuals
from src.physics_model import calculate_theoretical_power


def run_pipeline():
    print("Starting Wind Rotor Performance Analysis Framework...\n")

    os.makedirs("reports", exist_ok=True)

    df = load_data("data/wind_data.csv")
    target_column = "power_output"

    # ---------------- ML Pipeline ----------------
    X_train, X_test, y_train, y_test = preprocess_data(df, target_column)

    model = train_model(X_train, y_train)
    y_pred_ml = predict(model, X_test)

    ml_metrics = evaluate_model(y_test, y_pred_ml)

    print("ML Model Performance:")
    for key, value in ml_metrics.items():
        print(f"{key}: {value:.4f}")

    # ---------------- Physics Model ----------------
    physics_power = calculate_theoretical_power(
        df["air_density"],
        df["rotor_radius"],
        df["wind_speed"]
    )

    physics_metrics = evaluate_model(df[target_column], physics_power)

    print("\nPhysics Model Performance:")
    for key, value in physics_metrics.items():
        print(f"{key}: {value:.4f}")

    # ---------------- Improvement Calculation ----------------
    improvement = (
        (physics_metrics["RMSE"] - ml_metrics["RMSE"])
        / physics_metrics["RMSE"]
    ) * 100

    print(f"\nML Improvement over Physics (RMSE Reduction): {improvement:.2f}%")

    # ---------------- Cross Validation ----------------
    X_full = df.drop(columns=[target_column])
    y_full = df[target_column]

    cv_results = cross_validate_model(model, X_full, y_full)

    print("\nCross Validation Results:")
    for key, value in cv_results.items():
        print(f"{key}: {value:.4f}")

    # ---------------- Visualizations ----------------
    plot_actual_vs_predicted(y_test, y_pred_ml)
    plot_residuals(y_test, y_pred_ml)

    # ---------------- Save Report ----------------
    with open("reports/performance_report.txt", "w") as f:
        f.write("Wind Rotor Performance Hybrid Analysis Report\n\n")

        f.write("ML Model Performance:\n")
        for key, value in ml_metrics.items():
            f.write(f"{key}: {value:.4f}\n")

        f.write("\nPhysics Model Performance:\n")
        for key, value in physics_metrics.items():
            f.write(f"{key}: {value:.4f}\n")

        f.write(f"\nML Improvement over Physics (RMSE Reduction): {improvement:.2f}%\n")

        f.write("\nCross Validation Results:\n")
        for key, value in cv_results.items():
            f.write(f"{key}: {value:.4f}\n")

    print("\nReport and visualizations saved in 'reports/' folder.")
    print("Framework Execution Completed Successfully.")