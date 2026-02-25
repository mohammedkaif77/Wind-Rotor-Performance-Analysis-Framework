import matplotlib.pyplot as plt
import os

def plot_actual_vs_predicted(y_true, y_pred, save_path="reports/actual_vs_predicted.png"):
    plt.figure()
    plt.scatter(y_true, y_pred)
    plt.xlabel("Actual Power")
    plt.ylabel("Predicted Power")
    plt.title("Actual vs Predicted Rotor Power")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_residuals(y_true, y_pred, save_path="reports/residual_plot.png"):
    residuals = y_true - y_pred

    plt.figure()
    plt.scatter(y_pred, residuals)
    plt.xlabel("Predicted Power")
    plt.ylabel("Residuals")
    plt.title("Residual Analysis")
    plt.axhline(y=0)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()