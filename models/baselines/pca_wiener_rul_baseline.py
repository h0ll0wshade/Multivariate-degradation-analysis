import os
import zipfile
import urllib.request
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler

# Set random seed
np.random.seed(42)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "datasets" / "CMAPSS"
FIGURES_DIR = PROJECT_ROOT / "results" / "images"

# ==========================================
# 1. DOWNLOAD & LOAD NASA C-MAPSS DATASET
# ==========================================
def load_local_cmapss(
    train_path=DATA_DIR / "train_FD001.txt",
    test_path=DATA_DIR / "test_FD001.txt",
    rul_path=DATA_DIR / "RUL_FD001.txt",
):
    """Loads C-MAPSS FD001 dataset from local files in the working directory."""
    col_names = [
        "unit",
        "cycle",
        "op_setting_1",
        "op_setting_2",
        "op_setting_3",
    ] + [f"sensor_{i}" for i in range(1, 22)]

    print("Loading local C-MAPSS FD001 dataset...")
    train_df = pd.read_csv(train_path, sep=r"\s+", header=None, names=col_names)
    test_df = pd.read_csv(test_path, sep=r"\s+", header=None, names=col_names)
    rul_df = pd.read_csv(rul_path, sep=r"\s+", header=None, names=["true_rul"])

    print(
        f"Data Loaded: {train_df['unit'].nunique()} Training Engines, {test_df['unit'].nunique()} Testing Engines.\n"
    )
    return train_df, test_df, rul_df

# ==========================================
# 2. FEATURE SELECTION & HEALTH INDICATOR (HI)
# ==========================================
def build_health_indicator(train_df, test_df):
    """
    1. Removes non-informative/constant sensors in FD001.
    2. Fuses active multivariate sensors into a 1D Health Indicator via PCA.
    """
    # Active sensors showing degradation trends in FD001
    active_sensors = ['sensor_2', 'sensor_3', 'sensor_4', 'sensor_7', 'sensor_8', 
                      'sensor_11', 'sensor_12', 'sensor_13', 'sensor_15', 'sensor_17', 'sensor_20', 'sensor_21']

    scaler = MinMaxScaler()
    pca = PCA(n_components=1)

    # Scale and apply PCA on training set
    train_scaled = scaler.fit_transform(train_df[active_sensors])
    train_df['HI'] = pca.fit_transform(train_scaled)

    # Scale and apply PCA on testing set
    test_scaled = scaler.transform(test_df[active_sensors])
    test_df['HI'] = pca.transform(test_scaled)

    # Ensure HI increases strictly as engine degrades (flip sign if PCA yielded negative correlation)
    unit1_train = train_df[train_df['unit'] == 1]['HI'].values
    if unit1_train[-1] < unit1_train[0]:
        train_df['HI'] = -train_df['HI']
        test_df['HI'] = -test_df['HI']

    # Shift baseline so HI starts near 0.0 for healthy engines
    min_hi = train_df['HI'].min()
    train_df['HI'] -= min_hi
    test_df['HI'] -= min_hi

    return train_df, test_df

# ==========================================
# 3. WIENER MODEL FIT & RUL PREDICTION
# ==========================================
class CMAPSSWienerModel:
    def __init__(self, failure_threshold):
        self.w = failure_threshold
        self.mu_0 = None
        self.sigma_0_sq = None
        self.sigma = None

    def fit_fleet(self, train_df):
        """
        Estimates population parameters (mu_0, sigma_0^2, sigma) from historical training engines.
        """
        unit_mus = []
        all_increments = []

        for unit_id, group in train_df.groupby('unit'):
            hi_path = group['HI'].values
            cycles = group['cycle'].values
            
            # Unit drift estimation: (HI_end - HI_start) / max_cycle
            drift_i = (hi_path[-1] - hi_path[0]) / cycles[-1]
            unit_mus.append(drift_i)

            # Process increments
            increments = np.diff(hi_path)
            all_increments.extend(increments)

        self.mu_0 = np.mean(unit_mus)
        self.sigma_0_sq = np.var(unit_mus, ddof=1)
        self.sigma = np.std(all_increments)

        print("=== Offline Parameter Estimation (C-MAPSS FD001) ===")
        print(f"Population Mean Drift (μ_0): {self.mu_0:.6f}")
        print(f"Drift Variance (σ_0²):       {self.sigma_0_sq:.8f}")
        print(f"Diffusion Volatility (σ):    {self.sigma:.6f}")
        print(f"Failure Threshold (ω):       {self.w:.4f}\n")

    def predict_unit_rul(self, t_obs, x_obs):
        """
        Bayesian update of unit drift and mode RUL calculation.
        """
        prior_prec = 1.0 / self.sigma_0_sq
        data_prec = t_obs / (self.sigma ** 2)

        post_var = 1.0 / (prior_prec + data_prec)
        post_mu = post_var * ((self.mu_0 / self.sigma_0_sq) + (x_obs / (self.sigma ** 2)))

        dist_to_fail = self.w - x_obs
        if dist_to_fail <= 0:
            return 0.0

        # Mode of First Passage Time (Inverse Gaussian) Distribution
        predicted_rul = dist_to_fail / post_mu
        return max(0.0, predicted_rul)

# ==========================================
# 4. EXECUTION & EVALUATION
# ==========================================
if __name__ == "__main__":
    # 1. Download & Preprocess Data
    train_df, test_df, true_rul_df = load_local_cmapss()
    train_df, test_df = build_health_indicator(train_df, test_df)

    # Define Failure Threshold based on 95th percentile HI at end-of-life in training set
    eol_hi_values = train_df.groupby('unit')['HI'].last().values
    failure_threshold = np.percentile(eol_hi_values, 90)

    # 2. Fit Wiener Degradation Model
    model = CMAPSSWienerModel(failure_threshold=failure_threshold)
    model.fit_fleet(train_df)

    # 3. Predict RUL for all Test Engines
    predicted_ruls = []
    actual_ruls = true_rul_df['true_rul'].values
    test_units = test_df['unit'].unique()

    for unit_id in test_units:
        unit_data = test_df[test_df['unit'] == unit_id]
        t_obs = unit_data['cycle'].max()
        x_obs = unit_data['HI'].iloc[-1]

        pred_rul = model.predict_unit_rul(t_obs, x_obs)
        
        # Clip RUL at 125 cycles (Standard evaluation practice on C-MAPSS)
        pred_rul = min(125.0, pred_rul)
        predicted_ruls.append(pred_rul)

    actual_ruls_clipped = np.minimum(125.0, actual_ruls)

    # 4. Compute Metrics
    rmse = np.sqrt(np.mean((np.array(predicted_ruls) - actual_ruls_clipped) ** 2))
    mae = np.mean(np.abs(np.array(predicted_ruls) - actual_ruls_clipped))

    print(f"=== Model Performance on C-MAPSS FD001 Test Set ===")
    print(f"Root Mean Squared Error (RMSE): {rmse:.2f} cycles")
    print(f"Mean Absolute Error (MAE):     {mae:.2f} cycles\n")

    # ==========================================
    # 5. VISUALIZATION FOR PRESENTATION
    # ==========================================
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=120)

    # Subplot 1: PCA Health Indicator Trajectories
    sample_units = [1, 2, 3, 4, 5]
    for uid in sample_units:
        unit_df = train_df[train_df['unit'] == uid]
        axes[0].plot(unit_df['cycle'], unit_df['HI'], alpha=0.7, label=f"Engine #{uid}")
    axes[0].axhline(y=failure_threshold, color='r', linestyle='--', linewidth=2, label="Failure Threshold (ω)")
    axes[0].set_title("C-MAPSS FD001 Health Indicator (PCA Fusion)", fontweight='bold')
    axes[0].set_xlabel("Flight Cycles")
    axes[0].set_ylabel("Constructed Health Indicator (HI)")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Subplot 2: Predicted RUL vs True RUL (Parity Plot)
    axes[1].scatter(actual_ruls_clipped, predicted_ruls, color='blue', alpha=0.6, edgecolors='k')
    axes[1].plot([0, 125], [0, 125], 'r--', linewidth=2, label="Perfect Ideal Prediction")
    axes[1].set_title(f"RUL Prediction Accuracy (RMSE: {rmse:.1f} Cycles)", fontweight='bold')
    axes[1].set_xlabel("True Remaining Useful Life (Cycles)")
    axes[1].set_ylabel("Predicted Remaining Useful Life (Cycles)")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(FIGURES_DIR / "cmapss_rul_results.png", dpi=300)
    plt.show()
