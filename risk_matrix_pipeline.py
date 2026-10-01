import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def load_data(weights_path, risk_path, is_synthetic=False):
    """Loads raw or synthetic data based on execution mode."""
    if is_synthetic:
        return pd.read_csv(weights_path)
    
    weights_df = pd.read_csv(weights_path)
    risk_df = pd.read_csv(risk_path)
    return pd.merge(weights_df, risk_df, on="Supplier_Ticker")

def preprocess_and_impute(df):
    """
    Handles NaNs with aggressive ESG penalization, maps categorical data, 
    and normalizes risk vectors.
    """
    df = df.copy()
    
    def_col = "1Y_Default_Prob"
    # Dynamically find the ESG column regardless of earlier naming conventions
    esg_col = "ESG_Score" if "ESG_Score" in df.columns else "ESG_Disclosure_Score"
    
    # 1. Aggressive ESG Imputation: Penalize missing data with 0
    df[esg_col] = pd.to_numeric(df[esg_col], errors="coerce")
    missing_esg = df[esg_col].isnull().sum()
    df[esg_col] = df[esg_col].fillna(0.0)
    print(f"-> Data Governance: Penalized {missing_esg} suppliers with 0 for missing {esg_col}.")
    
    # 2. Median Imputation for Default Probability
    df[def_col] = pd.to_numeric(df[def_col], errors="coerce")
    if df[def_col].isnull().sum() > 0:
        median_def = df[def_col].median()
        df[def_col] = df[def_col].fillna(median_def)
        print(f"-> Imputed missing values in '{def_col}' using median: {median_def:.6f}")
        
    # 3. Categorical Mapping: Country of Domicile -> Geopolitical Risk Index [0, 1]
    geo_risk_map = {"US": 0.10, "JP": 0.15, "KR": 0.25, "TW": 0.40, "CN": 0.60}
    df["Geopolitical_Risk"] = df["Country_Domicile"].map(lambda c: geo_risk_map.get(str(c).strip(), 0.30))
    
    # 4. Min-Max Normalization to [0, 1] scale
    # Financial Risk: Higher default probability = higher risk
    min_def, max_def = df[def_col].min(), df[def_col].max()
    df["Financial_Risk_Norm"] = (df[def_col] - min_def) / (max_def - min_def + 1e-9)
    
    # Operational Risk: Higher ESG score = LOWER risk (Inverted)
    min_esg, max_esg = df[esg_col].min(), df[esg_col].max()
    df["Operational_Risk_Norm"] = 1.0 - ((df[esg_col] - min_esg) / (max_esg - min_esg + 1e-9))
    
    # Normalize Dependency Weights W so they sum to 1.0
    df["Revenue_Dependency"] = pd.to_numeric(df["Revenue_Dependency"], errors="coerce").fillna(0)
    df["Dependency_Weight_Norm"] = df["Revenue_Dependency"] / df["Revenue_Dependency"].sum()
    
    return df

def calculate_supply_chain_vulnerability(df):
    """Executes Linear Algebra Dot Products using NumPy."""
    W = df["Dependency_Weight_Norm"].to_numpy().reshape(1, -1)
    
    R_cols = ["Financial_Risk_Norm", "Operational_Risk_Norm", "Geopolitical_Risk"]
    R = df[R_cols].to_numpy()
    
    K = np.array([0.40, 0.30, 0.30]).reshape(-1, 1)
    
    E = np.dot(W, R)
    V = np.dot(E, K)
    
    summary = {
        "Financial_Risk_Exposure": E[0, 0],
        "Operational_Risk_Exposure": E[0, 1],
        "Geopolitical_Risk_Exposure": E[0, 2],
        "Final_Vulnerability_Score": V[0, 0]
    }
    return summary, W, R, K, df, R_cols

def generate_visualizations(df, R, E, R_cols, is_synthetic):
    """Generates and saves the Risk Matrix Heatmap and Exposure Bar Chart."""
    os.makedirs("outputs", exist_ok=True)
    prefix = "synthetic" if is_synthetic else "apple_real"
    
    # 1. Risk Matrix Heatmap
    plt.figure(figsize=(8, 10))
    # Using the supplier tickers as the y-axis labels
    sns.heatmap(R, annot=True, cmap="YlOrRd", xticklabels=["Financial", "Operational", "Geopolitical"], 
                yticklabels=df["Supplier_Ticker"].tolist(), cbar_kws={'label': 'Normalized Risk Score (0-1)'})
    plt.title("Supplier Risk Matrix ($R$)")
    plt.tight_layout()
    plt.savefig(f"outputs/{prefix}_risk_matrix_heatmap.png", dpi=300)
    plt.close()
    
    # 2. Exposure Vector Bar Chart
    plt.figure(figsize=(8, 5))
    categories = ["Financial", "Operational", "Geopolitical"]
    colors = ['#1f77b4', '#ff7f0e', '#d62728']
    plt.bar(categories, E.flatten(), color=colors, edgecolor='black')
    plt.ylim(0, 1.0)
    plt.title("Supply Chain Risk Exposure Vector ($E = W \cdot R$)")
    plt.ylabel("Weighted Exposure (0 to 1)")
    
    # Add data labels on top of bars
    for i, v in enumerate(E.flatten()):
        plt.text(i, v + 0.02, f"{v:.4f}", ha='center', fontweight='bold')
        
    plt.tight_layout()
    plt.savefig(f"outputs/{prefix}_exposure_vector.png", dpi=300)
    plt.close()
    print(f"-> Visualizations successfully saved to the 'outputs/' directory.")

if __name__ == "__main__":
    # FLIP TO False TO RUN REAL APPLE DATA FOR YOUR PDF REPORT
    # FLIP TO True BEFORE PUSHING TO GITHUB
    USE_SYNTHETIC = False
    
    if USE_SYNTHETIC:
        print("=== EXECUTION MODE: SYNTHETIC DATA ===")
        df_raw = load_data("data/synthetic_data.csv", None, is_synthetic=True)
    else:
        print("=== EXECUTION MODE: REAL APPLE DATA ===")
        df_raw = load_data(
            "data/raw_supplier_weights.csv", 
            "data/raw_supplier_risk_factors.csv", 
            is_synthetic=False
        )
        
    df_clean = preprocess_and_impute(df_raw)
    results, W_vec, R_mat, K_vec, df_processed, R_cols = calculate_supply_chain_vulnerability(df_clean)
    
    # Generate the Charts
    generate_visualizations(df_processed, R_mat, np.dot(W_vec, R_mat), R_cols, USE_SYNTHETIC)
    
    print("\n--- SUPPLY CHAIN RISK RESULTS ---")
    for metric, score in results.items():
        print(f"{metric}: {score:.4f}")