import numpy as np
import pandas as pd

def generate_synthetic_dataset(num_suppliers=50, output_file="data/synthetic_data.csv"):
    """
    Generates a lightweight synthetic dataset mirroring the Bloomberg schema.
    Used exclusively for public GitHub repository execution.
    """
    np.random.seed(42)  # Ensures reproducible results
    
    tickers = [f"SUPP_{i:02d}_US_Equity" for i in range(1, num_suppliers + 1)]
    names = [f"Supplier Company {i}" for i in range(1, num_suppliers + 1)]
    
    # 1. Weights: Revenue dependency percentages (0.01 to 0.90)
    rev_dependency = np.random.uniform(0.01, 0.90, num_suppliers)
    
    # 2. Financial Risk: 1Y Default Probabilities
    default_prob = np.random.uniform(0.00001, 0.05, num_suppliers)
    # Inject 15% missing values (NaNs)
    nan_mask_def = np.random.rand(num_suppliers) < 0.15
    default_prob[nan_mask_def] = np.nan
    
    # 3. Operational Risk: ESG Scores (1.0 to 8.5)
    esg_scores = np.random.uniform(1.0, 8.5, num_suppliers)
    # Inject 20% missing values (NaNs)
    nan_mask_esg = np.random.rand(num_suppliers) < 0.20
    esg_scores[nan_mask_esg] = np.nan
    
    # 4. Geopolitical Risk: Countries of Domicile
    countries = ["US", "TW", "KR", "CN", "JP"]
    country_domicile = np.random.choice(countries, num_suppliers)
    
    df = pd.DataFrame({
        "Supplier_Ticker": tickers,
        "Supplier_Name": names,
        "Revenue_Dependency": rev_dependency,
        "1Y_Default_Prob": default_prob,
        "ESG_Score": esg_scores,
        "Country_Domicile": country_domicile
    })
    
    df.to_csv(output_file, index=False)
    print(f"Synthetic dataset successfully saved to '{output_file}' with shape {df.shape}")

if __name__ == "__main__":
    generate_synthetic_dataset()