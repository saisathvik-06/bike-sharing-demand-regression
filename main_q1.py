import sys
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Adjust path to find sibling modules
sys.path.append(str(Path(__file__).parent.parent))

from q1_bike_sharing.dataset import BikeSharingDataset
from q1_bike_sharing.regression import ManualLinearRegression, PolynomialRegressor, QuadraticInteractionRegressor

def main():
    print("="*60)
    print("Question 1: Bike Sharing Demand Regression (From Scratch)")
    print("="*60)
    
    # 1. Setup Paths
    current_dir = Path(__file__).parent
    data_path = current_dir.parent.parent / "data" / "bike_sharing" / "train.csv"
    figures_dir = current_dir.parent.parent / "figures"
    results_dir = current_dir.parent.parent / "results"
    figures_dir.mkdir(exist_ok=True)
    results_dir.mkdir(exist_ok=True)
    
    # 2. Load Data
    print("\n[Data] Loading and preprocessing...")
    if not data_path.exists():
         print(f"Error: {data_path} not found. Please download train.csv.")
         return
         
    # Using 80-20 split
    dataset = BikeSharingDataset(data_path, test_size=0.2, random_state=42)
    X_train, X_test, y_train, y_test = dataset.load_and_preprocess()
    
    print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")
    print(f"Features: {dataset.get_feature_names()}")
    
    # 3. Model Training & Evaluation
    results = []
    
    # Define models to run
    models = [
        ("Linear Regression", ManualLinearRegression()),
        ("Polynomial (d=2)", PolynomialRegressor(degree=2)),
        ("Polynomial (d=3)", PolynomialRegressor(degree=3)),
        ("Polynomial (d=4)", PolynomialRegressor(degree=4)),
        ("Quadratic w/ Interactions", QuadraticInteractionRegressor())
    ]
    
    print("\n[Training] Training models...")
    
    for name, model in models:
        print(f"   Training {name}...")
        try:
            model.fit(X_train, y_train)
            mse = model.mse(X_test, y_test)
            r2 = model.score(X_test, y_test)
            
            results.append({
                "Model": name,
                "MSE": mse,
                "R2": r2,
                "Object": model
            })
            print(f"      -> MSE: {mse:.2f}, R2: {r2:.4f}")
        except Exception as e:
            print(f"      -> Failed: {str(e)}")
            
    # 4. Comparison & Reporting
    results_df = pd.DataFrame(results).sort_values("R2", ascending=False)
    print("\n[Results] Model Performance Sorted by R2 (Validation Set):")
    print(results_df[["Model", "MSE", "R2"]].to_string(index=False))
    
    # Save CSV to results directory
    results_df[["Model", "MSE", "R2"]].to_csv(results_dir / "q1_results.csv", index=False)
    
    # 5. Visualizations
    best_model_name = results_df.iloc[0]["Model"]
    best_model = results_df.iloc[0]["Object"]
    
    print(f"\n[Visualization] Generating plots for Best Model: {best_model_name}...")
    
    y_pred = best_model.predict(X_test)
    
    # Plot 1: True vs Predicted Plot
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.5, color='blue', s=10)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.title(f"True vs Predicted Plot: {best_model_name}\nR2: {results_df.iloc[0]['R2']:.4f}")
    plt.xlabel("Actual Demand")
    plt.ylabel("Predicted Demand")
    plt.grid(True, alpha=0.3)
    plt.savefig(figures_dir / "q1_best_model_pred_vs_actual.png")
    plt.close()
    
    # Plot 2: Residuals Plot
    residuals = y_test - y_pred
    plt.figure(figsize=(10, 6))
    plt.scatter(y_pred, residuals, alpha=0.5, color='green', s=10)
    plt.axhline(0, color='red', linestyle='--')
    plt.title(f"Residuals Plot: {best_model_name}")
    plt.xlabel("Predicted Demand")
    plt.ylabel("Residuals")
    plt.grid(True, alpha=0.3)
    plt.savefig(figures_dir / "q1_best_model_residuals.png")
    plt.close()
    
    # Plot 3: Bar Chart of R2 Scores
    plt.figure(figsize=(10, 6))
    plt.barh(results_df["Model"], results_df["R2"], color='purple')
    plt.xlabel("R2 Score")
    plt.title("Model Comparison (R2 Score)")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(figures_dir / "q1_model_comparison_bar.png")
    plt.close()

    print(f"\nPlots saved to {figures_dir}")
    print("Done.")

if __name__ == "__main__":
    main()
