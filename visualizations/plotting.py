import os
import json
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

def generate_plots(results_dir="results", plots_dir="visualizations"):
    os.makedirs(plots_dir, exist_ok=True)
    
    summary_path = os.path.join(results_dir, "benchmark_summary.json")
    if not os.path.exists(summary_path):
        print(f"Summary file {summary_path} not found.")
        return
        
    with open(summary_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    systems = list(data.keys())
    
    # 1. Bar Chart: Robustness Drop (Clean vs Noisy FIRI)
    df_data = []
    for sys in systems:
        df_data.append({"System": sys.upper(), "Condition": "Clean", "Score": data[sys]["clean_firi"]})
        df_data.append({"System": sys.upper(), "Condition": "Noisy", "Score": data[sys]["noisy_firi"]})
        
    df_firi = pd.DataFrame(df_data)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(data=df_firi, x="System", y="Score", hue="Condition", palette="coolwarm")
    plt.title("Financial Intelligence Readiness Index (FIRI): Clean vs Noisy OCR")
    plt.ylabel("FIRI Score")
    plt.ylim(0, 1)
    plt.legend(title="OCR Condition")
    plt.savefig(os.path.join(plots_dir, "robustness_comparison.png"), dpi=300, bbox_inches="tight")
    plt.close()
    
    # 2. Performance Drop Chart
    drop_scores = [data[sys]["robustness_drop"] for sys in systems]
    plt.figure(figsize=(8, 5))
    sns.barplot(x=[s.upper() for s in systems], y=drop_scores, palette="Reds_r")
    plt.title("Performance Drop under OCR Noise (Lower is Better)")
    plt.ylabel("FIRI Score Drop")
    plt.savefig(os.path.join(plots_dir, "performance_drop.png"), dpi=300, bbox_inches="tight")
    plt.close()
    
    # 3. Bar Chart: Metric Breakdown for Noisy Data
    metrics = ['merchant_accuracy', 'amount_accuracy', 'category_f1', 'json_completeness']
    df_noisy_data = []
    for sys in systems:
        for m in metrics:
            df_noisy_data.append({
                "System": sys.upper(),
                "Metric": m.replace("_", " ").title(),
                "Score": data[sys]["noisy"][m]
            })
    df_noisy = pd.DataFrame(df_noisy_data)
    
    plt.figure(figsize=(12, 6))
    sns.barplot(data=df_noisy, x="Metric", y="Score", hue="System", palette="mako")
    plt.title("Detailed Metric Breakdown (Under OCR Noise)")
    plt.ylim(0, 1)
    plt.savefig(os.path.join(plots_dir, "noisy_metric_breakdown.png"), dpi=300, bbox_inches="tight")
    plt.close()
    
    print(f"Plots saved to {plots_dir}/")

if __name__ == "__main__":
    generate_plots()
