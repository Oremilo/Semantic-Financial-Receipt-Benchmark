import sys
import os

# Add parent directory to path so imports work correctly
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataset.build_dataset import build_subset
from ocr.ocr_engine import process_dataset
from benchmark.evaluator import Evaluator
from visualizations.plotting import generate_plots
from reports.report_generator import generate_report

def main():
    print("================================================================")
    print(" LLM-Augmented Semantic Financial Receipt Understanding Pipeline")
    print("================================================================\n")
    
    # Check for Groq API Key
    if not os.environ.get("GROQ_API_KEY"):
        print("WARNING: GROQ_API_KEY is not set. The LLM pipeline will return empty results.")
        print("Please set it: export GROQ_API_KEY='your_key'")
        
    num_samples = 50 # Run on a smaller subset by default to save time/API calls
    
    # 1. Dataset Parsing
    print("\n[PHASE 1] Building Dataset Subset...")
    build_subset(split="train", num_samples=num_samples, output_dir="eval_data")
    
    # 2. OCR Pipeline
    print("\n[PHASE 2] Running OCR Engine...")
    process_dataset(data_dir="eval_data", output_path="eval_data/ocr_results.json")
    
    # 3. Evaluation
    print("\n[PHASE 3, 4, 5, 6] Running Benchmark Evaluation...")
    evaluator = Evaluator(data_dir="eval_data", ocr_file="ocr_results.json")
    evaluator.evaluate(results_dir="results")
    
    # 4. Visualizations & Reports
    print("\n[PHASE 7] Generating Visualizations and Reports...")
    generate_plots(results_dir="results", plots_dir="visualizations")
    generate_report(results_dir="results", report_dir="reports")
    
    print("\n================================================================")
    print(" Pipeline Execution Complete!")
    print(" Check 'results/', 'visualizations/', and 'reports/' directories.")
    print("================================================================")

if __name__ == "__main__":
    main()
