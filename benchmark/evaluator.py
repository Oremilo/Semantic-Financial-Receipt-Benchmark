import os
import json
import csv
from tqdm import tqdm
from .metrics import BenchmarkMetrics
from baselines.regex_baseline import RegexBaseline
from baselines.ml_baseline import train_ml_baseline
from llm.llm_pipeline import LLMPipeline
from dataset.noise_simulator import simulate_ocr_noise

class Evaluator:
    def __init__(self, data_dir="eval_data", ocr_file="ocr_results.json"):
        self.data_dir = data_dir
        self.ocr_file = os.path.join(data_dir, ocr_file)
        self.annotations_file = os.path.join(data_dir, "annotations.json")
        
        # Load Data
        with open(self.annotations_file, "r", encoding="utf-8") as f:
            self.dataset = json.load(f)
            
        with open(self.ocr_file, "r", encoding="utf-8") as f:
            self.ocr_results = json.load(f)
            
        # Initialize baselines
        self.regex_system = RegexBaseline()
        self.ml_system = train_ml_baseline(data_dir, self.ocr_file)
        self.llm_system = LLMPipeline()
        
    def evaluate(self, results_dir="results"):
        os.makedirs(results_dir, exist_ok=True)
        
        all_results = []
        
        print("Evaluating systems on Clean and Noisy OCR...")
        for item in tqdm(self.dataset):
            doc_id = item["id"]
            gt = item["ground_truth"]
            ocr_text_clean = self.ocr_results.get(doc_id, "")
            ocr_text_noisy = simulate_ocr_noise(ocr_text_clean, noise_probability=0.2)
            
            # Clean Evaluation
            regex_clean = self.regex_system.parse(ocr_text_clean)
            ml_clean = {"merchant": "", "total_amount": 0.0, "category": self.ml_system.predict_category(ocr_text_clean), "items": []}
            llm_clean = self.llm_system.parse(ocr_text_clean)
            
            # Noisy Evaluation
            regex_noisy = self.regex_system.parse(ocr_text_noisy)
            ml_noisy = {"merchant": "", "total_amount": 0.0, "category": self.ml_system.predict_category(ocr_text_noisy), "items": []}
            llm_noisy = self.llm_system.parse(ocr_text_noisy)
            
            all_results.append({
                "id": doc_id,
                "ground_truth": gt,
                "clean": {
                    "regex": regex_clean,
                    "ml": ml_clean,
                    "llm": llm_clean
                },
                "noisy": {
                    "regex": regex_noisy,
                    "ml": ml_noisy,
                    "llm": llm_noisy
                }
            })
            
        systems = ["regex", "ml", "llm"]
        summary = {}
        
        for sys in systems:
            # Clean Metrics
            gt_merchants = [r["ground_truth"].get("merchant", "") for r in all_results]
            clean_merchants = [r["clean"][sys].get("merchant", "") for r in all_results]
            clean_merchant_acc = BenchmarkMetrics.calculate_merchant_accuracy(clean_merchants, gt_merchants)
            
            gt_amounts = [r["ground_truth"].get("total_amount", 0.0) for r in all_results]
            clean_amounts = [r["clean"][sys].get("total_amount", 0.0) for r in all_results]
            clean_amount_acc = BenchmarkMetrics.calculate_amount_accuracy(clean_amounts, gt_amounts)
            
            gt_cats = [r["ground_truth"].get("category", "Other") for r in all_results]
            clean_cats = [r["clean"][sys].get("category", "Other") for r in all_results]
            clean_cat_acc, clean_cat_p, clean_cat_r, clean_cat_f1 = BenchmarkMetrics.calculate_category_metrics(clean_cats, gt_cats)
            
            clean_json_comp = BenchmarkMetrics.calculate_json_completeness([r["clean"][sys] for r in all_results])
            clean_firi = BenchmarkMetrics.calculate_firi(clean_merchant_acc, clean_amount_acc, clean_cat_acc)
            
            # Noisy Metrics
            noisy_merchants = [r["noisy"][sys].get("merchant", "") for r in all_results]
            noisy_merchant_acc = BenchmarkMetrics.calculate_merchant_accuracy(noisy_merchants, gt_merchants)
            
            noisy_amounts = [r["noisy"][sys].get("total_amount", 0.0) for r in all_results]
            noisy_amount_acc = BenchmarkMetrics.calculate_amount_accuracy(noisy_amounts, gt_amounts)
            
            noisy_cats = [r["noisy"][sys].get("category", "Other") for r in all_results]
            noisy_cat_acc, noisy_cat_p, noisy_cat_r, noisy_cat_f1 = BenchmarkMetrics.calculate_category_metrics(noisy_cats, gt_cats)
            
            noisy_json_comp = BenchmarkMetrics.calculate_json_completeness([r["noisy"][sys] for r in all_results])
            noisy_firi = BenchmarkMetrics.calculate_firi(noisy_merchant_acc, noisy_amount_acc, noisy_cat_acc)
            
            # Robustness Drop
            robustness_drop = clean_firi - noisy_firi
            
            summary[sys] = {
                "clean_firi": clean_firi,
                "noisy_firi": noisy_firi,
                "robustness_drop": robustness_drop,
                "clean": {
                    "merchant_accuracy": clean_merchant_acc,
                    "amount_accuracy": clean_amount_acc,
                    "category_accuracy": clean_cat_acc,
                    "category_f1": clean_cat_f1,
                    "json_completeness": clean_json_comp
                },
                "noisy": {
                    "merchant_accuracy": noisy_merchant_acc,
                    "amount_accuracy": noisy_amount_acc,
                    "category_accuracy": noisy_cat_acc,
                    "category_f1": noisy_cat_f1,
                    "json_completeness": noisy_json_comp
                }
            }
            
        # Save results
        summary_path = os.path.join(results_dir, "benchmark_summary.json")
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=4)
            
        csv_path = os.path.join(results_dir, "benchmark_results.csv")
        with open(csv_path, "w", newline='', encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["System", "Condition", "Merchant_Acc", "Amount_Acc", "Category_Acc", "Category_F1", "JSON_Completeness", "FIRI"])
            for sys in systems:
                s_cl = summary[sys]["clean"]
                writer.writerow([sys.upper(), "Clean", f"{s_cl['merchant_accuracy']:.4f}", f"{s_cl['amount_accuracy']:.4f}", 
                                 f"{s_cl['category_accuracy']:.4f}", f"{s_cl['category_f1']:.4f}", 
                                 f"{s_cl['json_completeness']:.4f}", f"{summary[sys]['clean_firi']:.4f}"])
                
                s_no = summary[sys]["noisy"]
                writer.writerow([sys.upper(), "Noisy", f"{s_no['merchant_accuracy']:.4f}", f"{s_no['amount_accuracy']:.4f}", 
                                 f"{s_no['category_accuracy']:.4f}", f"{s_no['category_f1']:.4f}", 
                                 f"{s_no['json_completeness']:.4f}", f"{summary[sys]['noisy_firi']:.4f}"])
                                 
        print(f"Evaluation complete. Summary saved to {summary_path}")
        return summary

if __name__ == "__main__":
    evaluator = Evaluator()
    evaluator.evaluate()
