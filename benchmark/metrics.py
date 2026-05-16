from difflib import SequenceMatcher
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
import numpy as np

def string_similarity(a: str, b: str) -> float:
    a = str(a).strip().lower()
    b = str(b).strip().lower()
    if not a and not b:
        return 1.0 # Both empty
    if not a or not b:
        return 0.0 # One empty, other not
    return SequenceMatcher(None, a, b).ratio()

class BenchmarkMetrics:
    @staticmethod
    def calculate_merchant_accuracy(predictions: list, ground_truths: list) -> float:
        similarities = [string_similarity(p, g) for p, g in zip(predictions, ground_truths)]
        return sum(similarities) / len(similarities) if similarities else 0.0

    @staticmethod
    def calculate_amount_accuracy(predictions: list, ground_truths: list, tolerance=0.01) -> float:
        correct = 0
        for p, g in zip(predictions, ground_truths):
            try:
                p_val = float(str(p).replace(',', ''))
            except:
                p_val = 0.0
            try:
                g_val = float(str(g).replace(',', ''))
            except:
                g_val = 0.0
                
            if abs(p_val - g_val) <= tolerance:
                correct += 1
        return correct / len(predictions) if predictions else 0.0

    @staticmethod
    def calculate_category_metrics(predictions: list, ground_truths: list):
        if not predictions:
            return 0.0, 0.0, 0.0, 0.0
            
        p, r, f1, _ = precision_recall_fscore_support(ground_truths, predictions, average='weighted', zero_division=0)
        acc = accuracy_score(ground_truths, predictions)
        return acc, p, r, f1

    @staticmethod
    def calculate_json_completeness(parsed_jsons: list) -> float:
        expected_keys = ["merchant", "total_amount", "category", "items"]
        total_score = 0
        for p in parsed_jsons:
            if not isinstance(p, dict):
                continue
            # A field is complete if it exists and is truthy, or if it's explicitly 0.0 for amounts
            keys_present = 0
            for k in expected_keys:
                if k in p:
                    if k == "total_amount" and p[k] >= 0:
                        keys_present += 1
                    elif p[k]:
                        keys_present += 1
                        
            total_score += (keys_present / len(expected_keys))
        return total_score / len(parsed_jsons) if parsed_jsons else 0.0

    @staticmethod
    def calculate_firi(merchant_acc: float, amount_acc: float, category_acc: float) -> float:
        """
        Financial Intelligence Readiness Index (FIRI)
        Weights: 0.3 (Merchant), 0.4 (Amount), 0.3 (Category)
        """
        return (0.3 * merchant_acc) + (0.4 * amount_acc) + (0.3 * category_acc)
