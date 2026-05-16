import os
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import numpy as np

class MLBaseline:
    def __init__(self):
        self.model = Pipeline([
            ('tfidf', TfidfVectorizer(max_features=5000, ngram_range=(1, 2))),
            ('clf', LogisticRegression(class_weight='balanced', random_state=42))
        ])
        self.is_trained = False
        self.default_category = "Food" # Fallback
        
    def train(self, texts: list, labels: list):
        """
        Train the model on the given dataset.
        """
        if not texts or not labels:
            print("No data provided for training.")
            return
            
        # Ensure we have more than 1 class
        unique_labels = list(set(labels))
        if len(unique_labels) < 2:
            print("Warning: Only 1 class found in training data. ML model will always predict:", unique_labels[0])
            self.default_category = unique_labels[0]
            self.is_trained = False
            return
            
        self.model.fit(texts, labels)
        self.is_trained = True
        
    def predict_category(self, text: str) -> str:
        if not self.is_trained:
            return self.default_category
            
        try:
            return self.model.predict([text])[0]
        except Exception:
            return self.default_category

def train_ml_baseline(data_dir="eval_data", ocr_results_path="eval_data/ocr_results.json"):
    annotations_path = os.path.join(data_dir, "annotations.json")
    
    if not os.path.exists(annotations_path) or not os.path.exists(ocr_results_path):
        print("Data files not found for training.")
        return MLBaseline()
        
    with open(annotations_path, "r", encoding="utf-8") as f:
        annotations = json.load(f)
        
    with open(ocr_results_path, "r", encoding="utf-8") as f:
        ocr_results = json.load(f)
        
    texts = []
    labels = []
    
    for item in annotations:
        doc_id = item["id"]
        category = item["ground_truth"].get("category", "Food")
        text = ocr_results.get(doc_id, "")
        
        # We need realistic training data, but CORD is mostly Food.
        # So we might not have a balanced dataset in this specific subset.
        texts.append(text)
        labels.append(category)
        
    baseline = MLBaseline()
    baseline.train(texts, labels)
    return baseline

if __name__ == "__main__":
    baseline = MLBaseline()
    print("Untrained prediction:", baseline.predict_category("uber trip"))
