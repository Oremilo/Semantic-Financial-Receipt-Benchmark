import os
import easyocr
import json
from tqdm import tqdm

class OCREngine:
    def __init__(self, gpu=False):
        print(f"Initializing EasyOCR (GPU={'enabled' if gpu else 'disabled'})...")
        # Initialize the EasyOCR reader. 
        # Warning: downloading models first time might take a bit.
        self.reader = easyocr.Reader(['en'], gpu=gpu)

    def extract_text(self, image_path: str) -> str:
        """
        Extracts raw text from an image using EasyOCR.
        Returns the joined text with newlines.
        """
        try:
            results = self.reader.readtext(image_path)
            # EasyOCR returns list of (bbox, text, prob)
            # We sort by Y coordinate first, then X to roughly get line-by-line reading
            # Simple sorting: y_center then x_center
            
            def get_centers(bbox):
                xs = [p[0] for p in bbox]
                ys = [p[1] for p in bbox]
                return sum(xs)/4, sum(ys)/4

            # Add center coords
            blocks = []
            for bbox, text, prob in results:
                cx, cy = get_centers(bbox)
                blocks.append({"text": text, "cx": cx, "cy": cy, "bbox": bbox})

            # Sort by cy (with a tolerance to group same line)
            # This is a naive line-sorting algorithm
            blocks.sort(key=lambda b: b["cy"])
            
            grouped_lines = []
            current_line = []
            current_y = None
            y_tolerance = 15 # pixels
            
            for b in blocks:
                if current_y is None:
                    current_y = b["cy"]
                    current_line.append(b)
                elif abs(b["cy"] - current_y) <= y_tolerance:
                    current_line.append(b)
                else:
                    # Sort current line by x
                    current_line.sort(key=lambda x: x["cx"])
                    grouped_lines.append(" ".join([x["text"] for x in current_line]))
                    current_line = [b]
                    current_y = b["cy"]
            
            if current_line:
                current_line.sort(key=lambda x: x["cx"])
                grouped_lines.append(" ".join([x["text"] for x in current_line]))
                
            return "\n".join(grouped_lines)
            
        except Exception as e:
            print(f"Error extracting text from {image_path}: {e}")
            return ""

def process_dataset(data_dir="eval_data", output_path="eval_data/ocr_results.json"):
    annotations_path = os.path.join(data_dir, "annotations.json")
    if not os.path.exists(annotations_path):
        print(f"Dataset not found at {annotations_path}")
        return
        
    with open(annotations_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)
        
    engine = OCREngine(gpu=False)
    
    ocr_results = {}
    print("Running OCR on dataset...")
    for item in tqdm(dataset):
        img_path = item["image_path"]
        text = engine.extract_text(img_path)
        ocr_results[item["id"]] = text
        
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(ocr_results, f, indent=4, ensure_ascii=False)
        
    print(f"Saved OCR results to {output_path}")

if __name__ == "__main__":
    process_dataset()
