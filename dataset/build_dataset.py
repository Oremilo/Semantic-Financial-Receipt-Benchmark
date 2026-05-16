import os
import json
import random
from datasets import load_dataset
from tqdm import tqdm
from PIL import Image, ImageDraw, ImageFont

def generate_synthetic_receipt(category: str, output_path: str) -> dict:
    """Generates a synthetic receipt image and returns its ground truth."""
    # Templates for synthetic receipts
    templates = {
        "Transport": [
            {"merchant": "UBER RIDES", "items": [{"name": "Trip to Airport", "qty": "1", "price": "45.50"}], "total": 45.50},
            {"merchant": "CITY TRANSIT", "items": [{"name": "Monthly Pass", "qty": "1", "price": "120.00"}], "total": 120.00}
        ],
        "Medical": [
            {"merchant": "CITY HOSPITAL", "items": [{"name": "Consultation", "qty": "1", "price": "150.00"}, {"name": "Blood Test", "qty": "1", "price": "75.00"}], "total": 225.00},
            {"merchant": "GREEN PHARMACY", "items": [{"name": "Aspirin", "qty": "2", "price": "12.50"}], "total": 25.00}
        ],
        "Utilities": [
            {"merchant": "POWER CO", "items": [{"name": "Electricity Bill", "qty": "1", "price": "89.45"}], "total": 89.45},
            {"merchant": "WATER WORKS", "items": [{"name": "Water Bill", "qty": "1", "price": "45.20"}], "total": 45.20}
        ],
        "Shopping": [
            {"merchant": "SUPERMART", "items": [{"name": "Milk", "qty": "2", "price": "4.50"}, {"name": "Bread", "qty": "1", "price": "2.50"}], "total": 11.50},
            {"merchant": "TECH STORE", "items": [{"name": "USB Cable", "qty": "1", "price": "15.00"}], "total": 15.00}
        ],
        "Entertainment": [
            {"merchant": "CINEMA MAX", "items": [{"name": "Movie Ticket", "qty": "2", "price": "24.00"}, {"name": "Popcorn", "qty": "1", "price": "8.00"}], "total": 56.00},
            {"merchant": "NETFLIX", "items": [{"name": "Monthly Sub", "qty": "1", "price": "15.99"}], "total": 15.99}
        ]
    }
    
    template = random.choice(templates[category])
    
    # Create image
    img = Image.new('RGB', (400, 600), color=(255, 255, 255))
    d = ImageDraw.Draw(img)
    
    y_text = 20
    d.text((150, y_text), template["merchant"], fill=(0, 0, 0))
    y_text += 40
    
    d.text((20, y_text), "ITEM", fill=(0, 0, 0))
    d.text((250, y_text), "QTY", fill=(0, 0, 0))
    d.text((320, y_text), "PRICE", fill=(0, 0, 0))
    y_text += 30
    
    gt_items = []
    for item in template["items"]:
        d.text((20, y_text), item["name"], fill=(0, 0, 0))
        d.text((250, y_text), item["qty"], fill=(0, 0, 0))
        d.text((320, y_text), item["price"], fill=(0, 0, 0))
        y_text += 30
        gt_items.append({
            "name": item["name"],
            "quantity": item["qty"],
            "price": item["price"]
        })
        
    y_text += 30
    d.text((20, y_text), "TOTAL", fill=(0, 0, 0))
    d.text((320, y_text), f"{template['total']:.2f}", fill=(0, 0, 0))
    
    img.save(output_path)
    
    return {
        "merchant": template["merchant"],
        "items": gt_items,
        "total_amount": float(template["total"]),
        "category": category
    }

def clean_amount(val) -> float:
    try:
        if isinstance(val, str):
            val = val.replace(',', '')
        return float(val)
    except (ValueError, TypeError):
        return 0.0

def build_subset(split="train", num_samples=50, output_dir="eval_data"):
    os.makedirs(output_dir, exist_ok=True)
    images_dir = os.path.join(output_dir, "images")
    os.makedirs(images_dir, exist_ok=True)

    print(f"Loading CORD dataset (split: {split})...")
    dataset = load_dataset("naver-clova-ix/cord-v1", split=split)
    
    # Take a subset for Food category
    num_cord = max(min(num_samples // 2, len(dataset)), 1)
    subset = dataset.select(range(num_cord))
    
    annotations = []
    
    print(f"Processing {len(subset)} CORD samples...")
    idx = 0
    for sample in tqdm(subset):
        image = sample["image"]
        ground_truth_str = sample["ground_truth"]
        
        try:
            gt = json.loads(ground_truth_str)
            gt_parse = gt.get("gt_parse", {})
        except json.JSONDecodeError:
            continue
            
        merchant = "Unknown Food Store"
        if "store" in gt_parse and "name" in gt_parse["store"]:
            merchant = gt_parse["store"]["name"]
        elif "store_name" in gt_parse:
            merchant = gt_parse["store_name"]
            
        items = []
        for menu_item in gt_parse.get("menu", []):
            if isinstance(menu_item, dict):
                items.append({
                    "name": menu_item.get("nm", ""),
                    "quantity": menu_item.get("cnt", ""),
                    "price": menu_item.get("price", "")
                })
            elif isinstance(menu_item, list):
                for sub_item in menu_item:
                    if isinstance(sub_item, dict):
                        items.append({
                            "name": sub_item.get("nm", ""),
                            "quantity": sub_item.get("cnt", ""),
                            "price": sub_item.get("price", "")
                        })
            
        total_amount = 0.0
        if "total" in gt_parse and "total_price" in gt_parse["total"]:
            total_amount = clean_amount(gt_parse["total"]["total_price"])
            
        category = "Food"
        
        image_filename = f"receipt_{idx:04d}.jpg"
        image_path = os.path.join(images_dir, image_filename)
        
        if image.mode != "RGB":
            image = image.convert("RGB")
        image.save(image_path)
        
        annotations.append({
            "id": f"receipt_{idx:04d}",
            "image_path": image_path,
            "ground_truth": {
                "merchant": merchant,
                "items": items,
                "total_amount": total_amount,
                "category": category
            }
        })
        idx += 1
        
    # Generate synthetic diverse samples
    print("Generating synthetic diverse category samples...")
    synthetic_cats = ["Transport", "Medical", "Utilities", "Shopping", "Entertainment"]
    num_synthetic = num_samples - num_cord
    
    for _ in range(num_synthetic):
        cat = random.choice(synthetic_cats)
        image_filename = f"receipt_{idx:04d}.jpg"
        image_path = os.path.join(images_dir, image_filename)
        
        gt = generate_synthetic_receipt(cat, image_path)
        
        annotations.append({
            "id": f"receipt_{idx:04d}",
            "image_path": image_path,
            "ground_truth": gt
        })
        idx += 1
        
    annotations_path = os.path.join(output_dir, "annotations.json")
    with open(annotations_path, "w", encoding="utf-8") as f:
        json.dump(annotations, f, indent=4, ensure_ascii=False)
        
    print(f"Dataset built successfully! Saved {len(annotations)} items to {output_dir}/")
    print(f"Annotations saved to {annotations_path}")

if __name__ == "__main__":
    build_subset(split="train", num_samples=10, output_dir="eval_data")
