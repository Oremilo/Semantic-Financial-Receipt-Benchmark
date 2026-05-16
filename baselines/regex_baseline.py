import re

class RegexBaseline:
    def __init__(self):
        # Enhanced keywords for hybrid categorization
        self.category_keywords = {
            "Food": ["restaurant", "cafe", "food", "burger", "pizza", "nasi", "ayam", "kopi", "tea", "coffee", "menu", "kitchen", "bakery", "mcdonalds", "kfc", "subway"],
            "Transport": ["uber", "grab", "taxi", "gojek", "train", "bus", "parking", "toll", "transit", "flight", "lyft", "ticket"],
            "Shopping": ["mart", "supermarket", "store", "mall", "shop", "apparel", "boutique", "market", "grocery", "tech", "retail"],
            "Utilities": ["electric", "water", "internet", "telkomsel", "pln", "bill", "power", "utility", "gas"],
            "Medical": ["hospital", "clinic", "pharmacy", "apotek", "medical", "doctor", "health", "care", "dental"],
            "Entertainment": ["cinema", "movie", "ticket", "netflix", "spotify", "game", "theater", "show", "subscription"]
        }
        
    def extract_amount(self, text: str) -> float:
        """
        Extract the largest numeric value that looks like an amount.
        Handles formats like 1,000.00 or 150000
        """
        # Look for numbers with optional commas and decimals
        matches = re.findall(r'\b\d{1,3}(?:,\d{3})*(?:\.\d{2})?\b', text)
        if not matches:
            # Fallback for plain numbers
            matches = re.findall(r'\b\d+\b', text)
            
        amounts = []
        for m in matches:
            clean_str = m.replace(',', '')
            try:
                amounts.append(float(clean_str))
            except ValueError:
                pass
                
        if not amounts:
            return 0.0
            
        # Often the total is the largest number on the receipt
        return max(amounts)
        
    def extract_merchant(self, text: str) -> str:
        """
        Heuristic: The merchant is usually at the top of the receipt.
        Return the first non-empty line with alpha characters.
        """
        lines = text.split('\n')
        for line in lines:
            line = line.strip()
            if len(line) > 2 and any(c.isalpha() for c in line):
                # Clean up weird OCR chars at the start
                clean_line = re.sub(r'^[^a-zA-Z]+', '', line)
                if len(clean_line) > 2:
                    return clean_line
        return ""
        
    def categorize(self, text: str) -> str:
        """
        Keyword based categorization.
        """
        text_lower = text.lower()
        
        scores = {cat: 0 for cat in self.category_keywords}
        
        for category, keywords in self.category_keywords.items():
            for kw in keywords:
                if kw in text_lower:
                    scores[category] += 1
                    
        best_category = max(scores, key=scores.get)
        if scores[best_category] == 0:
            return "Other"
            
        return best_category
        
    def parse(self, text: str) -> dict:
        """
        Returns structured json baseline.
        """
        return {
            "merchant": self.extract_merchant(text),
            "total_amount": self.extract_amount(text),
            "category": self.categorize(text),
            "items": [] # Regex baseline doesn't extract full structured items well
        }

if __name__ == "__main__":
    sample = "MCDONALDS\nBURGER x 2\nTOTAL 15.00"
    baseline = RegexBaseline()
    print(baseline.parse(sample))
