import os
import json
import re
import time
from groq import Groq
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class LLMPipeline:
    def __init__(self, model_name="llama-3.3-70b-versatile"):
        self.api_key = os.getenv("GROQ_API_KEY")
        if not self.api_key or self.api_key == "your key" or self.api_key.startswith("gsk_your_api"):
            print("WARNING: GROQ_API_KEY environment variable is not set or invalid. LLM Pipeline will return empty results.")
            self.client = None
        else:
            self.client = Groq(api_key=self.api_key)
        self.model_name = model_name
        
    def _clean_json_string(self, raw_str: str) -> str:
        """Strips markdown json wrappers if present."""
        raw_str = raw_str.strip()
        if raw_str.startswith("```json"):
            raw_str = raw_str[7:]
        elif raw_str.startswith("```"):
            raw_str = raw_str[3:]
            
        if raw_str.endswith("```"):
            raw_str = raw_str[:-3]
            
        return raw_str.strip()

    def parse(self, ocr_text: str, max_retries=3) -> dict:
        """
        Sends the OCR text to the LLM and asks for a structured JSON output.
        Includes retry logic for malformed JSON.
        """
        if not self.client:
            return {"merchant": "", "total_amount": 0.0, "category": "Other", "items": []}
            
        system_prompt = (
            "You are an expert AI extraction system for noisy financial receipts. "
            "Your task is to take raw, potentially noisy OCR text from a receipt and output "
            "a valid JSON object containing exactly the following keys: 'merchant', 'total_amount', "
            "'category', and 'items'.\n"
            "- 'merchant': (string) The name of the store, restaurant, or service provider.\n"
            "- 'total_amount': (float) The final total amount paid. If not found, return 0.0.\n"
            "- 'category': (string) EXACTLY ONE of ['Food', 'Transport', 'Shopping', 'Utilities', 'Medical', 'Entertainment', 'Other']. "
            "Infer this from the items and merchant name.\n"
            "- 'items': (list of dicts) Each dict should have 'name' (string), 'quantity' (string), 'price' (string).\n"
            "Respond ONLY with valid JSON. Do not include markdown formatting, backticks, or any extra text."
        )
        
        user_prompt = f"Raw OCR Text:\n{ocr_text}"
        
        for attempt in range(max_retries):
            try:
                chat_completion = self.client.chat.completions.create(
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    model=self.model_name,
                    temperature=0.0, # Zero temperature for deterministic extraction
                    max_tokens=1024,
                    response_format={"type": "json_object"}
                )
                
                result_str = chat_completion.choices[0].message.content
                cleaned_str = self._clean_json_string(result_str)
                parsed_json = json.loads(cleaned_str)
                
                # Ensure fields exist
                if "merchant" not in parsed_json: parsed_json["merchant"] = ""
                if "items" not in parsed_json: parsed_json["items"] = []
                if "category" not in parsed_json: parsed_json["category"] = "Other"
                
                # Enforce valid category
                valid_cats = ['Food', 'Transport', 'Shopping', 'Utilities', 'Medical', 'Entertainment', 'Other']
                if parsed_json["category"] not in valid_cats:
                    parsed_json["category"] = "Other"
                
                # Ensure total_amount is float
                if "total_amount" in parsed_json:
                    try:
                        if isinstance(parsed_json["total_amount"], str):
                            clean_val = parsed_json["total_amount"].replace(',', '')
                            parsed_json["total_amount"] = float(clean_val)
                        else:
                            parsed_json["total_amount"] = float(parsed_json["total_amount"])
                    except (ValueError, TypeError):
                        parsed_json["total_amount"] = 0.0
                else:
                    parsed_json["total_amount"] = 0.0
                        
                return parsed_json
                
            except json.JSONDecodeError as e:
                print(f"LLM parsing error on attempt {attempt+1}: {e}")
                time.sleep(1) # wait before retry
            except Exception as e:
                print(f"Groq API Error: {e}")
                break
                
        return {"merchant": "", "total_amount": 0.0, "category": "Other", "items": []}

if __name__ == "__main__":
    sample = "UBER RIDES\nTRIP TO AIRPORT 45.50\nTOTAL 45.50"
    pipeline = LLMPipeline()
    print(pipeline.parse(sample))
