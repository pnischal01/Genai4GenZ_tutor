import os
import requests
from dotenv import load_dotenv

load_dotenv()

def prune_text_with_scaledown(query, long_context):
    api_key = os.getenv("SCALEDOWN_API_KEY")
    
    url = "https://api.scaledown.xyz/compress/raw/"
    
    headers = {
        "x-api-key": api_key,
        "Content-Type": "application/json"
    }
    
    payload = {
        "context": long_context,
        "prompt": query,
        "scaledown": {
            "rate": "auto"
        }
    }
    
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=8)
        response.raise_for_status()
        data = response.json()

        # ✅ FIX: compressed_prompt lives inside data["results"], not at the top level
        results = data.get("results", {})
        compressed_text = results.get("compressed_prompt", "")

        # ✅ FIX: token counts live at the top level
        tokens_before = data.get("total_original_tokens", len(long_context.split()))
        tokens_after = data.get("total_compressed_tokens", len(compressed_text.split()))

        # Safety floor — just in case
        if not compressed_text.strip() or tokens_after < 50:
            print("WARNING: ScaleDown returned empty text. Falling back to raw context.")
            return long_context, tokens_before, tokens_before

        return compressed_text, tokens_before, tokens_after

    except Exception as e:
        fallback_text = long_context[:1200]
        t_before = len(long_context.split())
        t_after = len(fallback_text.split())
        print(f"ScaleDown Error: {e}")
        return fallback_text, t_before, t_after