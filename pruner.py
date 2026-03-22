import os
import requests
from dotenv import load_dotenv

# Load the hidden API key
load_dotenv()

def prune_text_with_scaledown(query, long_context):
    """
    Sends text to ScaleDown for compression using their official XYZ endpoint.
    """
    api_key = os.getenv("SCALEDOWN_API_KEY")
    
    # 1. THE CORRECT URL (.xyz instead of .ai)
    url = "https://api.scaledown.xyz/compress/raw/"
    
    # 2. THE CORRECT HEADERS
    headers = {
        "x-api-key": api_key,  # ScaleDown uses x-api-key, not Bearer
        "Content-Type": "application/json"
    }
    
    # 3. THE CORRECT PAYLOAD SCHEMA
    payload = {
        "context": long_context,
        "prompt": query,
        "scaledown": {
            "rate": "auto"  # Let their AI decide the best compression
        }
    }
    
    try:
        # Give it an 8-second timeout since cloud compression takes a moment
        response = requests.post(url, headers=headers, json=payload, timeout=8)
        response.raise_for_status() 
        data = response.json()
        
        # 4. EXTRACTING THE RIGHT DATA
        compressed_text = data.get("compressed_prompt", "No text returned.")
        tokens_before = data.get("original_prompt_tokens", len(long_context.split()))
        tokens_after = data.get("compressed_prompt_tokens", len(compressed_text.split()))
        
        return compressed_text, tokens_before, tokens_after
        
    except Exception as e:
        # THE SAFETY NET (What saved your app earlier)
        fallback_text = long_context[:1200]
        t_before = len(long_context.split())
        t_after = len(fallback_text.split())
        
        print(f"ScaleDown Error: {e}")
        return fallback_text, t_before, t_after