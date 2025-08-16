import os, json, re
from dotenv import load_dotenv
from huggingface_hub import InferenceClient

load_dotenv()
HF_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")

def parse_request_text(user_input: str):
    if not HF_TOKEN:
        # Fallback simple regex
        match = re.match(r".*install ([A-Za-z0-9 ]+)(?: ([0-9]+))?", user_input)
        if match:
            return {"application": match.group(1).strip(), "version": match.group(2), "remarks": user_input}
        return {"application": user_input, "version": None, "remarks": user_input}
    try:
        client = InferenceClient(token=HF_TOKEN)
        prompt = f"Extract software name and version from: {user_input}. Return JSON with keys application, version, remarks."
        response = client.text_generation(model="google/flan-t5-base", prompt=prompt, max_new_tokens=64)
        text = response.strip()
        return json.loads(text)
    except Exception as e:
        return {"application": user_input, "version": None, "remarks": user_input}
