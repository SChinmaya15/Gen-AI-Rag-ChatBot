# import os
# import json
# from groq import Groq

# # Initialize Groq client
# client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# def extract_intent(user_message: str) -> dict:
#     """
#     Uses Groq LLM to parse user message into structured intent.
#     Returns a dictionary with keys: {intent: "...", software: "..."}
#     """

#     prompt = f"""
#     You are an intent extraction assistant for a software installation bot.
#     The user message is: "{user_message}".

#     Your task:
#     - If the user wants to install software, return JSON with intent="install" and software name.
#     - Otherwise return JSON with intent="other".

#     Example output:
#     {{"intent": "install", "software": "zoom"}}
#     or
#     {{"intent": "other"}}
#     """

#     response = client.chat.completions.create(
#         model="llama-3.1-8b-instant",  # Groq fast LLaMA model
#         messages=[
#             {"role": "system", "content": "You are a helpful intent parser."},
#             {"role": "user", "content": prompt},
#         ],
#         temperature=0,
#         max_tokens=150,
#     )

#     content = response.choices[0].message.content.strip()

#     try:
#         return json.loads(content)
#     except Exception:
#         # Fallback: heuristic if JSON parsing fails
#         if "install" in content.lower():
#             return {"intent": "install", "software": user_message.split()[-1]}
#         return {"intent": "other"}
