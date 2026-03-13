from google import genai
from dotenv import load_dotenv
from datetime import datetime
import os
import json

load_dotenv()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

def extract_profile(current_profile: dict, user_message: str, myrror_response: str) -> dict:

    prompt = f"""You are a data extraction system. Analyze this exchange and update the user's JSON profile.

CURRENT PROFILE:
{json.dumps(current_profile, ensure_ascii=False, indent=2)}

EXCHANGE:
User: "{user_message}"
MYRROR: "{myrror_response}"

INSTRUCTIONS:
- Extract ONLY what the user explicitly said.
- Never invent data.
- Update or add relevant fields to the profile.
- If the user mentioned name, job, goals, fears, tone preferences — save them.
- If a piece of advice didn't work, add it to failed_advice.
- If you detect a behavioral pattern, add it to detected_patterns.
- Keep ALL existing profile fields. Only add or update, never delete.
- Return ONLY the updated JSON. No extra text. No markdown. Pure JSON only.

USEFUL FIELDS:
- name, age, location, job
- goals, fears, strengths
- preferred_tone, communication_style
- tech_level, skills, learning
- failed_advice, detected_patterns
- personal_contracts
"""

    response = client.models.generate_content(
        model="gemini-3.1-flash-lite-preview",
        contents=prompt
    )

    text = response.text.strip()
    text = text.replace("```json", "").replace("```", "").strip()

    try:
        new_data = json.loads(text)
        current_profile.update(new_data)
        current_profile["last_conversation"] = datetime.now().strftime("%Y-%m-%d %H:%M")
        current_profile["total_conversations"] = current_profile.get("total_conversations", 0) + 1
        return current_profile
    except Exception as e:
        print(f"Error parsing profile: {e}")
        print(f"Received text: {text}")
        return current_profile