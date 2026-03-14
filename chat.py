import logging
import os
from google import genai
from dotenv import load_dotenv
from prompt import SYSTEM_PROMPT
from database import get_profile, save_profile, get_messages, save_message
from extractor import extract_profile, get_profile_for_context
from datetime import datetime

load_dotenv()

logger = logging.getLogger(__name__)
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

def get_response(user_id: str, content: str, new_session: bool = False) -> str:
    profile = get_profile(user_id)
    history = get_messages(user_id)

    ctx = SYSTEM_PROMPT
    ctx += f"\n\nCURRENT DATE AND TIME: {datetime.now().strftime('%Y-%m-%d %H:%M')}"

    if new_session:
        ctx += "\n\nNEW SESSION — start fresh. No tension from before."

    if profile:
        ctx += f"\n\nWHAT YOU KNOW ABOUT THIS USER:\n{get_profile_for_context(profile, content)}"

    if history:
        ctx += "\n\nRECENT HISTORY:\n"
        for msg in history:
            role = "User" if msg["role"] == "user" else "MYRROR"
            ctx += f"{role}: {msg['content']}\n"

    ctx += f"\nUser: {content}\nMYRROR:"

    try:
        response = client.models.generate_content(
            model="gemini-3.1-flash-lite-preview",
            contents=ctx
        )
        text = response.text
    except Exception as e:
        logger.error(f"Gemini error for {user_id}: {e}", exc_info=True)
        raise

    try:
        updated_profile = extract_profile(profile, content, text)
        save_profile(user_id, updated_profile)
    except Exception as e:
        logger.error(f"Profile update error for {user_id}: {e}", exc_info=True)

    try:
        save_message(user_id, "user", content)
        save_message(user_id, "assistant", text)
    except Exception as e:
        logger.error(f"Message save error for {user_id}: {e}", exc_info=True)

    return text