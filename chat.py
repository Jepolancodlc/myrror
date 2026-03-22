import logging
import os
import asyncio
from google import genai
from dotenv import load_dotenv
from prompt import SYSTEM_PROMPT
from database import get_profile, save_profile, get_messages, save_message
from extractor import extract_profile, get_profile_for_context
from datetime import datetime

load_dotenv()

logger = logging.getLogger(__name__)
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

async def analyze_conversation_context(content: str, history: list) -> str:
    """
    Understand the conversational flow naturally.
    Detects topic changes, avoidance, unresolved threads, tone shifts.
    """
    if not history or len(history) < 2:
        return ""

    last_messages = history[-6:]
    conversation = "\n".join([
        f"{'User' if m['role'] == 'user' else 'MYRROR'}: {m['content'][:200]}"
        for m in last_messages
    ])

    prompt = f"""You are helping MYRROR understand the conversational flow.

RECENT HISTORY:
{conversation}

NEW MESSAGE: "{content}"

In ONE natural sentence, describe what's happening conversationally.
Focus on: topic changes, avoidance patterns, unresolved threads, tone shifts.
Be observational, not mechanical. No labels or categories.

Examples:
- "The user is continuing the same topic naturally."
- "The user just shifted subject — they were talking about work, now asking something lighter."
- "The user seems to be avoiding the previous question by changing topic."
- "Something important was left unresolved — worth noting but don't force it."
- "The user's energy feels lighter than before."

One sentence only.
"""

    try:
        result = client.models.generate_content(
            model="gemini-3.1-flash-lite-preview",
            contents=prompt
        )
        return result.text.strip()
    except Exception as e:
        logger.error(f"Context analysis error: {e}")
        return ""

def get_response(user_id: str, content: str, new_session: bool = False) -> str:
    profile = get_profile(user_id)
    history = get_messages(user_id)

    # Build context
    ctx = SYSTEM_PROMPT

    # Time awareness — let Gemini figure out what to do with it
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    last = profile.get("last_conversation", "")
    ctx += f"\n\nCURRENT TIME: {now}"
    if last:
        ctx += f"\nLAST CONVERSATION: {last}"

    if new_session:
        ctx += "\n\nThe user explicitly started a new session."

    # Profile — layered by context
    if profile:
        ctx += f"\n\nWHAT YOU KNOW ABOUT THIS USER:\n{get_profile_for_context(profile, content)}"

    # Conversational context — natural, not mechanical
    if history and len(history) >= 2:
        try:
            loop = asyncio.get_event_loop()
            if not loop.is_running():
                conv_ctx = loop.run_until_complete(
                    analyze_conversation_context(content, history)
                )
                if conv_ctx:
                    ctx += f"\n\nCONVERSATIONAL CONTEXT: {conv_ctx}"
        except Exception as e:
            logger.error(f"Conversation context error: {e}")

    # Recent history
    if history:
        ctx += "\n\nRECENT HISTORY:\n"
        for msg in history:
            role = "User" if msg["role"] == "user" else "MYRROR"
            ctx += f"{role}: {msg['content']}\n"

    ctx += f"\nUser: {content}\nMYRROR:"

    # Call Gemini
    try:
        response = client.models.generate_content(
            model="gemini-3.1-flash-lite-preview",
            contents=ctx
        )
        text = response.text
    except Exception as e:
        logger.error(f"Gemini error for {user_id}: {e}", exc_info=True)
        raise

    # Update profile
    try:
        updated_profile = extract_profile(profile, content, text)
        save_profile(user_id, updated_profile)
    except Exception as e:
        logger.error(f"Profile update error for {user_id}: {e}", exc_info=True)

    # Save messages
    try:
        save_message(user_id, "user", content)
        save_message(user_id, "assistant", text)
    except Exception as e:
        logger.error(f"Message save error for {user_id}: {e}", exc_info=True)

    return text
