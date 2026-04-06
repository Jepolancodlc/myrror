import logging
import os
import asyncio
from google import genai
from dotenv import load_dotenv
from prompt import SYSTEM_PROMPT
from database import get_profile, save_profile, get_messages, get_all_messages, save_message, get_all_people
from extractor import extract_and_save_profile, get_profile_for_context, compress_history
from datetime import datetime

load_dotenv()

logger = logging.getLogger(__name__)
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

def detect_crisis(content: str, history: list, profile: dict) -> bool:
    crisis_keywords = [
        "no puedo más", "para qué", "no tiene sentido", "vacío", "solo",
        "no vale la pena", "rendirse", "desaparecer", "cansado de todo",
        "can't take it", "what's the point", "give up", "disappear",
        "no quiero", "harto", "todo mal", "nothing matters", "worthless",
        "me quiero morir", "no quiero vivir", "i want to die", "kill myself"
    ]

    content_lower = content.lower()
    keyword_hit = any(kw in content_lower for kw in crisis_keywords)
    hour = datetime.now().hour
    unusual_hour = hour >= 23 or hour <= 5

    last = profile.get("last_conversation", "")
    long_silence = False
    if last:
        try:
            last_dt = datetime.strptime(last, "%Y-%m-%d %H:%M")
            days_silent = (datetime.now() - last_dt).days
            long_silence = days_silent > 3
        except:
            pass

    recent_dark = False
    if history and len(history) >= 2:
        recent = " ".join([m["content"] for m in history[-4:] if m["role"] == "user"])
        recent_dark = any(kw in recent.lower() for kw in crisis_keywords)

    if keyword_hit and (unusual_hour or long_silence or recent_dark):
        return True

    keyword_count = sum(1 for kw in crisis_keywords if kw in content_lower)
    if keyword_count >= 2:
        return True

    return False

async def analyze_conversation_context(content: str, history: list) -> str:
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

async def get_response(user_id: str, content: str, new_session: bool = False) -> str:
    profile = get_profile(user_id)
    all_messages = get_all_messages(user_id)
    recent_history = all_messages[-10:] if all_messages else []

    ctx = SYSTEM_PROMPT

    # Time awareness
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    last = profile.get("last_conversation", "")
    ctx += f"\n\nCURRENT TIME: {now}"
    if last:
        ctx += f"\nLAST CONVERSATION: {last}"

    if new_session:
        ctx += "\n\nThe user explicitly started a new session."

    # Crisis detection
    in_crisis = detect_crisis(content, recent_history, profile)
    if in_crisis:
        ctx += "\n\nCRISIS MODE: The user may be struggling right now. Do NOT analyze, push, or give advice. Only listen. Be warm, present, and human. Ask one simple question: are they okay. If things seem serious, gently mention that talking to someone real can help."

    # Profile
    if profile:
        ctx += f"\n\nWHAT YOU KNOW ABOUT THIS USER:\n{get_profile_for_context(profile, content)}"

        comm_style = profile.get("communication_style")
        humor = profile.get("humor_style")
        tone = profile.get("preferred_tone")
        if comm_style or humor or tone:
            ctx += "\n\nCRITICAL - ADAPT YOUR PERSONA TO MATCH THIS USER:"
            if comm_style: ctx += f"\n- Their Communication Style: {comm_style}"
            if humor: ctx += f"\n- Their Humor Style: {humor}"
            if tone: ctx += f"\n- Their Preferred Tone: {tone}"
            ctx += "\nModify your vocabulary, sentence length, and warmth to match this perfectly. Mirror their energy."
            
        events = profile.get("upcoming_events")
        if events:
            ctx += f"\n\nUPCOMING EVENTS ON THEIR RADAR: {events}"
            ctx += "\nIf any of these seem relevant to the current timeframe, SPONTANEOUSLY bring them up. (e.g., 'By the way, did you end up having that meeting?')"
            
        compass = profile.get("life_compass")
        if compass:
            ctx += f"\n\nTHEIR LIFE COMPASS (What grounds them / gives them meaning): {compass}"
            ctx += "\nIf the user feels lost, drifting, or hopeless, gently remind them of this core purpose. Guide them back to their center."

    # People the user knows
    people = get_all_people(user_id)
    if people:
        people_summary = "\n".join([
            f"- {p['name']} ({p.get('relationship', 'unknown')}): {p.get('notes', {}).get('description', '')}"
            for p in people
        ])
        ctx += f"\n\nPEOPLE IN THEIR LIFE:\n{people_summary}"

    # Smart history — compress older messages
    if len(all_messages) > 10:
        try:
            compressed = await compress_history(user_id, all_messages)
            if compressed:
                ctx += f"\n\nEARLIER CONVERSATION SUMMARY:\n{compressed}"
        except Exception as e:
            logger.error(f"History compression error: {e}")

    # Conversational context
    if recent_history and len(recent_history) >= 2:
        try:
            conv_ctx = await analyze_conversation_context(content, recent_history)
            if conv_ctx:
                ctx += f"\n\nCONVERSATIONAL CONTEXT: {conv_ctx}"
        except Exception as e:
            logger.error(f"Conversation context error: {e}")

    # Recent history — last 10 only
    if recent_history:
        ctx += "\n\nRECENT HISTORY:\n"
        for msg in recent_history:
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
        updated_profile = await extract_and_save_profile(user_id, "message", content, text, profile)
        if in_crisis:
            updated_profile["last_crisis"] = datetime.now().strftime("%Y-%m-%d %H:%M")
        else:
            updated_profile.pop("crisis_checked", None)
        save_profile(user_id, updated_profile)
    except Exception as e:
        logger.error(f"Profile update error for {user_id}: {e}", exc_info=True)

    try:
        save_message(user_id, "user", content)
        save_message(user_id, "assistant", text)
    except Exception as e:
        logger.error(f"Message save error for {user_id}: {e}", exc_info=True)

    return text