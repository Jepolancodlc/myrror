import logging
import json
import os
from google import genai
from dotenv import load_dotenv
from database import save_profile, save_episode
from datetime import datetime

load_dotenv()

logger = logging.getLogger(__name__)
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# UTILS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def parse_json_response(text: str):
    try:
        clean = text.strip().replace("```json", "").replace("```", "").strip()
        return json.loads(clean)
    except Exception as e:
        logger.error(f"JSON parse error: {e} | Text: {text[:200]}")
        return None

def deep_merge(base: dict, updates: dict) -> dict:
    result = base.copy()
    for key, value in updates.items():
        if key in result:
            if isinstance(result[key], list) and isinstance(value, list):
                existing = [str(x) for x in result[key]]
                for item in value:
                    if str(item) not in existing:
                        result[key].append(item)
            elif isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = deep_merge(result[key], value)
            else:
                result[key] = value
        else:
            result[key] = value
    return result

def track_evolution(profile: dict, new_data: dict) -> list:
    tracked_fields = [
        "emotional_state", "goals", "job", "location",
        "self_perception", "emotional_patterns", "relationship_patterns",
        "life_situations", "learning", "detected_patterns"
    ]
    evolution = profile.get("evolution", [])
    now = datetime.now().strftime("%Y-%m-%d")

    for field in tracked_fields:
        if field not in new_data:
            continue
        old_value = profile.get(field)
        new_value = new_data[field]
        if old_value is None or old_value == new_value:
            continue

        change_note = ""
        changed = False

        if isinstance(old_value, list) and isinstance(new_value, list):
            added = [x for x in new_value if x not in old_value]
            removed = [x for x in old_value if x not in new_value]
            if added or removed:
                changed = True
                change_note = f"Added: {added}" if added else ""
                if removed:
                    change_note += f" | Removed: {removed}"
        else:
            if str(old_value) != str(new_value):
                changed = True
                change_note = f"From '{old_value}' to '{new_value}'"

        if changed:
            evolution.append({
                "date": now,
                "field": field,
                "from": old_value,
                "to": new_value,
                "note": change_note
            })

    return evolution

def update_confidence(profile: dict, new_data: dict, source: str) -> dict:
    confidence_map = profile.get("confidence", {})
    confidence_level = "high" if source == "explicit" else "medium"
    now = datetime.now().strftime("%Y-%m-%d")

    for key in new_data:
        if key in ["last_conversation", "total_conversations", "evolution", "confidence"]:
            continue
        if key not in confidence_map:
            confidence_map[key] = {"level": confidence_level, "source": source, "updated": now}
        else:
            if source == "explicit" and confidence_map[key]["level"] != "high":
                confidence_map[key] = {"level": "high", "source": source, "updated": now}

    return confidence_map

def get_profile_for_context(profile: dict, context: str) -> str:
    """Return only relevant profile fields based on conversation context."""
    layer1 = {
        "name": profile.get("name"),
        "age": profile.get("age"),
        "emotional_state": profile.get("emotional_state"),
        "goals": profile.get("goals"),
        "personal_contracts": profile.get("personal_contracts"),
        "total_conversations": profile.get("total_conversations"),
        "last_conversation": profile.get("last_conversation"),
    }

    context_lower = context.lower()
    layer2 = {}

    relationship_keywords = ["relationship", "girl", "boy", "love", "dating", "miss", "heart", "ex", "crush", "teun"]
    work_keywords = ["work", "job", "code", "tech", "myrror", "python", "programming", "career", "factory", "fábrica"]
    emotional_keywords = ["sad", "depressed", "anxious", "stress", "angry", "lonely", "overwhelmed", "tired"]
    growth_keywords = ["improve", "learn", "better", "change", "grow", "habit", "study", "progress"]
    identity_keywords = ["who am i", "quién soy", "what do you know", "my profile", "about me", "que sabes de mi"]

    if any(w in context_lower for w in identity_keywords):
        clean = {k: v for k, v in profile.items() if k not in ["evolution", "confidence"] and v}
        return json.dumps(clean, ensure_ascii=False, separators=(',', ':'))

    if any(w in context_lower for w in relationship_keywords):
        layer2.update({
            "relationship_patterns": profile.get("relationship_patterns"),
            "emotional_patterns": profile.get("emotional_patterns"),
            "insights_from_files": profile.get("insights_from_files"),
        })

    if any(w in context_lower for w in work_keywords):
        layer2.update({
            "job": profile.get("job"),
            "skills": profile.get("skills"),
            "tech_level": profile.get("tech_level"),
            "learning": profile.get("learning"),
        })

    if any(w in context_lower for w in emotional_keywords):
        layer2.update({
            "fears": profile.get("fears"),
            "emotional_patterns": profile.get("emotional_patterns"),
            "failed_advice": profile.get("failed_advice"),
            "detected_patterns": profile.get("detected_patterns"),
        })

    if any(w in context_lower for w in growth_keywords):
        layer2.update({
            "strengths": profile.get("strengths"),
            "weaknesses": profile.get("weaknesses"),
            "growth_areas": profile.get("growth_areas"),
            "contradictions": profile.get("contradictions"),
        })

    combined = {**layer1, **layer2}
    combined = {k: v for k, v in combined.items() if v}
    return json.dumps(combined, ensure_ascii=False, separators=(',', ':'))

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# PROFILE EXTRACTOR
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

async def extract_and_save_profile(user_id: str, context_type: str, content: str, response: str, profile: dict) -> dict:
    prompt = f"""You are the most advanced profile extraction system ever built.

Your job is to build the most complete, accurate and nuanced profile of a person
based on every signal they give.

CONTEXT TYPE: {context_type}
CURRENT PROFILE: {json.dumps(profile, ensure_ascii=False, separators=(',', ':'))}

INTERACTION:
User: "{content}"
MYRROR: "{response}"

Extract EVERYTHING about the USER. Think like a psychologist, life coach and detective.

Look for: explicit facts, personality traits, emotional state, emotional patterns,
communication style, relationship patterns, core values, hidden fears, dreams,
self-perception, life situations, skills, cultural background, humor style,
decision-making style, failed advice, personal contracts, contradictions, growth areas,
insights from shared files.

Also detect if data is "explicit" (user stated it) or "inferred" (you deduced it).

RULES:
- Extract ONLY about the USER.
- Never invent. Only extract what's actually there.
- Keep ALL existing fields.
- For lists, add without removing.
- Return ONLY pure JSON. No markdown.

{{
  "name": "string", "age": number, "location": "string", "job": "string",
  "goals": [], "fears": [], "strengths": [], "weaknesses": [],
  "personality_traits": [], "emotional_state": "string", "emotional_patterns": [],
  "communication_style": "string", "relationship_patterns": [], "core_values": [],
  "humor_style": "string", "decision_making": "string", "self_perception": "string",
  "life_situations": [], "skills": [], "learning": [], "tech_level": "string",
  "cultural_background": "string", "preferred_tone": "string", "failed_advice": [],
  "detected_patterns": [], "contradictions": [], "personal_contracts": [],
  "insights_from_files": [], "growth_areas": [], "data_source": "explicit|inferred"
}}
"""

    try:
        result = client.models.generate_content(
            model="gemini-3.1-flash-lite-preview",
            contents=prompt
        )
        new_data = parse_json_response(result.text)
        if not new_data:
            return profile

        evolution = track_evolution(profile, new_data)
        source = new_data.pop("data_source", "inferred")
        confidence = update_confidence(profile, new_data, source)

        updated = deep_merge(profile, new_data)
        updated["evolution"] = evolution
        updated["confidence"] = confidence
        updated["last_conversation"] = datetime.now().strftime("%Y-%m-%d %H:%M")
        updated["total_conversations"] = updated.get("total_conversations", 0) + 1

        save_profile(user_id, updated)
        logger.info(f"Profile updated for {user_id} | Context: {context_type}")
        return updated

    except Exception as e:
        logger.error(f"Profile extraction error for {user_id}: {e}", exc_info=True)
        return profile

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# EPISODE EXTRACTOR
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

async def extract_episodes_from_content(user_id: str, content: str, response: str):
    prompt = f"""You are an episode detection system.

Identify significant life moments worth remembering permanently.

User: "{content}"
MYRROR: "{response}"

Only extract REAL, SIGNIFICANT events. Ignore small talk and tests.
Return ONLY a JSON array. Empty [] if nothing significant.

[{{"event": "string", "domain": "tech|work|personal|health|finance|relationships|learning|emotional", "impact": "high|medium|low"}}]
"""
    try:
        result = client.models.generate_content(
            model="gemini-3.1-flash-lite-preview",
            contents=prompt
        )
        episodes = parse_json_response(result.text)
        if not episodes or not isinstance(episodes, list):
            return
        for episode in episodes:
            if episode.get("event"):
                save_episode(user_id=user_id, event=episode["event"], domain=episode.get("domain"), impact=episode.get("impact"))
        logger.info(f"Saved {len(episodes)} episodes for {user_id}")
    except Exception as e:
        logger.error(f"Episode extraction error for {user_id}: {e}", exc_info=True)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# DAILY SUMMARY
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

async def generate_daily_summary(user_id: str, profile: dict, messages: list):
    if not messages:
        return
    conversation = "\n".join([
        f"{'User' if m['role'] == 'user' else 'MYRROR'}: {m['content']}"
        for m in messages[-30:]
    ])
    prompt = f"""Summarize what was learned about this person today.

PROFILE: {json.dumps(profile, ensure_ascii=False, separators=(',', ':'))}
CONVERSATION: {conversation}

Write 3-5 sentences covering:
1. What they talked about
2. New things learned about them
3. Emotional shifts or realizations
4. What to remember next time

Be specific. Third person. Return only the summary text.
"""
    try:
        result = client.models.generate_content(
            model="gemini-3.1-flash-lite-preview",
            contents=prompt
        )
        summary = result.text.strip()
        save_episode(user_id=user_id, event=f"Daily summary: {summary}", domain="personal", impact="medium")
        logger.info(f"Daily summary saved for {user_id}")
    except Exception as e:
        logger.error(f"Daily summary error for {user_id}: {e}", exc_info=True)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# LEGACY
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def extract_profile(current_profile: dict, user_message: str, myrror_response: str) -> dict:
    import asyncio
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            return current_profile
        return loop.run_until_complete(
            extract_and_save_profile("sync", "message", user_message, myrror_response, current_profile)
        )
    except Exception:
        return current_profile