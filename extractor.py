import logging
import json
import os
from google import genai
from google.genai import types
from dotenv import load_dotenv
from database import save_profile, save_episode, save_person, get_all_people
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
        "life_situations", "learning", "detected_patterns", "current_mood_score"
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
    layer1 = {
        "name": profile.get("name"),
        "age": profile.get("age"),
        "emotional_state": profile.get("emotional_state"),
        "goals": profile.get("goals"),
        "personal_contracts": profile.get("personal_contracts"),
        "total_conversations": profile.get("total_conversations"),
        "last_conversation": profile.get("last_conversation"),
        "current_mood_score": profile.get("current_mood_score"),
        "life_compass": profile.get("life_compass"),
        "communication_style": profile.get("communication_style"),
        "humor_style": profile.get("humor_style"),
        "preferred_tone": profile.get("preferred_tone"),
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
            "upcoming_events": profile.get("upcoming_events"),
            "strengths": profile.get("strengths"),
            "weaknesses": profile.get("weaknesses"),
            "growth_areas": profile.get("growth_areas"),
            "contradictions": profile.get("contradictions"),
        })

    combined = {**layer1, **layer2}
    combined = {k: v for k, v in combined.items() if v}
    return json.dumps(combined, ensure_ascii=False, separators=(',', ':'))

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# PEOPLE EXTRACTOR
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

async def extract_people(user_id: str, content: str, response: str):
    """Extract mentions of important people and save them."""
    prompt = f"""You are a people extraction system.

Analyze this conversation and identify any people mentioned by the user.

User: "{content}"
MYRROR: "{response}"

For each person mentioned, extract:
- Their name
- Their relationship to the user (friend, girlfriend, boss, family, etc.)
- Any relevant notes about them

Only extract REAL people explicitly mentioned. Ignore generic references.
Return ONLY pure JSON array. Empty [] if no people mentioned.

[
  {{
    "name": "string",
    "relationship": "string",
    "notes": {{
      "description": "brief description",
      "context": "how they came up in conversation"
    }}
  }}
]
"""
    try:
        result = client.models.generate_content(
            model="gemini-3.1-flash-lite-preview",
            contents=prompt,
            config=types.GenerateContentConfig(response_mime_type="application/json")
        )
        people = parse_json_response(result.text)
        if not people or not isinstance(people, list):
            return

        for person in people:
            if person.get("name"):
                save_person(
                    user_id=user_id,
                    name=person["name"],
                    relationship=person.get("relationship"),
                    notes=person.get("notes", {})
                )
        if people:
            logger.info(f"Saved {len(people)} people for {user_id}")

    except Exception as e:
        logger.error(f"People extraction error for {user_id}: {e}", exc_info=True)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# PROFILE EXTRACTOR
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

async def extract_and_save_profile(user_id: str, context_type: str, content: str, response: str, profile: dict) -> dict:
    prompt = f"""You are the most advanced profile extraction system ever built.

CONTEXT TYPE: {context_type}
CURRENT PROFILE: {json.dumps(profile, ensure_ascii=False, separators=(',', ':'))}

INTERACTION:
User: "{content}"
MYRROR: "{response}"

Extract EVERYTHING about the USER. Think like a psychologist, life coach and detective.
Pay special attention to upcoming events they mention (meetings, exams, trips).

Look for: explicit facts, personality traits, emotional state, emotional patterns,
communication style, relationship patterns, core values, hidden fears, dreams,
self-perception, life situations, skills, cultural background, humor style,
decision-making style, failed advice, personal contracts, contradictions, growth areas,
insights from shared files.

RULES:
- Extract ONLY about the USER.
- Never invent.
- Keep ALL existing fields.
- For lists, add without removing.
- Return ONLY pure JSON. No markdown.

{{
  "name": "string", "age": number, "location": "string", "job": "string",
  "life_compass": "string", "current_mood_score": 5, "upcoming_events": [{{"event": "string", "timeframe": "string"}}],
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
            contents=prompt,
            config=types.GenerateContentConfig(response_mime_type="application/json")
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
            contents=prompt,
            config=types.GenerateContentConfig(response_mime_type="application/json")
        )
        episodes = parse_json_response(result.text)
        if not episodes or not isinstance(episodes, list):
            return
        for episode in episodes:
            if episode.get("event"):
                save_episode(
                    user_id=user_id,
                    event=episode["event"],
                    domain=episode.get("domain"),
                    impact=episode.get("impact")
                )
        logger.info(f"Saved {len(episodes)} episodes for {user_id}")
    except Exception as e:
        logger.error(f"Episode extraction error for {user_id}: {e}", exc_info=True)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# WEEKLY SUMMARY
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

async def generate_weekly_summary(user_id: str, profile: dict, messages: list, episodes: list) -> str:
    if not messages and not episodes:
        return ""

    conversation = "\n".join([
        f"{'User' if m['role'] == 'user' else 'MYRROR'}: {m['content'][:150]}"
        for m in messages[-40:]
    ])

    episodes_text = "\n".join([
        f"- [{ep.get('domain')}] {ep.get('event')}"
        for ep in episodes
        if not ep.get("event", "").startswith("Daily summary")
        and not ep.get("event", "").startswith("Weekly summary")
    ]) or "No significant episodes this week."

    prompt = f"""Generate an honest weekly reflection for this person.

PROFILE: {json.dumps(profile, ensure_ascii=False, separators=(',', ':'))}

THIS WEEK'S CONVERSATIONS:
{conversation}

THIS WEEK'S EPISODES:
{episodes_text}

Write a weekly summary covering:
1. What they talked about most
2. Commitments made vs kept
3. Emotional patterns this week
4. What they avoided or postponed
5. One focus for next week

Be specific, honest, direct. Reference real things.
3-5 sentences max per point.
Respond in the user's language.
"""
    try:
        result = client.models.generate_content(
            model="gemini-3.1-flash-lite-preview",
            contents=prompt
        )
        summary = result.text.strip()
        save_episode(
            user_id=user_id,
            event=f"Weekly summary: {summary[:200]}",
            domain="personal",
            impact="high"
        )
        logger.info(f"Weekly summary saved for {user_id}")
        return summary
    except Exception as e:
        logger.error(f"Weekly summary error for {user_id}: {e}", exc_info=True)
        return ""

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# DAILY SUMMARY
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

async def generate_daily_summary(user_id: str, profile: dict, messages: list):
    if not messages:
        return
    conversation = "\n".join([
        f"{'User' if m['role'] == 'user' else 'MYRROR'}: {m['content'][:150]}"
        for m in messages[-20:]
    ])
    prompt = f"""Summarize what was learned about this person today.

PROFILE: {json.dumps(profile, ensure_ascii=False, separators=(',', ':'))}
CONVERSATION: {conversation}

3 sentences max. Third person. Specific. What to remember next time.
"""
    try:
        result = client.models.generate_content(
            model="gemini-3.1-flash-lite-preview",
            contents=prompt
        )
        summary = result.text.strip()
        save_episode(
            user_id=user_id,
            event=f"Daily summary: {summary}",
            domain="personal",
            impact="medium"
        )
        logger.info(f"Daily summary saved for {user_id}")
    except Exception as e:
        logger.error(f"Daily summary error for {user_id}: {e}", exc_info=True)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SMART HISTORY COMPRESSION
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

async def compress_history(user_id: str, messages: list) -> str:
    """
    Compress older messages into a summary.
    Returns a compressed context string.
    """
    if len(messages) <= 10:
        return ""

    older = messages[:-10]
    conversation = "\n".join([
        f"{'User' if m['role'] == 'user' else 'MYRROR'}: {m['content'][:150]}"
        for m in older
    ])

    prompt = f"""Summarize this conversation history in 3-5 sentences.
Focus on: key topics discussed, emotional state, commitments made, important revelations.
Be specific. Third person for MYRROR, first person context for user.

CONVERSATION:
{conversation}

Return only the summary. No labels.
"""
    try:
        result = client.models.generate_content(
            model="gemini-3.1-flash-lite-preview",
            contents=prompt
        )
        return result.text.strip()
    except Exception as e:
        logger.error(f"History compression error for {user_id}: {e}", exc_info=True)
        return ""

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