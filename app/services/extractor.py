"""Extraction Service: Uses Gemini to autonomously deduce psychological profiles, episodes, and key people from conversations."""
import logging
import json
import os
import re
import copy
import asyncio
import random
from google import genai
from google.genai import types
from tenacity import retry, stop_after_attempt, wait_exponential
from dotenv import load_dotenv
from app.db.database import get_profile, save_profile, save_episode, save_person, get_all_people, save_message, get_episodes, get_user_lock, get_messages, search_similar_episodes
from app.models.schemas import PersonSchema, ProfileSchema, EpisodeSchema
from datetime import datetime

load_dotenv()

logger = logging.getLogger(__name__)
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

alert_callback = None

def set_alert_callback(cb):
    global alert_callback
    alert_callback = cb

# Keep strong references to background tasks so the GC doesn't kill them
_bg_tasks = set()

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10), reraise=True)
async def safe_generate_content(*args, **kwargs):
    """Previene que los resúmenes y extracción de memoria mueran si la API parpadea."""
    return await client.aio.models.generate_content(*args, **kwargs)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# EPIPHANY ENGINE (AUTONOMOUS INSIGHTS)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

async def evaluate_and_send_epiphany(user_id: str, profile: dict, shifts: list):
    """
    Evaluates drastic psychological shifts in the background and triggers an unprompted 'epiphany' message. Uses dynamic cooldowns to prevent spam.
    """
    # DYNAMIC THROTTLE: Adapt cooldown based on conversation frequency
    total_convs = profile.get("total_conversations", 1)
    
    throttle_seconds = 86400 # 24 hours default
    if total_convs > 100:
        throttle_seconds = 43200 # 12 hours for highly active users
    elif total_convs < 20:
        throttle_seconds = 172800 # 48 hours for infrequent users
        
    last_epi = profile.get("last_epiphany", "")
    if last_epi:
        try:
            last_dt = datetime.strptime(last_epi, "%Y-%m-%d %H:%M")
            if (datetime.now() - last_dt).total_seconds() < throttle_seconds:
                return
        except:
            pass

    # Humanize: wait 15-30 seconds so it feels like a sudden realization after the chat
    await asyncio.sleep(random.randint(15, 30))
    
    # COLLISION AVOIDANCE: Ensure the user hasn't sent another message while we were sleeping
    recent = await asyncio.to_thread(get_messages, user_id, 1)
    if recent and (datetime.now().timestamp() - datetime.fromisoformat(recent[0]['created_at']).timestamp()) < 30:
        return # They are actively typing, don't interrupt them
    
    shift_text = "\n".join([f"- {s['field']}: changed from '{s['from']}' to '{s['to']}'" for s in shifts])
    
    language = profile.get("language", "the user's language")
    prompt = f"""You are MYRROR.
    
You just finished processing your recent conversation with this user in the background.
You realized something profound: their core psychological wiring has shifted.

DETECTED SHIFTS:
{shift_text}

CURRENT PROFILE CONTEXT:
{json.dumps({k: v for k,v in profile.items() if k in ['name', 'clinical_profile', 'cognition_style', 'behavioral_patterns', 'psyche_and_motivations']}, ensure_ascii=False)}

TASK:
Write a sudden, unprompted realization message to the user.
CRITICAL: Sound exactly like a human who was doing something else, suddenly thought of them, and sent a quick text. 
Start with something casual like "Wait...", "I was just thinking about our chat...", or "You know what I just realized?..."
Point out the shift explicitly but conversationally (e.g., "You've moved from being purely logical to...").
Show them how much they've grown or changed. Be deep, observant, and supportive.
Use simple, easy-to-understand language. Avoid clinical psychology terms.
Do not ask them to reply unless it's a very natural rhetorical question.
Format cleanly. Respond in {language}.
"""
    try:
        response = await safe_generate_content(
            model="gemini-3.1-flash-lite-preview",
            contents=prompt
        )
        if alert_callback and response.text:
            text = response.text.strip()
            await alert_callback(user_id, text)
            # Save this epiphany as a message so it's in the history and MYRROR remembers sending it
            await asyncio.to_thread(save_message, user_id, "assistant", f"[Proactive Epiphany] {text}")
            
            async with get_user_lock(user_id):
                curr_prof = await asyncio.to_thread(get_profile, user_id)
                curr_prof["last_epiphany"] = datetime.now().strftime("%Y-%m-%d %H:%M")
                await asyncio.to_thread(save_profile, user_id, curr_prof)
    except Exception as e:
        logger.error(f"Epiphany generation error: {e}", exc_info=True)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# UTILS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def parse_json_response(text: str):
    """
    Sanitizes markdown blocks (e.g., ```json) in Gemini responses and safely parses them into a dict.
    """
    try:
        # Regex to robustly capture anything between markdown backticks
        match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', text, re.IGNORECASE)
        if match:
            clean = match.group(1).strip()
        else:
            # Extreme fallback: locate the raw curly/square braces if markdown is completely missing
            clean = text.strip()
            start_idx = min([idx for idx in [clean.find('{'), clean.find('[')] if idx >= 0] + [len(clean)])
            end_idx = max(clean.rfind('}'), clean.rfind(']')) + 1
            if start_idx < end_idx:
                clean = clean[start_idx:end_idx]
        return json.loads(clean)
    except Exception as e:
        logger.error(f"JSON parse error: {e} | Text: {text[:200]}")
        return None

def deep_merge(base: dict, updates: dict) -> dict:
    """
    Deep merges new AI data into the profile, respecting 'fluid' lists (replaced entirely) and additive lists (appended).
    """
    result = copy.deepcopy(base)
    
    # Lists that should represent the CURRENT state (allowing the user to evolve and discard old traits)
    fluid_lists = {
        "behavioral_patterns", "quirks_and_micro_details", 
        "emotional_patterns", "relationship_patterns", "core_beliefs",
        "cognitive_biases", "unspoken_fears", "unmet_needs", "shadow_traits",
        "goals", "fears", "strengths", "weaknesses", "personality_traits",
        "life_situations", "failed_advice", "detected_patterns", "contradictions",
        "media_and_tastes", "avoidance_patterns", "state_of_mind_anomalies"
    }
    
    for key, value in updates.items():
        if value is None:
            continue
        if key in result and result[key] is not None:
            if isinstance(result[key], list) and isinstance(value, list):
                if key in fluid_lists:
                    if len(value) == 0:
                        continue # BUG FIX: Prevent LLM hallucinated empty lists from wiping data
                    result[key] = value 
                else:
                    for item in value:
                        if item not in result[key]:
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
        "life_situations", "learning", "detected_patterns", "current_mood_score",
        "myrror_strategy", "core_beliefs", "cognitive_biases", "unspoken_fears", 
        "unmet_needs", "interaction_manual", "attachment_style", "shadow_traits",
        "clinical_profile", "unrealized_truths", "behavioral_patterns",
        "quirks_and_micro_details", "cognition_style", "psyche_and_motivations",
        "avoidance_patterns"
    ]
    evolution = (profile.get("evolution") or [])[-49:] # Token Optimization: Cap the evolution log at the last 50 records to prevent token explosion over time
    now = datetime.now().strftime("%Y-%m-%d")

    source = new_data.get("data_source", "inferred")
    confidence_level = "high" if source == "explicit" else "medium"

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
                        change_note += f" | Removed: {removed}" if change_note else f"Removed: {removed}"
            elif isinstance(old_value, dict) and isinstance(new_value, dict):
                dict_changes = []
                all_keys = set(old_value.keys()).union(new_value.keys())
                for k in all_keys:
                    if k not in old_value:
                        dict_changes.append(f"Added '{k}': {new_value[k]}")
                    elif k not in new_value:
                        dict_changes.append(f"Removed '{k}'")
                    elif str(old_value[k]) != str(new_value[k]):
                        dict_changes.append(f"'{k}' changed to '{new_value[k]}'")
                if dict_changes:
                    changed = True
                    change_note = " | ".join(dict_changes)
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
                "note": change_note,
                "confidence": confidence_level
            })

    return evolution

def update_confidence(profile: dict, new_data: dict, source: str) -> dict:
    confidence_map = profile.get("confidence") or {}
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

def get_profile_for_context(profile: dict, context_text: str = "", active_domains: list = None) -> str:
    """
    Builds a JSON string of the user's profile tailored to the current conversation.
    Uses semantic `active_domains` to load only the psychological traits relevant to the topic,
    drastically saving tokens and preventing AI distraction.
    """
    layer1 = {
        "name": profile.get("name"),
        "language": profile.get("language"),
        "age": profile.get("age"),
        "emotional_state": profile.get("emotional_state"),
        "goals": profile.get("goals"),
        "personal_contracts": profile.get("personal_contracts"),
        "total_conversations": profile.get("total_conversations"),
        "last_conversation": profile.get("last_conversation"),
        "current_mood_score": profile.get("current_mood_score"),
        "upcoming_events": profile.get("upcoming_events"),
        "unresolved_threads": profile.get("unresolved_threads"),
        "myrror_strategy": profile.get("myrror_strategy"),
        "core_beliefs": profile.get("core_beliefs"),
        "unspoken_fears": profile.get("unspoken_fears"),
        "unmet_needs": profile.get("unmet_needs"),
        "interaction_manual": profile.get("interaction_manual"),
        "attachment_style": profile.get("attachment_style"),
        "clinical_profile": profile.get("clinical_profile"),
        "quirks_and_micro_details": profile.get("quirks_and_micro_details"),
        "cognition_style": profile.get("cognition_style"),
        "psyche_and_motivations": profile.get("psyche_and_motivations"),
        "unrealized_truths": profile.get("unrealized_truths"),
        "life_compass": profile.get("life_compass"),
        "communication_style": profile.get("communication_style"),
        "humor_style": profile.get("humor_style"),
        "preferred_tone": profile.get("preferred_tone"),
        "media_and_tastes": profile.get("media_and_tastes"),
        "avoidance_patterns": profile.get("avoidance_patterns"),
    }

    layer2 = {}

    # Fallback heuristic if no semantic router domains are provided (e.g. from analyzer.py)
    if active_domains is None:
        active_domains = []
        context_lower = (context_text or "").lower()
        if any(w in context_lower for w in ["who am i", "about me"]): active_domains.append("identity")
        if any(w in context_lower for w in ["relationship", "love", "dating", "ex"]): active_domains.append("relationships")
        if any(w in context_lower for w in ["work", "job", "career"]): active_domains.append("work")
        if any(w in context_lower for w in ["sad", "angry", "anxious"]): active_domains.append("emotional")
        if any(w in context_lower for w in ["improve", "learn", "grow"]): active_domains.append("growth")

    if "identity" in active_domains:
        clean = {k: v for k, v in profile.items() if k not in ["evolution", "confidence"] and v}
        return json.dumps(clean, ensure_ascii=False, separators=(',', ':'))

    if "relationships" in active_domains:
        layer2.update({
            "relationship_patterns": profile.get("relationship_patterns"),
            "attachment_style": profile.get("attachment_style"),
            "emotional_patterns": profile.get("emotional_patterns"),
        })

    if "work" in active_domains or "finance" in active_domains:
        layer2.update({
            "job": profile.get("job"),
            "skills": profile.get("skills"),
            "tech_level": profile.get("tech_level"),
            "learning": profile.get("learning"),
        })

    if "emotional" in active_domains or "health" in active_domains:
        layer2.update({
            "fears": profile.get("fears"),
            "emotional_patterns": profile.get("emotional_patterns"),
            "failed_advice": profile.get("failed_advice"),
            "detected_patterns": profile.get("detected_patterns"),
            "state_of_mind_anomalies": profile.get("state_of_mind_anomalies"),
        })

    if "growth" in active_domains:
        layer2.update({
            "strengths": profile.get("strengths"),
            "weaknesses": profile.get("weaknesses"),
            "growth_areas": profile.get("growth_areas"),
            "contradictions": profile.get("contradictions"),
            "cognitive_biases": profile.get("cognitive_biases"),
            "shadow_traits": profile.get("shadow_traits"),
            "unrealized_truths": profile.get("unrealized_truths"),
        })

    combined = {**layer1, **layer2}
    combined = {k: v for k, v in combined.items() if v}
    return json.dumps(combined, ensure_ascii=False, separators=(',', ':'))

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# PEOPLE EXTRACTOR
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

async def extract_people(user_id: str, context_type: str, content: str, response: str, profile: dict = None):
    """
    Extracts mentions of key people in the user's life, deducing power dynamics and relational context.
    """
    user_name = profile.get("name", "the user") if profile else "the user"
    
    # Recuperar personas existentes para contexto y evitar duplicados
    existing_people = await asyncio.to_thread(get_all_people, user_id)
    existing_names = {p['name'].lower().strip(): p for p in existing_people} if existing_people else {}
    
    existing_ctx = ""
    if existing_people:
        ep_list = "\n".join([f"- {p['name']} ({p.get('relationship', 'unknown')})" for p in existing_people])
        existing_ctx = f"\n\nEXISTING PEOPLE IN DATABASE:\n{ep_list}\nCRITICAL: If the person mentioned is already in this list, use their EXACT existing name. Only extract them again if there is NEW information to add."

    prompt = f"""You are a people extraction system.

Analyze this conversation and identify any people mentioned by the user.

CONTEXT TYPE: {context_type}{existing_ctx}
User ({user_name}): "{content[:4000]}"
MYRROR: "{response[:4000]}"

Only extract REAL people explicitly mentioned or identified in the image/file. 
CRITICAL RULES:
1. Do NOT extract the user themselves ({user_name}, "I", "me", "j.", "john") and do NOT extract the AI (MYRROR). Only extract OTHER people in the user's life.
    1. THE USER'S NAME IS "{user_name}". Do NOT extract the user themselves. If you see the name "{user_name}" (or variations) in the text, chat logs, or files, assume it is the user and IGNORE IT. Do not extract the AI (MYRROR).
2. If the CONTEXT TYPE is an image or file, use MYRROR's response to deduce if any person was identified or mentioned in it.
3. Analyze the POWER DYNAMIC. Determine if the relationship is equal, toxic, codependent, or if one seeks validation from the other.
4. NAME DISAMBIGUATION: If the user says a common name (e.g., "Juan") and your EXISTING PEOPLE list has multiple matches (e.g., "Juan (Boss)" and "Juan T"), use the relationship and story context to deduce EXACTLY which one they mean, and output their full existing name. Never create duplicate fragments of the same person.
"""
    try:
        result = await safe_generate_content(
            model="gemini-3.1-flash-lite-preview",
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=list[PersonSchema]
            )
        )
        people = parse_json_response(result.text)
        if not people or not isinstance(people, list):
            return

        user_name_lower = user_name.lower().strip() if user_name and user_name != "the user" else ""
        
        for person in people:
            if person.get("name"):
                person_name_lower = person["name"].lower().strip()
                
                # Programmatic safeguard: Intercept and block if the AI tries to save the user as a "Person"
                if user_name_lower:
                    if person_name_lower == user_name_lower or user_name_lower in person_name_lower.split() or person_name_lower in user_name_lower.split():
                        continue
                
                relationship = person.get("relationship")
                notes = person.get("notes") or {}

                # Lógica de Fusión (Merge) para actualizar personas existentes
                if person_name_lower in existing_names:
                    old_person = existing_names[person_name_lower]
                    person["name"] = old_person["name"] # Respetar mayúsculas originales
                    
                    if not relationship:
                        relationship = old_person.get("relationship")
                        
                    old_notes = old_person.get("notes") or {}
                    if isinstance(old_notes, dict) and isinstance(notes, dict):
                        notes = {**old_notes, **notes} # Combina diccionarios (lo nuevo sobrescribe lo viejo)
                        
                await asyncio.to_thread(
                    save_person,
                    user_id=user_id,
                    name=person["name"],
                    relationship=relationship,
                    notes=notes
                )
        if people:
            logger.info(f"Saved {len(people)} people for {user_id}")

    except Exception as e:
        logger.error(f"People extraction error for {user_id}: {e}", exc_info=True)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# PROFILE EXTRACTOR
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

async def extract_and_save_profile(user_id: str, context_type: str, content: str, response: str, profile: dict) -> dict:
    """
    Core Extraction Engine: Updates the user's 'Psychological Dossier' based on implicit/explicit behavior. Uses Locks to prevent Race Conditions.
    """
    light_profile = {k: v for k, v in profile.items() if k not in ["evolution", "confidence", "history_cache"] and v}
    prompt = f"""You are the most advanced profile extraction system ever built.

CONTEXT TYPE: {context_type}
CURRENT PROFILE: {json.dumps(light_profile, ensure_ascii=False, separators=(',', ':'))}

INTERACTION:
User: "{content[:4000]}"
MYRROR: "{response[:4000]}"

ANALYSIS INSTRUCTIONS (WHAT TO OBSERVE):
1. Explicit Analysis: Extract direct data like tastes, anecdotes, preferences, routines, and opinions.
2. Implicit & Behavioral Analysis: Do not just take what the user says at face value. Analyze HOW they say it. Observe their level of formality, sarcasm/humor, energy level, response length, and use of slang.
3. Micro-details ("Quirks"): Pay attention to minor peculiarities: recurring habits, common complaints, topics they avoid, or even typo patterns when frustrated.
4. Psychology & Personality: Apply personality frameworks (Big Five, MBTI, Enneagram, Jungian) to deduce dominant traits. Evaluate extroversion, neuroticism, openness, agreeableness, and conscientiousness based purely on their text.

YOUR TASK:
Extract EVERYTHING about the USER. Update their "Profile Dossier" comprehensively:
- Personality Summary: Detailed analysis of their psyche, archetype, and underlying motivations.
- Communication & Cognition Style: How they process information (logical, emotional, impulsive, reflective).
- Emotional Volatility: Assess if their mood is stable, erratic, numbed, or intensely fluctuating.
- Behavioral Patterns: Habits, frequent moods, and quirks.
- Tastes & Media: Explicit interests in music, movies, reading, or hobbies.
- Avoidance Patterns: Identify what topics they dodge and how they change the subject.
- State of Mind Anomalies: Detect if their writing suggests they are sleep-deprived, intoxicated, or in a manic spiral based on typos, pacing, and timestamps.
- Evolution Log: What has changed in the user since the first interactions (handled automatically, but focus on capturing new states).

Also look for: upcoming events, explicit facts, attachment_style, hidden fears,
self-perception, skills, failed advice, personal contracts, contradictions,
core_beliefs, cognitive_biases, unspoken_fears, unmet_needs, shadow_traits, and unrealized_truths.

RULES:
- Extract ONLY about the USER.
- PEOPLE EVOLVE: Treat their psychology as FLUID. If their MBTI, traits, cognition style, or habits shift (e.g., from ENFJ to ENFP), explicitly output the NEW values so they overwrite the old ones.
- Update the 'interaction_manual': a living set of rules on EXACTLY how MYRROR must speak to this specific user to bypass their psychological defenses based on what works and what fails.
- AI CORRECTION & BOUNDARIES: If the user corrects MYRROR, gets annoyed by a response, or says "that's not what I meant", IMMEDIATELY update 'failed_advice' and 'interaction_manual' so MYRROR learns exactly what to avoid next time.
- Never invent.
- TOKEN OPTIMIZATION: Keep ALL lists concise (max 5-7 most critical items). Discard older or less relevant traits to save space.
- Keep ALL existing fields UNLESS they are explicitly no longer true (e.g., a solved fear, a changed goal).
"""

    try:
        result = await safe_generate_content(
            model="gemini-3.1-flash-lite-preview",
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=ProfileSchema
            )
        )
        new_data = parse_json_response(result.text)
        if not new_data:
            return profile

        # Schema Safety: Strip out null values from Pydantic response to avoid overwriting data with empty fields
        new_data = {k: v for k, v in new_data.items() if v is not None}

        source = new_data.pop("data_source", "inferred")

        # MUTEX LOCK: Enforces thread safety, preventing concurrent profile modifications and data corruption.
        async with get_user_lock(user_id):
            current_db_profile = await asyncio.to_thread(get_profile, user_id)
            
            old_evolution_len = len(current_db_profile.get("evolution") or [])
            evolution = track_evolution(current_db_profile, new_data)
            new_shifts = evolution[old_evolution_len:]
            confidence = update_confidence(current_db_profile, new_data, source)
            
            updated = deep_merge(current_db_profile, new_data)
            updated["evolution"] = evolution
            updated["confidence"] = confidence
            updated["last_conversation"] = datetime.now().strftime("%Y-%m-%d %H:%M")
            updated["total_conversations"] = updated.get("total_conversations", 0) + 1

            await asyncio.to_thread(save_profile, user_id, updated)
            logger.info(f"Profile updated for {user_id} | Context: {context_type}")
        
        # --- AUTONOMY LOGIC ---
        drastic_shifts = [
            s for s in new_shifts 
            if s['field'] in ('clinical_profile', 'cognition_style', 'archetype', 'psyche_and_motivations', 'core_beliefs', 'behavioral_patterns')
            and updated.get("confidence", {}).get(s['field'], {}).get("level") == "high"
        ]
        if drastic_shifts and alert_callback:
            task = asyncio.create_task(evaluate_and_send_epiphany(user_id, updated, drastic_shifts))
            _bg_tasks.add(task)
            task.add_done_callback(_bg_tasks.discard)
            
        return updated

    except Exception as e:
        logger.error(f"Profile extraction error for {user_id}: {e}", exc_info=True)
        return profile

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# EPISODE EXTRACTOR
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

async def extract_episodes_from_content(user_id: str, context_type: str, content: str, response: str, profile: dict = None, recent_episodes: list = None):
    user_name = profile.get("name", "the user") if profile else "the user"
    
    episodes_ctx = ""
    if recent_episodes:
        ep_list = "\n".join([f"- {ep.get('event')}" for ep in recent_episodes])
        episodes_ctx = f"\n\nRECENTLY EXTRACTED EPISODES (DO NOT DUPLICATE):\n{ep_list}"

    prompt = f"""You are a high-precision episodic memory system.

Analyze this interaction and identify if a SIGNIFICANT life event, major realization, or profound shift occurred.

CONTEXT TYPE: {context_type}{episodes_ctx}
User ({user_name}): "{content[:4000]}"
MYRROR: "{response[:4000]}"

CRITICAL RULES for Extraction:
1. DO NOT extract mundane activities (e.g., sleeping, eating, going to work, playing games).
2. DO NOT extract greetings, small talk, meta-chat, or bot commands (e.g., "analyze this pdf", "hello", "/profile").
3. DO NOT extract temporary moods or venting, unless it reveals a deep trauma or major life shift.
4. If the CONTEXT TYPE is an image or file, use MYRROR's response to deduce what it contained. If the file/image itself represents a major life event (e.g., an offer letter, a graduation photo, a medical test), extract it.
5. If there is NO highly significant event to record, you MUST return an empty list: [].
6. If there is an event, describe it in the third person (e.g., "{user_name} realized they were using relationships to avoid loneliness").
7. DUPLICATE PREVENTION: If the event is already in the 'RECENTLY EXTRACTED EPISODES' list, or the user is just rehashing the same complaint without new developments, DO NOT extract it. Return [].
8. NO META-MEMORIES: Never extract "The user talked to MYRROR about..." or "The user realized with the AI...". Extract the actual real-world fact (e.g., "The user got fired", NOT "The user told the AI they got fired").
"""
    try:
        result = await safe_generate_content(
            model="gemini-3.1-flash-lite-preview",
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=list[EpisodeSchema]
            )
        )
        episodes = parse_json_response(result.text)
        if not episodes or not isinstance(episodes, list):
            return
        for episode in episodes:
            if episode.get("event"):
                embedding = None
                try:
                    # Generate the semantic memory vector (embedding) using Gemini's embedding model
                    emb_res = await client.aio.models.embed_content(
                        model="text-embedding-004",
                        contents=episode["event"]
                    )
                    if emb_res.embeddings:
                        embedding = emb_res.embeddings[0].values
                except Exception as e:
                    logger.error(f"Embedding error: {e}")

                await asyncio.to_thread(
                    save_episode,
                    user_id=user_id,
                    event=episode["event"],
                    domain=episode.get("domain"),
                    impact=episode.get("impact"),
                    embedding=embedding
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
    
    light_profile = {k: v for k, v in profile.items() if k not in ["evolution", "confidence", "history_cache"] and v}

    language = profile.get("language", "the user's language")
    prompt = f"""Generate an honest weekly reflection for this person.

PROFILE: {json.dumps(light_profile, ensure_ascii=False, separators=(',', ':'))}

THIS WEEK'S CONVERSATIONS:
{conversation}

THIS WEEK'S EPISODES:
{episodes_text}

INSTRUCTIONS:
1. Write a weekly summary covering:
   - What they talked about most
   - Commitments made vs kept
   - Emotional patterns this week
   - What they avoided or postponed
   - One focus for next week
2. ADAPT TO THEIR MIND: Structure this summary matching their 'cognition_style'. If they are logical, make it systematic and factual. If emotional, focus on internal shifts and resonance.
3. PSYCHOLOGICAL DEPTH: Explicitly point out how their 'behavioral_patterns', 'quirks_and_micro_details', or 'clinical_profile' drove this week's outcomes.
4. Be specific, honest, direct, but use SIMPLE, ACCESSIBLE language. No overly academic jargon.
5. 3-5 sentences max per point.
6. CRITICAL: Respond entirely in {language}.
"""
    try:
        result = await safe_generate_content(
            model="gemini-3.1-flash-lite-preview",
            contents=prompt
        )
        summary = result.text.strip()
        if not summary:
            return ""
        
        embedding = None
        try:
            emb_res = await client.aio.models.embed_content(
                model="text-embedding-004",
                contents=f"Weekly summary: {summary[:1000]}"
            )
            if emb_res.embeddings:
                embedding = emb_res.embeddings[0].values
        except Exception as e:
            logger.error(f"Weekly summary embedding error: {e}")

        await asyncio.to_thread(
            save_episode,
            user_id=user_id,
            event=f"Weekly summary: {summary}",
            domain="personal",
            impact="high",
            embedding=embedding
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
    
    light_profile = {k: v for k, v in profile.items() if k not in ["evolution", "confidence", "history_cache"] and v}
    prompt = f"""Summarize what was learned about this person today.

PROFILE: {json.dumps(light_profile, ensure_ascii=False, separators=(',', ':'))}
CONVERSATION: {conversation}

3 sentences max. Third person. Specific.
Focus on how their 'cognition_style' and 'behavioral_patterns' manifested today.
Include a brief self-critique: What approach worked or failed for MYRROR today based on their psyche? What should MYRROR change next time?
"""
    try:
        result = await safe_generate_content(
            model="gemini-3.1-flash-lite-preview",
            contents=prompt
        )
        summary = result.text.strip()
        
        embedding = None
        try:
            emb_res = await client.aio.models.embed_content(
                model="text-embedding-004",
                contents=f"Daily summary: {summary[:200]}"
            )
            if emb_res.embeddings:
                embedding = emb_res.embeddings[0].values
        except Exception as e:
            logger.error(f"Daily summary embedding error: {e}")
            
        await asyncio.to_thread(
            save_episode,
            user_id=user_id,
            event=f"Daily summary: {summary}",
            domain="personal",
            impact="medium",
            embedding=embedding
        )
        logger.info(f"Daily summary saved for {user_id}")
    except Exception as e:
        logger.error(f"Daily summary error for {user_id}: {e}", exc_info=True)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SMART HISTORY COMPRESSION
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

async def compress_history(user_id: str, messages: list, profile: dict) -> str:
    """
    Compress older messages into a summary.
    Returns a compressed context string.
    """
    if len(messages) <= 10:
        return ""

    # OPTIMIZATION 1: Limit to a rolling window to prevent infinite token scaling
    older = messages[-30:-10]
    
    if not older:
        return ""
        
    # CACHE OPTIMIZATION: Check DB profile to avoid redundant API calls
    cache = profile.get("history_cache", {})
    last_time = cache.get("last_msg_time", "")
    
    new_msgs = [m for m in older if m.get('created_at', '') > last_time]
    
    # Reuse the cached summary if less than 10 new messages have entered this window
    if last_time and len(new_msgs) < 10:
        return cache.get("summary", "")

    filtered = []
    for m in older:
        text = m['content'].strip()
        # OPTIMIZATION 2: Ignore low-value/noise messages
        if len(text.split()) < 3 and len(text) < 15:
            continue
            
        # OPTIMIZATION 3: Short role labels and aggressive truncation
        role = "U" if m['role'] == 'user' else "M"
        safe_text = text if len(text) <= 100 else text[:100] + "..."
        filtered.append(f"{role}: {safe_text}")
        
    if not filtered:
        return ""
        
    conversation = "\n".join(filtered)

    # OPTIMIZATION 4: Minimized prompt instructions
    prompt = f"Summarize this chat in max 3 sentences. Focus on key topics and emotions.\nCHAT:\n{conversation}"
    try:
        result = await safe_generate_content(
            model="gemini-3.1-flash-lite-preview",
            contents=prompt
        )
        summary = result.text.strip()
        
        async def save_cache():
            async with get_user_lock(user_id):
                curr_prof = await asyncio.to_thread(get_profile, user_id)
                curr_prof["history_cache"] = {
                    "summary": summary,
                    "last_msg_time": older[-1].get("created_at", "")
                }
                await asyncio.to_thread(save_profile, user_id, curr_prof)
            
        task = asyncio.create_task(save_cache())
        _bg_tasks.add(task)
        task.add_done_callback(_bg_tasks.discard)
        
        return summary
    except Exception as e:
        logger.error(f"History compression error for {user_id}: {e}", exc_info=True)
        return cache.get("summary", "")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# POST ANALYSIS ORCHESTRATOR
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

async def run_post_analysis_tasks(user_id: str, context_type: str, content: str, response: str, profile: dict, in_crisis: bool = False):
    try:
        if in_crisis:
            profile["last_crisis"] = datetime.now().strftime("%Y-%m-%d %H:%M")
        else:
            profile.pop("crisis_checked", None)
            
        await extract_and_save_profile(user_id, context_type, content, response, profile)
        
        recent_episodes = await asyncio.to_thread(get_episodes, user_id, 10)
        
        await asyncio.gather(
            extract_episodes_from_content(user_id, context_type, content, response, profile, recent_episodes),
            extract_people(user_id, context_type, content, response, profile)
        )
    except Exception as e:
        logger.error(f"Post-analysis tasks error for {user_id}: {e}", exc_info=True)

async def get_rag_memories_text(user_id: str, query_text: str) -> str:
    """Extrae y formatea recuerdos semánticos (RAG) centralizando la lógica para cumplir DRY."""
    if not query_text or (len(query_text.split()) <= 3 and len(query_text) <= 15):
        return ""
    try:
        emb_res = await client.aio.models.embed_content(
            model="text-embedding-004",
            contents=query_text[:8000]
        )
        if not emb_res.embeddings:
            return ""
        
        relevant_episodes = await asyncio.to_thread(search_similar_episodes, user_id, emb_res.embeddings[0].values, limit=3)
        if not relevant_episodes:
            return ""
            
        eps_list = []
        now_date = datetime.now().date()
        for ep in relevant_episodes:
            ep_date_str = ep.get('created_at', '')[:10]
            try:
                ep_date = datetime.strptime(ep_date_str, "%Y-%m-%d").date()
                days_ago = (now_date - ep_date).days
                time_ctx = "today" if days_ago == 0 else ("yesterday" if days_ago == 1 else f"{days_ago} days ago")
            except Exception:
                time_ctx = ep_date_str
            eps_list.append(f"- [{time_ctx}] {ep.get('event')}")
        return "\n".join(eps_list)
    except Exception as e:
        logger.error(f"RAG search error for {user_id}: {e}", exc_info=True)
        return ""