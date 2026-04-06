import logging
import json
import os
import copy
import asyncio
from google import genai
from google.genai import types
from dotenv import load_dotenv
from database import save_profile, save_episode, save_person, get_all_people
from datetime import datetime
from pydantic import BaseModel, Field
from typing import List, Optional, Dict

load_dotenv()

logger = logging.getLogger(__name__)
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# PYDANTIC SCHEMAS (STRUCTURED OUTPUTS)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class PersonNotesSchema(BaseModel):
    description: Optional[str] = Field(default=None, description="brief description")
    context: Optional[str] = Field(default=None, description="how they came up in conversation")

class PersonSchema(BaseModel):
    name: str
    relationship: Optional[str] = Field(default=None)
    notes: Optional[PersonNotesSchema] = Field(default=None)

class ClinicalProfileSchema(BaseModel):
    big_five: Optional[Dict[str, int]] = Field(default=None, description="Dict with O, C, E, A, N integer scores 1-10")
    enneagram: Optional[str] = Field(default=None)
    mbti: Optional[str] = Field(default=None, description="MBTI personality type (e.g., INTJ, ENFP)")
    archetype: Optional[str] = Field(default=None)

class EventItemSchema(BaseModel):
    event: str
    timeframe: str

class ProfileSchema(BaseModel):
    name: Optional[str] = Field(default=None)
    age: Optional[int] = Field(default=None)
    location: Optional[str] = Field(default=None)
    job: Optional[str] = Field(default=None)
    life_compass: Optional[str] = Field(default=None)
    current_mood_score: Optional[int] = Field(default=None)
    myrror_strategy: Optional[str] = Field(default=None, description="your master plan for guiding this user")
    upcoming_events: Optional[List[EventItemSchema]] = Field(default=None)
    unresolved_threads: Optional[List[str]] = Field(default=None, description="pending topics to follow up on")
    goals: Optional[List[str]] = Field(default=None)
    fears: Optional[List[str]] = Field(default=None)
    strengths: Optional[List[str]] = Field(default=None)
    weaknesses: Optional[List[str]] = Field(default=None)
    personality_traits: Optional[List[str]] = Field(default=None)
    emotional_state: Optional[str] = Field(default=None)
    emotional_patterns: Optional[List[str]] = Field(default=None)
    communication_style: Optional[str] = Field(default=None)
    relationship_patterns: Optional[List[str]] = Field(default=None)
    core_values: Optional[List[str]] = Field(default=None)
    humor_style: Optional[str] = Field(default=None)
    decision_making: Optional[str] = Field(default=None)
    self_perception: Optional[str] = Field(default=None)
    life_situations: Optional[List[str]] = Field(default=None)
    skills: Optional[List[str]] = Field(default=None)
    learning: Optional[List[str]] = Field(default=None)
    tech_level: Optional[str] = Field(default=None)
    cultural_background: Optional[str] = Field(default=None)
    preferred_tone: Optional[str] = Field(default=None)
    failed_advice: Optional[List[str]] = Field(default=None)
    detected_patterns: Optional[List[str]] = Field(default=None)
    contradictions: Optional[List[str]] = Field(default=None)
    personal_contracts: Optional[List[str]] = Field(default=None)
    insights_from_files: Optional[List[str]] = Field(default=None)
    growth_areas: Optional[List[str]] = Field(default=None)
    core_beliefs: Optional[List[str]] = Field(default=None)
    cognitive_biases: Optional[List[str]] = Field(default=None)
    data_source: Optional[str] = Field(default="inferred", description="explicit|inferred")
    unspoken_fears: Optional[List[str]] = Field(default=None)
    unmet_needs: Optional[List[str]] = Field(default=None)
    shadow_traits: Optional[List[str]] = Field(default=None)
    interaction_manual: Optional[List[str]] = Field(default=None)
    attachment_style: Optional[str] = Field(default=None)
    clinical_profile: Optional[ClinicalProfileSchema] = Field(default=None)
    behavioral_patterns: Optional[List[str]] = Field(default=None, description="Habits, frequent moods, and behavioral tendencies")
    quirks_and_micro_details: Optional[List[str]] = Field(default=None, description="Typo patterns, ignored topics, recurring complaints, minor quirks")
    cognition_style: Optional[str] = Field(default=None, description="How they process information: logical, emotional, impulsive, reflective, etc.")
    psyche_and_motivations: Optional[str] = Field(default=None, description="Detailed analysis of their psyche and underlying motivations")
    unrealized_truths: Optional[List[str]] = Field(default=None, description="objective facts they haven't realized")

class EpisodeSchema(BaseModel):
    event: str
    domain: Optional[str] = Field(default=None, description="tech|work|personal|health|finance|relationships|learning|emotional")
    impact: Optional[str] = Field(default=None, description="high|medium|low")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# UTILS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def parse_json_response(text: str):
    try:
        clean = text.strip()
        if "```json" in clean:
            clean = clean.split("```json")[1].split("```")
        elif "```" in clean:
            clean = clean.split("```").split("```")[0]
        return json.loads(clean.strip())
    except Exception as e:
        logger.error(f"JSON parse error: {e} | Text: {text[:200]}")
        return None

def deep_merge(base: dict, updates: dict) -> dict:
    result = copy.deepcopy(base)
    for key, value in updates.items():
        if value is None:
            continue
        if key in result and result[key] is not None:
            if isinstance(result[key], list) and isinstance(value, list):
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
        "quirks_and_micro_details", "cognition_style", "psyche_and_motivations"
    ]
    evolution = profile.get("evolution", [])[-49:] # Mantener un límite de los últimos 50 registros para evitar explosión de tokens
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
            "behavioral_patterns": profile.get("behavioral_patterns"),
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
            "cognitive_biases": profile.get("cognitive_biases"),
            "shadow_traits": profile.get("shadow_traits"),
            "clinical_profile": profile.get("clinical_profile"),
            "unrealized_truths": profile.get("unrealized_truths"),
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

Only extract REAL people explicitly mentioned. Ignore generic references.
"""
    try:
        result = await client.aio.models.generate_content(
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

        for person in people:
            if person.get("name"):
                await asyncio.to_thread(
                    save_person,
                    user_id=user_id,
                    name=person["name"],
                    relationship=person.get("relationship") or None,
                    notes=person.get("notes") or {}
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

ANALYSIS INSTRUCTIONS (WHAT TO OBSERVE):
1. Explicit Analysis: Extract direct data like tastes, anecdotes, preferences, routines, and opinions.
2. Implicit & Behavioral Analysis: Do not just take what the user says at face value. Analyze HOW they say it. Observe their level of formality, sarcasm/humor, energy level, response length, and use of slang.
3. Micro-details ("Quirks"): Pay attention to minor peculiarities: recurring habits, common complaints, topics they avoid, or even typo patterns when frustrated.
4. Psychology & Personality: Apply personality frameworks (Big Five, MBTI, Enneagram, Jungian) to deduce dominant traits. Evaluate extroversion, neuroticism, openness, agreeableness, and conscientiousness based purely on their text.

YOUR TASK:
Extract EVERYTHING about the USER. Update their "Profile Dossier" comprehensively:
- Personality Summary: Detailed analysis of their psyche, archetype, and underlying motivations.
- Communication & Cognition Style: How they process information (logical, emotional, impulsive, reflective).
- Behavioral Patterns: Habits, frequent moods, and quirks.
- Evolution Log: What has changed in the user since the first interactions (handled automatically, but focus on capturing new states).

Also look for: upcoming events, explicit facts, attachment_style, hidden fears,
self-perception, skills, failed advice, personal contracts, contradictions,
core_beliefs, cognitive_biases, unspoken_fears, unmet_needs, shadow_traits, and unrealized_truths.

RULES:
- Extract ONLY about the USER.
- Update the 'interaction_manual': a living set of rules on EXACTLY how MYRROR must speak to this specific user to bypass their psychological defenses based on what works and what fails.
- Never invent.
- Keep ALL existing fields.
- For lists, add without removing.
"""

    try:
        result = await client.aio.models.generate_content(
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

        # Limpiar datos nulos de la respuesta del Schema para no borrar info existente
        new_data = {k: v for k, v in new_data.items() if v is not None}

        evolution = track_evolution(profile, new_data)
        source = new_data.pop("data_source", "inferred")
        confidence = update_confidence(profile, new_data, source)

        updated = deep_merge(profile, new_data)
        updated["evolution"] = evolution
        updated["confidence"] = confidence
        updated["last_conversation"] = datetime.now().strftime("%Y-%m-%d %H:%M")
        updated["total_conversations"] = updated.get("total_conversations", 0) + 1

        await asyncio.to_thread(save_profile, user_id, updated)
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
"""
    try:
        result = await client.aio.models.generate_content(
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
                    # Generamos el vector de memoria usando Gemini
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

    prompt = f"""Generate an honest weekly reflection for this person.

PROFILE: {json.dumps(profile, ensure_ascii=False, separators=(',', ':'))}

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
4. Be specific, honest, direct. Reference real things.
5. 3-5 sentences max per point.
6. Respond in the user's language.
"""
    try:
        result = await client.aio.models.generate_content(
            model="gemini-3.1-flash-lite-preview",
            contents=prompt
        )
        summary = result.text.strip()
        if not summary:
            return ""
        if not summary:
            return
        
        embedding = None
        try:
            emb_res = await client.aio.models.embed_content(
                model="text-embedding-004",
                contents=f"Weekly summary: {summary[:200]}"
            )
            if emb_res.embeddings:
                embedding = emb_res.embeddings[0].values
        except Exception as e:
            logger.error(f"Weekly summary embedding error: {e}")

        await asyncio.to_thread(
            save_episode,
            user_id=user_id,
            event=f"Weekly summary: {summary[:200]}",
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
    prompt = f"""Summarize what was learned about this person today.

PROFILE: {json.dumps(profile, ensure_ascii=False, separators=(',', ':'))}
CONVERSATION: {conversation}

3 sentences max. Third person. Specific.
Focus on how their 'cognition_style' and 'behavioral_patterns' manifested today.
Include a brief self-critique: What approach worked or failed for MYRROR today based on their psyche? What should MYRROR change next time?
"""
    try:
        result = await client.aio.models.generate_content(
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
        result = await client.aio.models.generate_content(
            model="gemini-3.1-flash-lite-preview",
            contents=prompt
        )
        return result.text.strip()
    except Exception as e:
        logger.error(f"History compression error for {user_id}: {e}", exc_info=True)
        return ""

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
        
        await asyncio.gather(
            extract_episodes_from_content(user_id, content, response),
            extract_people(user_id, content, response)
        )
    except Exception as e:
        logger.error(f"Post-analysis tasks error for {user_id}: {e}", exc_info=True)