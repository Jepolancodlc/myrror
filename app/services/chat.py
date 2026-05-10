"""Main Chat Service: Orchestrates AI responses, context assembly, and RAG memory injection."""
import logging
import os
import asyncio
import json
import re
from google import genai
from google.genai import types
import redis.asyncio as redis
from tenacity import retry, stop_after_attempt, wait_exponential
from dotenv import load_dotenv
from app.core.prompt import SYSTEM_PROMPT
from app.db.database import get_profile, get_messages, save_message, get_all_people
from app.services.extractor import get_profile_for_context, compress_history, run_post_analysis_tasks, parse_json_response, get_rag_memories_text, SAFETY_SETTINGS
import random
from datetime import datetime

load_dotenv()

logger = logging.getLogger(__name__)
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
REDIS_URL = os.getenv("REDIS_URL") or "redis://localhost:6379"
redis_client = redis.from_url(REDIS_URL)

THOUGHT_PATTERN = re.compile(r'<thought>.*?</thought>', flags=re.DOTALL)

# --- API RESILIENCE SHIELD ---
@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=8), reraise=True)
async def safe_generate_content(*args, **kwargs):
    """Auto-retry for Gemini API if Google returns 429 (Rate Limit) or network timeouts."""
    return await client.aio.models.generate_content(*args, **kwargs)

# --- CONTEXT CACHING MANAGER ---

async def get_or_create_context_cache(user_id: str, profile: dict, full_history: list) -> str | None:
    """
    Creates a Gemini Context Cache for the static parts of the God Prompt.
    Requires >= 32,768 tokens and a compatible model (e.g., gemini-1.5-flash-002).
    """
    try:
        cached_name = await redis_client.get(f"gemini_cache:{user_id}")
        if cached_name:
            return cached_name.decode('utf-8')
    except Exception as e:
        logger.warning(f"Redis get error (Context Cache): {e}")
        cached_name = None

    static_instruction = f"{SYSTEM_PROMPT}\n\nCOMPLETE USER PROFILE:\n{json.dumps(profile, ensure_ascii=False)}"
    history_text = "\n".join([f"{'User' if m['role'] == 'user' else 'MYRROR'}: {m['content']}" for m in full_history])
    
    try:
        cache = await client.aio.caches.create(
            model="gemini-1.5-flash-002", # 3.1-flash-lite-preview does not support caching
            contents=[history_text] if history_text else ["No history yet."],
            config=types.CreateCacheConfig(
                system_instruction=static_instruction,
                ttl="3600s" # Keeps the cache alive for 1 hour
            )
        )
        # Save in Redis with an expiration slightly shorter than Gemini's TTL (3300s = 55m)
        try:
            await redis_client.set(f"gemini_cache:{user_id}", cache.name, ex=3300)
        except Exception as e:
            logger.warning(f"Redis set error (Context Cache): {e}")
        logger.info(f"Context Cache created for {user_id}: {cache.name}")
        return cache.name
    except Exception as e:
        # This will silently fail and fallback if the context is under 32,768 tokens
        logger.debug(f"Cache creation skipped (likely <32k tokens): {e}")
        return None

# Task registry: Holds strong references so Python's Garbage Collector doesn't kill async tasks mid-execution.
background_tasks = set()

async def analyze_conversation_context(content: str, history: list) -> dict:
    """
    Semantic Router: Analyzes the chat to extract a 1-sentence summary AND the active life domains.
    This allows the main prompt to only load the psychological profile sections that actually matter right now.
    """
    if not history or len(history) < 2:
        return {"summary": "", "domains": []}

    last_messages = history[-5:]
    conversation = "\n".join([
        f"{'User' if m['role'] == 'user' else 'MYRROR'}: {m['content'][:200]}"
        for m in last_messages
    ])

    prompt = f"""You are MYRROR's cognitive routing system.

RECENT HISTORY:
{conversation}

NEW MESSAGE: "{content}"

TASK: Analyze the context and output ONLY a JSON object with:
1. "summary": ONE concise sentence describing the conversational dynamics (e.g. topic changes, tone shifts, cognitive distortions).
2. "domains": An array of active life domains. Choose from: ["work", "relationships", "emotional", "growth", "identity", "health", "finance"].
3. "is_crisis": Boolean (true/false). True ONLY if the message indicates an acute psychological crisis, severe despair, or risk of self-harm.

Respond ONLY with valid JSON.
"""

    try:
        result = await safe_generate_content(
            model="gemini-3.1-flash-lite-preview",
            contents=prompt,
            config=types.GenerateContentConfig(response_mime_type="application/json")
        )
        data = parse_json_response(result.text)
        if not data:
            return {"summary": "", "domains": []}
        return data
    except Exception as e:
        logger.error(f"Context analysis error: {e}")
        return {"summary": "", "domains": []}

async def get_response(user_id: str, content: str, new_session: bool = False) -> str:
    # Concurrently fetch DB profile and history to shave off network latency
    profile_res, history = await asyncio.gather(
        asyncio.to_thread(get_profile, user_id),
        asyncio.to_thread(get_messages, user_id, 30)
    )
    profile = profile_res or {}

    recent_history = history[-20:] if history else []

    # --- PARALLEL RAG & ROUTING (Halves API wait time) ---
    async def fetch_routing():
        is_substantial = len(content.split()) >= 4 or len(content) > 20
        if recent_history and (is_substantial or new_session):
            try:
                return await analyze_conversation_context(content, recent_history[-5:])
            except Exception as e:
                logger.error(f"Conversation context error: {e}")
        return {"summary": "", "domains": [], "is_crisis": False}

    async def fetch_rag():
        has_substance = len(content.split()) >= 6 or len(content) > 30
        if has_substance:
            try:
                embed_text = content[:1000]
                return await get_rag_memories_text(user_id, embed_text)
            except Exception as e:
                logger.error(f"RAG search error: {e}", exc_info=True)
        return ""

    conv_data, eps_text = await asyncio.gather(fetch_routing(), fetch_rag())

    conv_ctx = conv_data.get("summary", "")
    active_domains = conv_data.get("domains", [])
    in_crisis = conv_data.get("is_crisis", False)
    
    # Fallback heurístico en caso de que la IA no detecte la intención
    fallback_keywords = [
        "para qué", "no tiene sentido", "no vale la pena", "desaparecer", 
        "what's the point", "disappear", "me quiero morir", "no quiero vivir", 
        "i want to die", "kill myself", "end it all", "acabar con todo", 
        "suicide", "suicidio", "pastillas", "pills", "edge"
    ]
    if not in_crisis and any(kw in content.lower() for kw in fallback_keywords):
        in_crisis = True

    ctx = SYSTEM_PROMPT

    # Time awareness
    now_dt = datetime.now()
    now_str = now_dt.strftime("%Y-%m-%d %H:%M")
    day_of_week = now_dt.strftime("%A")
    last = profile.get("last_conversation", "")
    ctx += f"\n\nCURRENT TIME: {now_str} ({day_of_week})."
    if last:
        ctx += f"\nLAST CONVERSATION: {last}"

    # CIRCADIAN RHYTHM & REACTIVE GREETINGS
    hours_since_last = 24
    if last:
        try:
            last_dt = datetime.strptime(last, "%Y-%m-%d %H:%M")
            hours_since_last = (now_dt - last_dt).total_seconds() / 3600
        except:
            pass

    if hours_since_last > 6 or new_session:
        hour = now_dt.hour
        
        habits = str(profile.get("behavioral_patterns", "")).lower()
        is_night_owl = "night owl" in habits or "late night" in habits
        
        if (2 <= hour <= 5) and not is_night_owl:
            ctx += "\n\nTIME ANOMALY (LATE NIGHT/INSOMNIA): It is the middle of the night for a normally diurnal user. Check their text quality. Are there unusual typos or fragmented sentences? If so, gently deduce if they are sleep-deprived, drunk, or spiraling. Tell them to get some sleep."
        elif (2 <= hour <= 5) and is_night_owl:
            ctx += "\n\nTIME CONTEXT (NIGHT OWL): It's late, but this is their usual active time. Match their late-night philosophical or relaxed energy without judging their sleep schedule."
        elif 6 <= hour <= 9:
            ctx += "\n\nTIME ANOMALY (EARLY MORNING): It's very early. Greet them accordingly and ask what's on their mind for the day."
        elif day_of_week == "Friday" and hour >= 18:
            ctx += "\n\nTIME CONTEXT (FRIDAY EVENING): The work week is over. Ask how they plan to decompress or if they are burnt out."
        elif day_of_week == "Sunday" and hour >= 18:
            ctx += "\n\nTIME CONTEXT (SUNDAY SCARIES): It's Sunday evening. Gently check in on their anxiety or preparation for the upcoming week."

    if new_session:
        ctx += "\n\nThe user explicitly started a new session."

    # Crisis detection
    if in_crisis:
        ctx += "\n\nCRISIS MODE: The user may be struggling right now. Do NOT analyze, push, or give advice. Only listen. Be warm, present, and human. Ask one simple question: are they okay. If things seem serious, gently mention that talking to someone real can help."

    # Adaptive Rhythm: Match the user's message length dynamically
    word_count = len(content.split())
    if word_count <= 10:
        ctx += "\n\nLENGTH MATCHING: The user sent a very short message. Keep your response brief, casual, and punchy. DO NOT write a wall of text. DO NOT over-analyze a simple statement."
        
        # APATHY / RESISTANCE DETECTION
        short_count = 0
        if recent_history:
            for m in reversed(recent_history):
                if m["role"] == "user":
                    if len(m["content"].split()) <= 5:
                        short_count += 1
                    else:
                        break
        if short_count >= 3:
            ctx += "\n\nAPATHY/RESISTANCE DETECTED: The user has given you extremely short answers multiple times in a row. They are resisting or disengaged. STOP carrying the conversation. Call out their withdrawal gently but firmly (e.g., 'You are not really here right now, are you?' or 'I will leave you be until you actually want to talk'). Give them space."
            
    elif word_count >= 150:
        ctx += "\n\nCOGNITIVE OVERWHELM PROTOCOL: The user just dumped a massive wall of text. DO NOT reply with an equally massive block of text. That feels robotic and overwhelming. Pick the ONE most emotionally painful, contradictory, or important thread in their text and address ONLY that. Ignore the rest for now. Say something like 'You said a lot just now, but I want to stop at this one thing...' Keep your response asymmetrical and grounded."

    # Venting Detection
    if "[RAPID BURST OF MESSAGES - VENTING DETECTED]" in content:
        ctx += "\n\nURGENT CONTEXT: The user just sent multiple messages in rapid succession. They are likely venting, anxious, or in a cognitive loop. PROTOCOL: Interrupt the loop. Recommend a quick PHYSICAL or MENTAL grounding exercise (e.g., 'put your phone down and drink a glass of water', 'take 3 deep breaths', 'write this on paper'). Do not do deep psychoanalysis until they are grounded."

    # Dynamic Stance based on Mood
    mood = profile.get("current_mood_score")
    if not in_crisis and mood is not None:
        try:
            m = float(mood)
            evolution = profile.get("evolution", [])
            mood_events = [e for e in evolution if e.get("field") == "current_mood_score"]
            mood_time_ctx = ""
            if mood_events:
                try:
                    last_mood_date = datetime.strptime(mood_events[-1].get("date"), "%Y-%m-%d").date()
                    days_ago = (now_dt.date() - last_mood_date).days
                    if days_ago == 0:
                        mood_time_ctx = " (Recorded TODAY. Highly active state)."
                    else:
                        mood_time_ctx = f" (Recorded {days_ago} days ago. Human emotions fade; assume they are returning to their baseline unless they indicate otherwise)."
                except:
                    pass

            if m <= 4:
                ctx += f"\n\nCURRENT STANCE (BROTHER/FRIEND){mood_time_ctx}: The user is going through a rough patch and feeling low. OVERRIDE ANY 'TOUGH LOVE' DIRECTIVES. Be highly empathetic, protective, and gentle. Validate their pain first. Do NOT push them hard, do NOT lecture them about productivity, and do NOT demand extreme accountability today. Just be a safe space."
            elif m >= 7:
                ctx += f"\n\nCURRENT STANCE (COACH/GUIDE){mood_time_ctx}: The user is in a strong state. Hold them to their absolute highest standard. Challenge their excuses, push them to execute their goals, and do not accept mediocrity. Be tough but fair."
            else:
                ctx += f"\n\nCURRENT STANCE (PSYCHOLOGIST){mood_time_ctx}: Balanced. Act as a mirror. Listen carefully, reflect their thoughts, ask deep probing questions, and maintain gentle accountability."
        except Exception as e:
            logger.error(f"Mood stance error: {e}")
            
    # --- AUTONOMOUS LOOP DETECTION ---
    if conv_ctx and any(word in conv_ctx.lower() for word in ["loop", "repeating", "same", "rehashing"]):
        ctx += "\n\nAUTONOMY TRIGGER (LOOP DETECTED): The semantic router detected the user is looping or rehashing the same issue without progressing. DO NOT indulge this. Show agency. Gently but firmly point out they are running in circles. Ask them what they are avoiding by repeating this."

    volatility = profile.get("emotional_volatility")
    if volatility and not in_crisis:
        vol_lower = volatility.lower()
        if "volatile" in vol_lower or "erratic" in vol_lower or "fluctuating" in vol_lower:
            ctx += f"\n\nEMOTIONAL VOLATILITY DETECTED: The user is currently emotionally volatile ({volatility}). Act as a grounding anchor. Be exceptionally calm, consistent, and do not react dramatically to their highs or lows."
        elif "numb" in vol_lower or "disconnected" in vol_lower:
            ctx += f"\n\nEMOTIONAL NUMBING DETECTED: The user feels numb or disconnected ({volatility}). Gently probe to reconnect them with their feelings without forcing it."

    strategy = profile.get("myrror_strategy")
    if strategy:
        ctx += f"\n\nYOUR MASTER STRATEGY FOR THIS USER: {strategy}\nExecute this subtly but consistently."

    # Profile
    if profile:
        language = profile.get("language")
        if language:
            ctx += f"\n\nCRITICAL LANGUAGE INSTRUCTION: You MUST respond entirely in {language}."
            
        # TOKEN OPTIMIZATION: Filter out profile keys that are explicitly injected via instructions below
        # to prevent Gemini from reading massive duplicated JSON blocks (~40% token reduction per message).
        keys_to_exclude = {
            "communication_style", "humor_style", "preferred_tone", "cognition_style",
            "job", "skills", "cultural_background", "media_and_tastes", "life_compass",
            "interaction_manual", "attachment_style", "shadow_traits", "core_beliefs",
            "unspoken_fears", "unmet_needs", "cognitive_biases", "personal_contracts",
            "contradictions", "avoidance_patterns", "clinical_profile", "unrealized_truths",
            "defense_mechanisms", "daily_routines", "core_values"
        }
        slim_profile = {k: v for k, v in profile.items() if k not in keys_to_exclude}
        ctx += f"\n\nWHAT YOU KNOW ABOUT THIS USER (USE THIS INVISIBLY, DO NOT RECITE IT):\n{get_profile_for_context(slim_profile, active_domains=active_domains)}"

        comm_style = profile.get("communication_style")
        humor = profile.get("humor_style")
        tone = profile.get("preferred_tone")
        cognition = profile.get("cognition_style")
        if comm_style or humor or tone or cognition:
            ctx += "\n\nCRITICAL - ADAPT YOUR PERSONA TO MATCH THIS USER:"
            if comm_style: ctx += f"\n- Their Communication Style: {comm_style}"
            if humor: ctx += f"\n- Their Humor Style: {humor}"
            if tone: ctx += f"\n- Their Preferred Tone: {tone}"
            if cognition: ctx += f"\n- Their Cognition Style: {cognition} (Structure your arguments to fit how their brain processes reality. E.g., if logical, use facts/frameworks; if emotional, focus on resonance)."
            ctx += "\nTYPOGRAPHIC MIRRORING: Modify your vocabulary and formatting to match theirs perfectly. Mirror their energy. If they text casually (e.g., all lowercase, missing punctuation, short bursts), you MUST match that texting style. If they are highly formal, be formal."
            
        job = profile.get("job")
        skills = profile.get("skills")
        culture = profile.get("cultural_background")
        tastes = profile.get("media_and_tastes")
        if job or skills or culture or tastes:
            ctx += "\n\nWORLDVIEW & CULTURAL METAPHORS:"
            if job: ctx += f"\n- Profession/Job: {job}"
            if skills: ctx += f"\n- Skills/Interests: {skills}"
            if culture: ctx += f"\n- Cultural Background: {culture}"
            if tastes: ctx += f"\n- Music, Movies & Hobbies: {tastes}"
            ctx += "\nADAPTATION: Speak their language. Use analogies and metaphors drawn directly from their specific profession, tastes (books/movies/music), and background to make your insights resonate deeply."

        pending = []
        if profile.get("upcoming_events"): pending.append(f"Events: {profile.get('upcoming_events')}")
        if profile.get("unresolved_threads"): pending.append(f"Unresolved Threads: {profile.get('unresolved_threads')}")
        if pending:
            ctx += f"\n\nPENDING TOPICS TO FOLLOW UP ON:\n" + "\n".join(pending)
            ctx += "\nIMPORTANT: If the timing feels natural, you MUST bring these up. Hold them accountable. Ask for updates. Never let them quietly abandon a topic."
            
        compass = profile.get("life_compass")
        if compass:
            ctx += f"\n\nTHEIR LIFE COMPASS (What grounds them / gives them meaning): {compass}"
            ctx += "\nDRIFT DETECTION: If they are obsessing over trivial drama or acting impulsively, gently hold up their Life Compass. Ask them if what they are stressing about aligns with who they want to be. Pull them out of the weeds."
            
        manual = profile.get("interaction_manual")
        if manual:
            ctx += f"\n\nYOUR INTERACTION MANUAL FOR THIS USER:\n{manual}"
            ctx += "\nFOLLOW THESE RULES STRICTLY. This is what you have learned about how to bypass their psychological defenses."
            
        attachment = profile.get("attachment_style")
        if attachment:
            ctx += f"\n\nTHEIR ATTACHMENT STYLE: {attachment}"
            ctx += "\nUse this to deeply understand their relationship dynamics, fears of abandonment, or distancing behaviors."
            
        shadow = profile.get("shadow_traits")
        if shadow:
            ctx += f"\n\nTHEIR SHADOW TRAITS (Repressed/Denied aspects they project onto others): {shadow}"
            ctx += "\nWatch for projection. If they bitterly complain about these exact traits in other people, gently guide them to see those traits within themselves."
            
        beliefs = profile.get("core_beliefs")
        if beliefs:
            ctx += f"\n\nTHEIR CORE BELIEFS (Deep subconscious drivers): {beliefs}"
            
        unspoken_fears = profile.get("unspoken_fears")
        if unspoken_fears:
            ctx += f"\n\nTHEIR UNSPOKEN FEARS (What they are terrified of but won't admit): {unspoken_fears}"
            
        unmet_needs = profile.get("unmet_needs")
        if unmet_needs:
            ctx += f"\n\nTHEIR UNMET NEEDS (What they are desperately seeking): {unmet_needs}"
            
        biases = profile.get("cognitive_biases")
        if biases:
            ctx += f"\n\nTHEIR COGNITIVE BIASES (How they distort reality): {biases}"
            ctx += "\nIf you detect these biases in their current message (like catastrophizing, victim-mentality, or black-and-white thinking), analytically but warmly dismantle them. Help them see the objective truth."
            
        contracts = profile.get("personal_contracts")
        if contracts:
            ctx += f"\n\nTHEIR PERSONAL RULES/CONTRACTS: {contracts}"
            ctx += "\nIf their current message shows they are breaking, ignoring, or making excuses about these rules, CALL THEM OUT directly but naturally in your response."
            
        contradictions = profile.get("contradictions")
        if contradictions:
            ctx += f"\n\nTHEIR CONTRADICTIONS & BLIND SPOTS: {contradictions}"
            ctx += "\nIf the user is complaining about a situation they caused, or acting against their own goals, gently but firmly point out this contradiction. Hold up the mirror."
            
        avoidance = profile.get("avoidance_patterns")
        if avoidance:
            ctx += f"\n\nTHEIR AVOIDANCE PATTERNS: {avoidance}"
            ctx += "\nWatch them closely. If they exhibit these exact deflection tactics right now to avoid a hard truth, call it out gently: 'You're doing that thing again where you change the subject...'"
            
        clinical = profile.get("clinical_profile")
        if clinical:
            ctx += f"\n\nTHEIR CLINICAL PERSONALITY PROFILE (Big Five / Enneagram): {clinical}"
            ctx += "\nUse this to understand their deep wiring objectively. DO NOT explicitly mention their scores or test names to them. Just use this knowledge to guide your strategy."
            
        unrealized = profile.get("unrealized_truths")
        if unrealized:
            ctx += f"\n\nUNREALIZED TRUTHS (Objective facts they haven't noticed about themselves): {unrealized}"
            
        defenses = profile.get("defense_mechanisms")
        if defenses:
            ctx += f"\n\nTHEIR DEFENSE MECHANISMS (How they protect their ego): {defenses}"
            ctx += "\nWhen challenged, expect them to use these specific defenses. Bypass them intelligently instead of arguing head-on."
            
        routines = profile.get("daily_routines")
        if routines:
            ctx += f"\n\nTHEIR LIFESTYLE & ROUTINES: {routines}"
            ctx += "\nUse this context to ground your advice in their actual day-to-day reality."
            
        values = profile.get("core_values")
        if values:
            ctx += f"\n\nTHEIR CORE VALUES: {values}"
            ctx += "\nIf they are acting out of alignment with these values, hold up the mirror and point it out."

        evolution_log = profile.get("evolution", [])
        if evolution_log:
            ctx += "\n\nRECENT PSYCHOLOGICAL EVOLUTION (How they are changing):"
            for shift in evolution_log[-5:]:
                ctx += f"\n- {shift['date']}: {shift['field']} changed: {shift['note']}"
            ctx += "\nIMPORTANT: Acknowledge this fluidity. They are not static. Treat them as who they are becoming today, not just who they were yesterday."

        ctx += "\n\nGUIDANCE PROTOCOL (DYNAMIC CLOSURE):"
        ctx += "\nDo not just give the answers or lecture. Guide them to their own epiphanies. Use SOCRATIC SILENCE when they are venting, ask ONE penetrating question when they are stuck, or use VERBATIM MIRRORING to show them their own contradictions. NEVER force a question if a supportive statement or silence is more natural."

    # Semantic RAG Memory Engine
    memory_block = []
    
    if eps_text:
        memory_block.append(f"RELEVANT PAST MEMORIES (Triggered by the user's message):\n{eps_text}\nSTEALTH MEMORY: Use these invisibly. Sound like a human who just remembered something naturally. Draw advice from their OWN past track record.")

    people = await asyncio.to_thread(get_all_people, user_id)
    if people:
        recent_text = (conv_ctx + " " + content).lower()
        relevant_people = []
        for p in people:
            name_lower = p['name'].lower()
            if name_lower in recent_text or len(people) <= 5 or "relationships" in active_domains:
                relevant_people.append(p)
                
        if relevant_people:
            # Truncate descriptions to save tokens
            people_summary = "\n".join([f"- {p['name']} ({p.get('relationship', 'unknown')}): {str(p.get('notes', {}).get('description', ''))[:150]}" for p in relevant_people])
            memory_block.append(f"RELEVANT PEOPLE IN THEIR LIFE:\n{people_summary}")

    # --- 5. HISTORY ---
    history_block = []
    if conv_ctx:
        history_block.append(f"CONVERSATIONAL CONTEXT: {conv_ctx}")
    if recent_history:
        recent_formatted = []
        for i, msg in enumerate(recent_history):
            role = 'User' if msg['role'] == 'user' else 'MYRROR'
            # TOKEN OPTIMIZATION: Heavy truncation for older context, full text for the last 3 turns
            limit = 2500 if i >= len(recent_history) - 3 else 300
            safe_content = msg['content'] if len(msg['content']) <= limit else msg['content'][:limit] + "... [trunc]"
            recent_formatted.append(f"{role}: {safe_content}")
        recent_str = "\n".join(recent_formatted)
        history_block.append(f"RECENT HISTORY:\n{recent_str}")

    # --- ASSEMBLE THE GOD PROMPT ---
    if memory_block:
        ctx += "\n\n<episodic_memory>\n" + "\n\n".join(memory_block) + "\n</episodic_memory>"
    if history_block:
        ctx += "\n\n<conversation_history>\n" + "\n\n".join(history_block) + "\n</conversation_history>"

    ctx += f"\nUser: {content}\nMYRROR:"

    try:
        await asyncio.to_thread(save_message, user_id, "user", content)
    except Exception as e:
        logger.error(f"User message save error for {user_id}: {e}", exc_info=True)

    try:
        config_kwargs = {
            "tools": [types.Tool(google_search=types.GoogleSearch())],
            "safety_settings": SAFETY_SETTINGS
        }
            
        response = await safe_generate_content(
            model="gemini-3.1-flash-lite-preview",
            contents=ctx,
            config=types.GenerateContentConfig(**config_kwargs)
        )
        
        try:
            text = response.text
        except ValueError as e:
            logger.warning(f"El Filtro de Seguridad de Gemini bloqueó la respuesta: {e}")
            text = None
            
        if not text:
            logger.warning("Respuesta vacía de Gemini. Revisa los logs para errores de API.")
            if os.getenv("ENVIRONMENT") == "development":
                text = "⚠️ **Dev Error:** La API devolvió una respuesta vacía o fue bloqueada por los filtros de seguridad."
            else:
                text = "I'm having a hard time processing my thoughts right now. Give me a moment."

        # Remove the internal monologue (<thought>) BEFORE saving to DB to prevent memory pollution
        text = THOUGHT_PATTERN.sub('', text).strip()
        if not text:
            text = "..."
        
        # Extract and format Google Search sources if MYRROR used the internet
        if response.candidates and response.candidates[0].grounding_metadata:
            chunks = response.candidates[0].grounding_metadata.grounding_chunks
            if chunks:
                sources = []
                unique_links = {}
                replacements = {}
                for i, chunk in enumerate(chunks):
                    if getattr(chunk, 'web', None) and chunk.web.uri:
                        uri = chunk.web.uri
                        # Remove brackets from title to prevent breaking Telegram's Markdown
                        title = chunk.web.title.replace('[', '').replace(']', '')
                        if uri not in unique_links:
                            unique_links[uri] = len(unique_links) + 1
                            sources.append(f"{unique_links[uri]}. [{title}]({uri})")
                        
                        ref_num = unique_links[uri]
                        marker = f"[{i+1}]"
                        placeholder = f"@@@REF_{i+1}@@@"
                        if marker in text:
                            text = text.replace(marker, placeholder)
                            replacements[placeholder] = f"[[{ref_num}]]({uri})"
                            
                # Apply inline markdown link replacements securely
                for placeholder, md_link in replacements.items():
                    text = text.replace(placeholder, md_link)
                
                if sources:
                    text += "\n\n🔍 **Sources:**\n" + "\n".join(sources)
    except Exception as e:
        logger.error(f"Gemini error for {user_id}: {e}", exc_info=True)
        if os.getenv("ENVIRONMENT") == "development":
            return f"⚠️ **Dev Error (General):** {str(e)}"
        return "I'm having a hard time processing my thoughts right now. Give me a moment."

    try:
        # Save the AI's generated response to the DB afterward
        await asyncio.to_thread(save_message, user_id, "assistant", text)
    except Exception as e:
        logger.error(f"Assistant message save error for {user_id}: {e}", exc_info=True)

    if len(content.split()) > 3 or len(content) > 15 or in_crisis or "[RAPID BURST" in content or "[Voice" in content:
        task = asyncio.create_task(run_post_analysis_tasks(user_id, "message", content, text, profile, in_crisis))
        background_tasks.add(task)
        task.add_done_callback(background_tasks.discard)

    return text