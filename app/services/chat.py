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
from app.services.extractor import get_profile_for_context, compress_history, run_post_analysis_tasks, parse_json_response, get_rag_memories_text
import random
from datetime import datetime

load_dotenv()

logger = logging.getLogger(__name__)
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
REDIS_URL = os.getenv("REDIS_URL") or "redis://localhost:6379"
redis_client = redis.from_url(REDIS_URL)

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
    cached_name = await redis_client.get(f"gemini_cache:{user_id}")
    if cached_name:
        return cached_name.decode('utf-8')

    static_instruction = f"{SYSTEM_PROMPT}\n\nCOMPLETE USER PROFILE:\n{json.dumps(profile, ensure_ascii=False)}"
    history_text = "\n".join([f"{'User' if m['role'] == 'user' else 'MYRROR'}: {m['content']}" for m in full_history])
    
    try:
        cache = await client.aio.caches.create(
            model="gemini-1.5-flash-002", # 3.1-flash-lite-preview does not support caching
            config=types.CreateCacheConfig(
                system_instruction=static_instruction,
                contents=[history_text] if history_text else ["No history yet."],
                ttl="3600s" # Keeps the cache alive for 1 hour
            )
        )
        # Save in Redis with an expiration slightly shorter than Gemini's TTL (3300s = 55m)
        await redis_client.set(f"gemini_cache:{user_id}", cache.name, ex=3300)
        logger.info(f"Context Cache created for {user_id}: {cache.name}")
        return cache.name
    except Exception as e:
        # This will silently fail and fallback if the context is under 32,768 tokens
        logger.debug(f"Cache creation skipped (likely <32k tokens): {e}")
        return None

# Task registry: Holds strong references so Python's Garbage Collector doesn't kill async tasks mid-execution.
background_tasks = set()

async def detect_crisis(content: str, history: list, profile: dict) -> bool:
    """
    Dynamic Crisis Detection: Uses Gemini to evaluate implicit subtext and despair,
    falling back to safety heuristics if the API blocks the request due to self-harm filters.
    """
    history_text = "\n".join([f"{m['role']}: {m['content'][:100]}" for m in history[-3:]]) if history else ""
    prompt = f"""You are a clinical safety evaluator.
Determine if the user's message indicates an acute psychological crisis, severe despair, or risk of self-harm.
Look for implicit signs of giving up, extreme apathy, or dangerous distress. 
Do NOT flag normal venting, frustration, casual sadness, or gaming/work complaints (e.g., "I give up on this code").

RECENT CONTEXT:
{history_text}

USER'S NEW MESSAGE:
"{content}"

Respond EXACTLY with a single word: "true" (if crisis) or "false" (if not)."""

    try:
        response = await safe_generate_content(
            model="gemini-3.1-flash-lite-preview",
            contents=prompt
        )
        # Gemini Safety filters often block self-harm text entirely, returning no text.
        if not response.text:
            logger.warning("Crisis detection API blocked by safety filters. Defaulting to True.")
            return True
            
        return "true" in response.text.strip().lower()
    except Exception as e:
        logger.error(f"Dynamic crisis detection error: {e}")
        # Fallback Heuristic if API fails
        fallback_keywords = [
            "para qué", "no tiene sentido", "no vale la pena", "desaparecer", 
            "what's the point", "disappear", "me quiero morir", "no quiero vivir", 
            "i want to die", "kill myself", "end it all", "acabar con todo", 
            "suicide", "suicidio"
        ]
        return any(kw in content.lower() for kw in fallback_keywords)

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
    profile = await asyncio.to_thread(get_profile, user_id) or {}
    
    # Retrieve a broader context window (30 messages) for potential history compression
    history = await asyncio.to_thread(get_messages, user_id, 30)
    recent_history = history[-10:] if history else []
    compressed_context = await compress_history(user_id, history, profile)

    # --- SEMANTIC ROUTING (DYNAMIC DOMAINS) ---
    conv_data = {"summary": "", "domains": []}
    if recent_history and len(recent_history) >= 1:
        try:
            conv_data = await analyze_conversation_context(content, recent_history)
        except Exception as e:
            logger.error(f"Conversation context error: {e}")

    conv_ctx = conv_data.get("summary", "")
    active_domains = conv_data.get("domains", [])

    # 1. Attempt to get cache (pass older history, excluding the last 10 messages)
    older_history = history[:-10] if len(history) > 10 else []
    cache_name = await get_or_create_context_cache(user_id, profile, older_history)

    ctx = "" if cache_name else SYSTEM_PROMPT

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
    in_crisis = await detect_crisis(content, recent_history, profile)
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
            
        ctx += f"\n\nWHAT YOU KNOW ABOUT THIS USER (USE THIS INVISIBLY, DO NOT RECITE IT):\n{get_profile_for_context(profile, active_domains=active_domains)}"

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
    # Semantic RAG Trigger: Only run costly vector searches if the message has substance (ignores "yes", "ok").
    if len(content.split()) > 3 or len(content) > 15 or conv_ctx:
        try:
            # HYPER-PRECISE RAG: Combine the high-level conversation summary with the raw message
            embed_text = f"Context: {conv_ctx}\nMessage: {content}"
            embed_text = embed_text if len(embed_text) <= 8000 else embed_text[:8000]
            eps_text = await get_rag_memories_text(user_id, embed_text)
            if eps_text:
                memory_block.append(f"RELEVANT PAST MEMORIES (Triggered by the user's message):\n{eps_text}\nSTEALTH MEMORY: Use these invisibly. Sound like a human who just remembered something naturally. Draw advice from their OWN past track record.")
        except Exception as e:
            logger.error(f"RAG search error for {user_id}: {e}", exc_info=True)

    people = await asyncio.to_thread(get_all_people, user_id)
    if people:
        recent_text = (conv_ctx + " " + content).lower()
        relevant_people = []
        for p in people:
            name_lower = p['name'].lower()
            if name_lower in recent_text or len(people) <= 5 or "relationships" in active_domains:
                relevant_people.append(p)
                
        if relevant_people:
            people_summary = "\n".join([f"- {p['name']} ({p.get('relationship', 'unknown')}): {p.get('notes', {}).get('description', '')}" for p in relevant_people])
            memory_block.append(f"RELEVANT PEOPLE IN THEIR LIFE:\n{people_summary}")

    # --- 5. HISTORY ---
    history_block = []
    if compressed_context:
        history_block.append(f"COMPRESSED OLDER CONTEXT:\n{compressed_context}")
    if conv_ctx:
        history_block.append(f"CONVERSATIONAL CONTEXT: {conv_ctx}")
    if recent_history:
        recent_str = "\n".join([f"{'User' if msg['role'] == 'user' else 'MYRROR'}: {msg['content'][:2500]}" for msg in recent_history])
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
        gen_model = "gemini-1.5-flash-002" if cache_name else "gemini-3.1-flash-lite-preview"
        config_kwargs = {"tools": [{"google_search": {}}]}
        if cache_name:
            config_kwargs["cached_content"] = cache_name
            
        response = await safe_generate_content(
            model=gen_model,
            contents=ctx,
            config=types.GenerateContentConfig(**config_kwargs)
        )
        text = response.text or "I'm having a hard time processing my thoughts right now. Give me a moment."
        
        # Remove the internal monologue (<thought>) BEFORE saving to DB to prevent memory pollution
        text = re.sub(r'<thought>.*?</thought>', '', text, flags=re.DOTALL).strip()
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