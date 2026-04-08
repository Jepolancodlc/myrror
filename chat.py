import logging
import os
import asyncio
from google import genai
from dotenv import load_dotenv
from prompt import SYSTEM_PROMPT
from database import get_profile, get_messages, save_message, get_all_people, search_similar_episodes
from extractor import get_profile_for_context, compress_history, run_post_analysis_tasks
import random
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
        result = await client.aio.models.generate_content(
            model="gemini-3.1-flash-lite-preview",
            contents=prompt
        )
        return result.text.strip()
    except Exception as e:
        logger.error(f"Context analysis error: {e}")
        return ""

async def get_response(user_id: str, content: str, new_session: bool = False) -> str:
    profile = await asyncio.to_thread(get_profile, user_id)
    
    # Obtener un contexto más amplio para compresión (30 mensajes)
    history = await asyncio.to_thread(get_messages, user_id, 30)
    recent_history = history[-10:] if history else []
    compressed_context = await compress_history(user_id, history, profile)

    ctx = SYSTEM_PROMPT

    # Time awareness
    now_dt = datetime.now()
    now_str = now_dt.strftime("%Y-%m-%d %H:%M")
    day_of_week = now_dt.strftime("%A")
    last = profile.get("last_conversation", "")
    ctx += f"\n\nCURRENT TIME: {now_str} ({day_of_week}). Use this time context naturally if relevant (e.g., late at night, early Monday morning)."
    if last:
        ctx += f"\nLAST CONVERSATION: {last}"

    if new_session:
        ctx += "\n\nThe user explicitly started a new session."

    # Crisis detection
    in_crisis = detect_crisis(content, recent_history, profile)
    if in_crisis:
        ctx += "\n\nCRISIS MODE: The user may be struggling right now. Do NOT analyze, push, or give advice. Only listen. Be warm, present, and human. Ask one simple question: are they okay. If things seem serious, gently mention that talking to someone real can help."

    # Adaptive Rhythm: Match the user's message length dynamically
    word_count = len(content.split())
    if word_count <= 10:
        ctx += "\n\nLENGTH MATCHING: The user sent a very short message. Keep your response brief, casual, and punchy. DO NOT write a wall of text. DO NOT over-analyze a simple statement."
    elif word_count >= 100:
        ctx += "\n\nLENGTH MATCHING: The user wrote a long, detailed message. Take your time to unpack it. A slightly longer, deeper, and highly structured response is appropriate here."

    # Conversational context (Moved UP to enhance RAG precision)
    conv_ctx = ""
    if recent_history and len(recent_history) >= 2:
        try:
            conv_ctx = await analyze_conversation_context(content, recent_history)
        except Exception as e:
            logger.error(f"Conversation context error: {e}")

    # Venting Detection
    if "[RAPID BURST OF MESSAGES - VENTING DETECTED]" in content:
        ctx += "\n\nURGENT CONTEXT: The user just sent multiple messages in rapid succession. They are likely venting, anxious, or overwhelmed. PROTOCOL: Do not interrupt or try to 'fix' their problem immediately. Validate their emotion, acknowledge the overwhelm, and provide a calm, grounding anchor."

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
            
    strategy = profile.get("myrror_strategy")
    if strategy:
        ctx += f"\n\nYOUR MASTER STRATEGY FOR THIS USER: {strategy}\nExecute this subtly but consistently."

    # Profile
    if profile:
        language = profile.get("language")
        if language:
            ctx += f"\n\nCRITICAL LANGUAGE INSTRUCTION: You MUST respond entirely in {language}."
            
        ctx += f"\n\nWHAT YOU KNOW ABOUT THIS USER (USE THIS INVISIBLY, DO NOT RECITE IT):\n{get_profile_for_context(profile, content)}"

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
            ctx += "\nModify your vocabulary, sentence length, and warmth to match this perfectly. Mirror their energy. If they text casually (e.g., lowercase, missing punctuation), match their texting style."
            
        job = profile.get("job")
        skills = profile.get("skills")
        culture = profile.get("cultural_background")
        if job or skills or culture:
            ctx += "\n\nWORLDVIEW & METAPHORS:"
            if job: ctx += f"\n- Profession/Job: {job}"
            if skills: ctx += f"\n- Skills/Interests: {skills}"
            if culture: ctx += f"\n- Cultural Background: {culture}"
            ctx += "\nADAPTATION: Speak their language. Use analogies, metaphors, and examples drawn directly from their specific profession, skills, and background to make your insights resonate deeply and feel tailor-made."

        pending = []
        if profile.get("upcoming_events"): pending.append(f"Events: {profile.get('upcoming_events')}")
        if profile.get("unresolved_threads"): pending.append(f"Unresolved Threads: {profile.get('unresolved_threads')}")
        if pending:
            ctx += f"\n\nPENDING TOPICS TO FOLLOW UP ON:\n" + "\n".join(pending)
            ctx += "\nIMPORTANT: If the timing feels natural, you MUST bring these up. Hold them accountable. Ask for updates. Never let them quietly abandon a topic."
            
        compass = profile.get("life_compass")
        if compass:
            ctx += f"\n\nTHEIR LIFE COMPASS (What grounds them / gives them meaning): {compass}"
            ctx += "\nIf the user feels lost, drifting, or hopeless, gently remind them of this core purpose. Guide them back to their center."
            
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

        ctx += "\n\nGUIDANCE PROTOCOL (SOCRATIC METHOD):"
        ctx += "\nDo not just give the user the answers or lecture them. Instead, ask the *one right question* that forces them to realize the truth themselves. Guide them to their own epiphanies."

    # Semantic RAG Memory Engine
    # Solo buscamos en la memoria si el mensaje tiene sustancia (evita buscar recuerdos para "sí", "ok", "jaja")
    if len(content.split()) > 3 or len(content) > 15 or conv_ctx:
        try:
            # HYPER-PRECISE RAG: Combine the high-level conversation summary with the raw message
            embed_text = f"Context: {conv_ctx}\nMessage: {content}"
            embed_text = embed_text if len(embed_text) <= 8000 else embed_text[:8000]
            
            emb_res = await client.aio.models.embed_content(
                model="text-embedding-004",
                contents=embed_text
            )
            if emb_res.embeddings:
                query_embedding = emb_res.embeddings[0].values
                relevant_episodes = await asyncio.to_thread(search_similar_episodes, user_id, query_embedding, limit=3)
                
                if relevant_episodes:
                    eps_list = []
                    now_date = datetime.now().date()
                    for ep in relevant_episodes:
                        ep_date_str = ep.get('created_at', '')[:10]
                        try:
                            ep_date = datetime.strptime(ep_date_str, "%Y-%m-%d").date()
                            days_ago = (now_date - ep_date).days
                            time_ctx = "today" if days_ago == 0 else ("yesterday" if days_ago == 1 else f"{days_ago} days ago")
                        except:
                            time_ctx = ep_date_str
                        eps_list.append(f"- [{time_ctx}] {ep.get('event')}")
                    eps_text = "\n".join(eps_list)
                    ctx += f"\n\nRELEVANT PAST MEMORIES (Triggered by what the user just said):\n{eps_text}"
                    ctx += "\nCRITICAL ADVICE PROTOCOL (STEALTH MEMORY): Use these past memories INVISIBLY. Sound like a human who just remembered something naturally. Say things like 'Wait, didn't this happen back when...?' or 'This reminds me of...'. Draw advice from their OWN past track record, and use exact quotes to break through denial. DO NOT sound like an AI retrieving data."
        except Exception as e:
            logger.error(f"RAG search error for {user_id}: {e}", exc_info=True)

    # People the user knows
    people = await asyncio.to_thread(get_all_people, user_id)
    if people:
        people_summary = "\n".join([
            f"- {p['name']} ({p.get('relationship', 'unknown')}): {p.get('notes', {}).get('description', '')}"
            for p in people
        ])
        ctx += f"\n\nPEOPLE IN THEIR LIFE:\n{people_summary}"

    # Inyectar el historial comprimido si existe
    if compressed_context:
        ctx += f"\n\nCOMPRESSED OLDER CONTEXT:\n{compressed_context}"

    if conv_ctx:
        ctx += f"\n\nCONVERSATIONAL CONTEXT: {conv_ctx}"

    # Recent history — last 10 only
    if recent_history:
        ctx += "\n\nRECENT HISTORY:\n"
        for msg in recent_history:
            role = "User" if msg["role"] == "user" else "MYRROR"
            safe_content = msg['content'] if len(msg['content']) <= 2500 else msg['content'][:2500] + "... [truncated]"
            ctx += f"{role}: {safe_content}\n"

    ctx += f"\nUser: {content}\nMYRROR:"

    try:
        response = await client.aio.models.generate_content(
            model="gemini-3.1-flash-lite-preview",
            contents=ctx
        )
        text = response.text
    except Exception as e:
        logger.error(f"Gemini error for {user_id}: {e}", exc_info=True)
        return "I'm having a hard time processing my thoughts right now. Give me a moment."

    try:
        # Grabación estricta y secuencial para preservar el orden cronológico
        await asyncio.to_thread(save_message, user_id, "user", content)
        await asyncio.to_thread(save_message, user_id, "assistant", text)
    except Exception as e:
        logger.error(f"Message save error for {user_id}: {e}", exc_info=True)

    if len(content.split()) > 3 or len(content) > 15 or in_crisis or "[RAPID BURST" in content or "[Voice" in content:
        asyncio.create_task(run_post_analysis_tasks(user_id, "message", content, text, profile, in_crisis))

    return text