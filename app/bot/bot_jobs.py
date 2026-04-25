"""Scheduled Background Tasks (Cron Jobs) for the Telegram Bot ecosystem."""
import logging
import asyncio
import random
import os
from datetime import datetime, timedelta
from google import genai
from google.genai import types
from app.db.database import get_null_episodes, update_episode_embedding, supabase, save_message, get_user_lock, get_profile, save_profile
from app.services.extractor import SAFETY_SETTINGS
from telegram.ext import ContextTypes
from tenacity import retry, stop_after_attempt, wait_exponential

logger = logging.getLogger(__name__)
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

# Registry for background tasks to avoid GC death
_bg_tasks = set()

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=8), reraise=True)
async def safe_generate_content(*args, **kwargs):
    """Protects proactive check-ins from network failures."""
    return await client.aio.models.generate_content(*args, **kwargs)

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=8), reraise=True)
async def safe_embed_content(*args, **kwargs):
    """Protects vector memory generation from network failures."""
    return await client.aio.models.embed_content(*args, **kwargs)

async def _process_user_checkin(context, user_id, data, days_silent, thresholds, now):
    """Handles individual proactive check-in asynchronously without blocking the main loop."""
    try:
        await context.bot.send_chat_action(chat_id=user_id, action="typing")
        await asyncio.sleep(random.uniform(2.5, 5.0))
        
        mood = data.get("current_mood_score", "unknown")
        unresolved = data.get("unresolved_threads", [])
        threads_ctx = f"Pending topics you guys left hanging: {unresolved}" if unresolved else ""
        events = data.get("upcoming_events", [])
        events_ctx = f"Upcoming/Recent events they had: {events}" if events else ""
        language = data.get("language", "their native language")

        if days_silent == thresholds[2]:
            prompt = f"The user {data.get('name', '')} has completely isolated themselves and ignored your check-ins for {days_silent} days. Their last mood was {mood}/10. Write a slightly more direct, 'tough love' but deeply caring message. Acknowledge that they are hiding/withdrawing, and tell them you are not going anywhere. CRITICAL: Respond entirely in {language}."
        else:
            prompt = f"The user {data.get('name', '')} hasn't spoken to you in {days_silent} days. Their last mood was {mood}/10. {threads_ctx}. {events_ctx}. Write a very short, warm, pressure-free message checking in. Sound like a friend who just thought of them. If they had an upcoming event, ASK ABOUT IT specifically. If they were sad last time, be gentle. CRITICAL: Respond entirely in {language}."
        
        response = await safe_generate_content(
            model="gemini-3.1-flash-lite-preview",
            contents=prompt,
            config=types.GenerateContentConfig(safety_settings=SAFETY_SETTINGS)
        )
        
        try:
            text = response.text.strip()
        except ValueError as e:
            logger.warning(f"Proactive check-in blocked by safety filters for {user_id}: {e}")
            return
            
        if not text:
            return
            
        await context.bot.send_message(chat_id=user_id, text=text)
        
        # CRITICAL MEMORY FIX: Save this to the chat history so MYRROR remembers reaching out
        await asyncio.to_thread(save_message, user_id, "assistant", f"[Proactive Check-in] {text}")
        
        async with get_user_lock(user_id):
            current_data = await asyncio.to_thread(get_profile, user_id)
            current_data["last_conversation"] = now.strftime("%Y-%m-%d %H:%M")
            await asyncio.to_thread(save_profile, user_id, current_data)
    except Exception as e:
        logger.error(f"Proactive job error for {user_id}: {e}")

async def proactive_check_job(context: ContextTypes.DEFAULT_TYPE):
    """Checks if users have been silent for 3+ days and reaches out to them."""
    now = datetime.now()
    
    # Query delegada a Supabase: Solo traer perfiles inactivos por al menos 2 días (umbral mínimo)
    cutoff_date = (now - timedelta(days=2)).strftime("%Y-%m-%d %H:%M")
    try:
        response = await asyncio.to_thread(
            lambda: supabase.table("profiles")
            .select("user_id, data")
            .lte("data->>last_conversation", cutoff_date)
            .limit(100)
            .execute()
        )
        profiles = response.data or []
    except Exception as e:
        logger.error(f"Proactive job db error: {e}")
        return
    
    for p in profiles:
        user_id = p["user_id"]
        data = p.get("data", {})
        last = data.get("last_conversation")
        if not last: continue
        
        try:
            last_dt = datetime.strptime(last, "%Y-%m-%d %H:%M")
            days_silent = (now - last_dt).days
            
            # DYNAMIC CHECK-IN: Adapt to attachment style and independence
            attachment = str(data.get("attachment_style", "")).lower()
            comm_style = str(data.get("communication_style", "")).lower()
            
            thresholds = [3, 7, 14]
            if "avoidant" in attachment or "independent" in comm_style:
                thresholds = [7, 14, 21] # Give them more space
            elif "anxious" in attachment or "clingy" in comm_style:
                thresholds = [2, 5, 10] # Reassure them sooner
            
            if days_silent in thresholds:
                # Dispatch as background task so we don't block the sequential job loop
                task = asyncio.create_task(_process_user_checkin(context, user_id, data, days_silent, thresholds, now))
                _bg_tasks.add(task)
                task.add_done_callback(_bg_tasks.discard)
        except Exception as e:
            logger.error(f"Proactive job error for {user_id}: {e}")

async def daily_maintenance_job(context: ContextTypes.DEFAULT_TYPE):
    """Runs daily to self-heal the semantic memory (generate missing embeddings)."""
    # Query delegada a Supabase: Extraer IDs únicos que realmente necesitan reparación
    try:
        response = await asyncio.to_thread(
            lambda: supabase.table("episodes")
            .select("user_id")
            .is_("embedding", "null")
            .limit(500)
            .execute()
        )
        if not response.data: 
            return
        user_ids = set(row["user_id"] for row in response.data)
    except Exception as e:
        logger.error(f"Maintenance job db error: {e}")
        return
        
    for user_id in user_ids:
        episodes = await asyncio.to_thread(get_null_episodes, user_id)
        if not episodes: continue
        
        fixed = 0
        for ep in episodes:
            event = ep.get("event")
            if not event: continue
            try:
                emb_res = await safe_embed_content(model="text-embedding-004", contents=event)
                if emb_res.embeddings:
                    await asyncio.to_thread(update_episode_embedding, ep["id"], emb_res.embeddings[0].values)
                    fixed += 1
                    await asyncio.sleep(0.5)
            except Exception as e:
                logger.error(f"Failed to fix memory {ep['id']}: {e}")
        if fixed > 0:
            logger.info(f"Auto-maintenance: Restored {fixed} fragmented memories for {user_id}.")