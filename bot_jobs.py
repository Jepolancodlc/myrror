import logging
import asyncio
import random
import os
from datetime import datetime
from google import genai
from database import get_all_profiles, get_null_episodes, update_episode_embedding, supabase
from telegram.ext import ContextTypes

logger = logging.getLogger(__name__)
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

async def proactive_check_job(context: ContextTypes.DEFAULT_TYPE):
    """Checks if users have been silent for 3+ days and reaches out to them."""
    profiles = await asyncio.to_thread(get_all_profiles)
    now = datetime.now()
    
    for p in profiles:
        user_id = p["user_id"]
        data = p.get("data", {})
        last = data.get("last_conversation")
        if not last: continue
        
        try:
            last_dt = datetime.strptime(last, "%Y-%m-%d %H:%M")
            days_silent = (now - last_dt).days
            
            if days_silent in [3, 7, 14]:
                await context.bot.send_chat_action(chat_id=user_id, action="typing")
                await asyncio.sleep(random.uniform(2.5, 5.0))
                
                if days_silent == 14:
                    prompt = f"The user {data.get('name', '')} has completely isolated themselves and ignored your check-ins for 2 weeks. Write a slightly more direct, 'tough love' but deeply caring message. Acknowledge that they are hiding/withdrawing, and tell them you are not going anywhere and will be here when they are ready to face things. CRITICAL: Respond entirely in the user's native language."
                else:
                    prompt = f"The user {data.get('name', '')} hasn't spoken to you in {days_silent} days. Write a very short, warm, pressure-free message checking in. Don't ask for a big update, just let them know you're there if they need to talk. CRITICAL: Respond entirely in the user's native language."
                
                response = await client.aio.models.generate_content(model="gemini-3.1-flash-lite-preview", contents=prompt)
                await context.bot.send_message(chat_id=user_id, text=response.text.strip())
                data["last_conversation"] = now.strftime("%Y-%m-%d %H:%M")
                await asyncio.to_thread(lambda: supabase.table("profile").update({"data": data}).eq("user_id", user_id).execute())
        except Exception as e:
            logger.error(f"Proactive job error for {user_id}: {e}")

async def daily_maintenance_job(context: ContextTypes.DEFAULT_TYPE):
    """Runs daily to self-heal the semantic memory (generate missing embeddings)."""
    profiles = await asyncio.to_thread(get_all_profiles)
    for p in profiles:
        user_id = p["user_id"]
        episodes = await asyncio.to_thread(get_null_episodes, user_id)
        if not episodes: continue
        
        fixed = 0
        for ep in episodes:
            event = ep.get("event")
            if not event: continue
            try:
                emb_res = await client.aio.models.embed_content(model="text-embedding-004", contents=event)
                if emb_res.embeddings:
                    await asyncio.to_thread(update_episode_embedding, ep["id"], emb_res.embeddings[0].values)
                    fixed += 1
                    await asyncio.sleep(0.5)
            except Exception as e:
                logger.error(f"Failed to fix memory {ep['id']}: {e}")
        if fixed > 0:
            logger.info(f"Auto-maintenance: Restored {fixed} fragmented memories for {user_id}.")