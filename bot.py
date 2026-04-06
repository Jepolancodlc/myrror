import logging
import os
import time
import asyncio
import json
import random
from telegram import Update
from telegram.ext import Application, MessageHandler, CommandHandler, CallbackQueryHandler, filters, ContextTypes
from telegram import InlineKeyboardButton, InlineKeyboardMarkup
from dotenv import load_dotenv
from database import get_profile, get_episodes, get_messages, get_all_people, get_all_profiles, supabase
from chat import get_response
from analyzer import analyze_image, analyze_document, analyze_voice
from extractor import (
    extract_and_save_profile, extract_episodes_from_content,
    track_evolution,
    extract_people, generate_weekly_summary, generate_daily_summary
)
from google import genai
from google.genai import types
from datetime import datetime

load_dotenv()

logger = logging.getLogger(__name__)
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

user_sessions = {}
user_last_message = {}
user_pending = {}

COOLDOWN_SECONDS = 3
MIN_MESSAGE_LENGTH = 2

def get_mood_keyboard():
    return InlineKeyboardMarkup([
        [
            InlineKeyboardButton("1 ⬛", callback_data="mood_1"),
            InlineKeyboardButton("2", callback_data="mood_2"),
            InlineKeyboardButton("3", callback_data="mood_3"),
            InlineKeyboardButton("4", callback_data="mood_4"),
            InlineKeyboardButton("5 🟨", callback_data="mood_5"),
        ],
        [
            InlineKeyboardButton("6", callback_data="mood_6"),
            InlineKeyboardButton("7", callback_data="mood_7"),
            InlineKeyboardButton("8", callback_data="mood_8"),
            InlineKeyboardButton("9", callback_data="mood_9"),
            InlineKeyboardButton("10 🟩", callback_data="mood_10"),
        ]
    ])

async def mood_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = str(update.effective_user.id)
    profile = await asyncio.to_thread(get_profile, user_id)
    evolution = profile.get("evolution", [])

    mood_data = [e for e in evolution if e.get("field") == "current_mood_score"]
    if len(mood_data) < 2:
        await update.message.reply_text("I need more conversations with you to track your mood evolution. Keep talking to me!")
        return

    try:
        import matplotlib.pyplot as plt
        import io
        
        dates = [e["date"][5:] for e in mood_data[-14:]] # Last 14 changes
        scores = [float(str(e["to"]).replace(',', '.')) for e in mood_data[-14:] if str(e["to"]).replace('.', '', 1).isdigit()]

        plt.figure(figsize=(8, 4))
        plt.plot(dates[:len(scores)], scores, marker='o', color='#4A90E2', linestyle='-', linewidth=2)
        plt.ylim(0, 10)
        plt.title("Your Emotional Evolution")
        plt.ylabel("Mood Score (1-10)")
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()

        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        plt.close()

        await update.message.reply_photo(photo=buf, caption="Here is how your mood has been trending.")
    except ImportError:
        await update.message.reply_text("Visual tracking requires matplotlib. (Run: pip install matplotlib)")
    except Exception as e:
        logger.error(f"Mood graph error: {e}")
        await update.message.reply_text("I couldn't generate the mood graph right now.")

async def sos_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = (
        "🚨 **CRISIS RESOURCES** 🚨\n\n"
        "I am an AI. I care about you, but I cannot replace real medical or psychological help.\n"
        "If you feel you are in danger, please reach out to humans who can help you right now:\n\n"
        "• **Emergency:** Call 112 (Europe) or 911 (Americas).\n"
        "• **Crisis Text Line:** Text HOME to 741741\n"
        "• **Global Helplines:** https://findahelpline.com/\n\n"
        "You are not alone. Please talk to a professional."
    )
    await update.message.reply_text(text, parse_mode="Markdown")

async def export_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = str(update.effective_user.id)
    profile = await asyncio.to_thread(get_profile, user_id)
    episodes = await asyncio.to_thread(get_episodes, user_id, limit=100)
    
    data = {"profile": profile, "episodes": episodes}
    file_bytes = json.dumps(data, indent=2, ensure_ascii=False).encode('utf-8')
    
    await update.message.reply_document(
        document=file_bytes,
        filename=f"myrror_backup.json",
        caption="Here is a complete backup of your psychological profile and episodes."
    )

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# PROACTIVE ENGAGEMENT (BACKGROUND JOB)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

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
            
            # Reach out exactly on the 3rd and 7th day of silence
            if days_silent in [3, 7, 14]:
                # Simulate human typing delay
                await context.bot.send_chat_action(chat_id=user_id, action="typing")
                await asyncio.sleep(random.uniform(2.5, 5.0))
                
                if days_silent == 14:
                    prompt = f"The user {data.get('name', '')} has completely isolated themselves and ignored your check-ins for 2 weeks. Write a slightly more direct, 'tough love' but deeply caring message. Acknowledge that they are hiding/withdrawing, and tell them you are not going anywhere and will be here when they are ready to face things."
                else:
                    prompt = f"The user {data.get('name', '')} hasn't spoken to you in {days_silent} days. Write a very short, warm, pressure-free message checking in. Don't ask for a big update, just let them know you're there if they need to talk."
                
                response = await client.aio.models.generate_content(model="gemini-3.1-flash-lite-preview", contents=prompt)
                await context.bot.send_message(chat_id=user_id, text=response.text.strip())
                # Update last conversation slightly so it doesn't trigger again today
                data["last_conversation"] = now.strftime("%Y-%m-%d %H:%M")
                await asyncio.to_thread(lambda: supabase.table("profile").update({"data": data}).eq("user_id", user_id).execute())
        except Exception as e:
            logger.error(f"Proactive job error for {user_id}: {e}")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# DAILY MAINTENANCE (BACKGROUND JOB)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

async def daily_maintenance_job(context: ContextTypes.DEFAULT_TYPE):
    """Runs daily to self-heal the semantic memory (generate missing embeddings)."""
    profiles = await asyncio.to_thread(get_all_profiles)
    from database import get_null_episodes, update_episode_embedding
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
                    await asyncio.sleep(0.5) # Anti-rate limit protection
            except Exception as e:
                logger.error(f"Failed to fix memory {ep['id']}: {e}")
        if fixed > 0:
            logger.info(f"Auto-maintenance: Restored {fixed} fragmented memories for {user_id}.")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# COMMANDS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = str(update.effective_user.id)
    user_sessions[user_id] = True
    await asyncio.to_thread(lambda: supabase.table("messages").delete().eq("user_id", user_id).execute())
    welcome_text = (
        "Hello. I am MYRROR.\n\n"
        "I am not a standard assistant. I am here to be a mirror for your mind, "
        "to help you track your growth, and to remember what matters to you.\n\n"
        "You can text me, send me voice notes, images, or documents. "
        "Over time, I will learn your patterns, hold you to your commitments, and "
        "support you when things get heavy.\n\n"
        "Tell me, what brings you here today?"
    )
    await update.message.reply_text(welcome_text)

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = (
        "Here is what you can ask me to do:\n\n"
        "🧠 **Self-Discovery**\n"
        "• /profile - See what I know about your personality & goals\n"
        "• /evolution - Track how you've changed over time\n"
        "• /episodes - View the significant moments of your life\n"
        "• /people - See who I remember from your stories\n"
        "• /mood - View a visual graph of your emotional evolution\n\n"
        "🪞 **Reflection & Action**\n"
        "• /reflect - Ask for a deep, honest reflection on where you are\n"
        "• /flashback - Revisit a random past memory we've discussed\n"
        "• /week - Get a summary of your week's patterns and commitments\n"
        "• /contract - See the personal rules you've asked me to hold you to\n\n"
        "⚙️ **System & Support**\n"
        "• /sos - Get emergency resources if you're in crisis\n"
        "• /export - Download a full backup of your data\n"
        "• /reset - Erase all your history and start completely fresh"
    )
    await update.message.reply_text(text, parse_mode="Markdown")

async def profile_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = str(update.effective_user.id)
    profile = await asyncio.to_thread(get_profile, user_id)
    if not profile:
        await update.message.reply_text("I don't know anything about you yet. Talk to me first.")
        return
    skip = ["evolution", "confidence", "last_conversation", "total_conversations"]
    lines = ["Here's what I know about you:\n"]
    for key, value in profile.items():
        if key not in skip and value:
            lines.append(f"• {key}: {value}")
    if "last_conversation" in profile:
        lines.append(f"\nLast conversation: {profile['last_conversation']}")
    if "total_conversations" in profile:
        lines.append(f"Total conversations: {profile['total_conversations']}")
    await update.message.reply_text("\n".join(lines))

async def evolution_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = str(update.effective_user.id)
    profile = await asyncio.to_thread(get_profile, user_id)
    evolution = profile.get("evolution", [])
    if not evolution:
        await update.message.reply_text("No changes tracked yet. Keep talking to me.")
        return
    lines = ["Your evolution over time:\n"]
    for e in evolution[-10:]:
        lines.append(f"• {e['date']} — {e['field']}: {e['note']}")
    await update.message.reply_text("\n".join(lines))

async def episodes_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = str(update.effective_user.id)
    episodes = await asyncio.to_thread(get_episodes, user_id, limit=15)
    if not episodes:
        await update.message.reply_text("No significant episodes recorded yet.")
        return
    lines = ["Your story so far:\n"]
    for ep in reversed(episodes):
        date = ep.get("created_at", "")[:10]
        event = ep.get("event", "")
        domain = ep.get("domain", "")
        impact = ep.get("impact", "")
        if not event.startswith("Daily summary") and not event.startswith("Weekly summary"):
            lines.append(f"• {date} [{domain}] {event} ({impact})")
    if len(lines) == 1:
        await update.message.reply_text("No significant episodes recorded yet.")
        return
    await update.message.reply_text("\n".join(lines))

async def people_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = str(update.effective_user.id)
    people = await asyncio.to_thread(get_all_people, user_id)
    if not people:
        await update.message.reply_text("I don't know anyone in your life yet. Tell me about the people around you.")
        return
    lines = ["People in your life:\n"]
    for p in people:
        name = p.get("name", "")
        rel = p.get("relationship", "")
        notes = p.get("notes", {})
        desc = notes.get("description", "") if isinstance(notes, dict) else ""
        lines.append(f"• {name} ({rel}){' — ' + desc if desc else ''}")
    await update.message.reply_text("\n".join(lines))

async def reflect_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = str(update.effective_user.id)
    profile = await asyncio.to_thread(get_profile, user_id)
    episodes = await asyncio.to_thread(get_episodes, user_id, limit=20)

    if not profile:
        await update.message.reply_text("I don't know you well enough yet. Talk to me more first.")
        return

    status_msg = await update.message.reply_text("🪞 *Reflecting on your journey...*", parse_mode="Markdown")

    episodes_text = "\n".join([
        f"- [{ep.get('domain')}] {ep.get('event')} ({ep.get('created_at', '')[:10]})"
        for ep in reversed(episodes)
        if not ep.get("event", "").startswith("Daily summary")
        and not ep.get("event", "").startswith("Weekly summary")
    ]) or "No significant episodes recorded yet."

    prompt = f"""You are MYRROR. Generate a deep, honest reflection for this person.

PROFILE:
{json.dumps(profile, ensure_ascii=False, separators=(',', ':'))}

SIGNIFICANT EPISODES:
{episodes_text}

Write a personal reflection:
1. Who they are right now — honestly, not flattering
2. Patterns you've noticed that they might not see
3. What they've done well
4. What they keep avoiding
5. One uncomfortable question they should sit with

Be direct. Be human. Be specific.
Respond in the user's language.
"""

    try:
        response = await client.aio.models.generate_content(
            model="gemini-3.1-flash-lite-preview",
            contents=prompt
        )
        await status_msg.edit_text(response.text)
    except Exception as e:
        logger.error(f"Reflect command error: {e}")
        await status_msg.edit_text("I had trouble generating your reflection. Try again in a moment.")

async def week_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = str(update.effective_user.id)
    profile = await asyncio.to_thread(get_profile, user_id)
    messages = await asyncio.to_thread(get_messages, user_id, 40)
    episodes = await asyncio.to_thread(get_episodes, user_id, limit=20)

    if not profile and not messages:
        await update.message.reply_text("Not enough data yet. Keep talking to me.")
        return

    status_msg = await update.message.reply_text("📅 *Reviewing your week...*", parse_mode="Markdown")

    summary = await generate_weekly_summary(user_id, profile, messages, episodes)
    if summary:
        await status_msg.edit_text(summary)
    else:
        await status_msg.edit_text("I had trouble generating your weekly summary. Try again in a moment.")

async def mood_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    user_id = str(query.from_user.id)
    mood_val = query.data.split("_")[1]
    
    profile = await asyncio.to_thread(get_profile, user_id)
    new_data = {"current_mood_score": int(mood_val)}
    
    evolution = track_evolution(profile, new_data)
    profile["current_mood_score"] = int(mood_val)
    if evolution:
        profile["evolution"] = evolution
        
    from database import save_profile
    await asyncio.to_thread(save_profile, user_id, profile)
    
    response_texts = {
        "low": "You clicked low... I'm sorry things are heavy right now. I'm here if you want to vent.",
        "mid": "Right down the middle. Surviving the day. Anything specific on your mind?",
        "high": "Glad to see you're doing well! Tell me what's making it a good day."
    }
    val = int(mood_val)
    category = "low" if val <= 4 else ("mid" if val <= 7 else "high")
    
    await query.edit_message_text(text=f"Mood recorded: {mood_val}/10.\n\n{response_texts[category]}")

async def contract_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = str(update.effective_user.id)
    profile = await asyncio.to_thread(get_profile, user_id)
    contracts = profile.get("personal_contracts", None)
    if not contracts:
        await update.message.reply_text(
            "No personal contracts yet.\n\n"
            "Tell me something like:\n"
            "'Don't let me justify skipping the gym'\n"
            "I'll remember and enforce it."
        )
        return
    lines = ["Your personal contracts:\n"]
    if isinstance(contracts, list):
        for c in contracts:
            lines.append(f"• {c}")
    else:
        lines.append(str(contracts))
    await update.message.reply_text("\n".join(lines))

async def reset_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = str(update.effective_user.id)
    await asyncio.to_thread(lambda: supabase.table("profile").delete().eq("user_id", user_id).execute())
    await asyncio.to_thread(lambda: supabase.table("messages").delete().eq("user_id", user_id).execute())
    await update.message.reply_text("Profile and history cleared. Starting fresh.")

async def flashback_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = str(update.effective_user.id)
    episodes = await asyncio.to_thread(get_episodes, user_id, limit=50)
    
    # Skip recent ones and system summaries
    valid_episodes = [ep for ep in episodes[10:] if not ep.get("event", "").startswith("Daily summary") and not ep.get("event", "").startswith("Weekly summary")]
    
    if not valid_episodes:
        await update.message.reply_text("We haven't shared enough history yet for a flashback. Let's make some memories first.")
        return
        
    episode = random.choice(valid_episodes)
    event = episode.get("event", "")
    date = episode.get("created_at", "")[:10]
    
    prompt = f"The user experienced this event on {date}: '{event}'. Ask a deeply thoughtful, curious question about how they feel about it now, or how it shaped them since then. Keep it to one brief paragraph."
    await update.message.chat.send_action("typing")
    try:
        response = await client.aio.models.generate_content(model="gemini-3.1-flash-lite-preview", contents=prompt)
        await update.message.reply_text(response.text.strip())
    except Exception as e:
        logger.error(f"Flashback error: {e}")
        await update.message.reply_text(f"I was just thinking about when you mentioned: '{event}' ({date}). How do you feel about that now?")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# RELAPSE DETECTION
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# FILE AND IMAGE HANDLERS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

async def handle_image(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = str(update.effective_user.id)
    await update.message.chat.send_action("typing")
    photo = update.message.photo[-1]
    file = await context.bot.get_file(photo.file_id)
    file_bytes = await file.download_as_bytearray()
    caption = update.message.caption or "Analyze this image and tell me what you think."
    status_msg = await update.message.reply_text("👁️ *Analyzing image...*", parse_mode="Markdown")
    try:
        text = await analyze_image(user_id, file_bytes, caption)
        profile = get_profile(user_id)
        asyncio.create_task(extract_and_save_profile(user_id, "image", caption, text, profile))
        asyncio.create_task(extract_episodes_from_content(user_id, f"Image: {caption}", text))
        asyncio.create_task(extract_people(user_id, f"Image: {caption}", text))
        await status_msg.edit_text(text)
    except Exception as e:
        logger.error(f"Image handler error for {user_id}: {e}", exc_info=True)
        await status_msg.edit_text("I had trouble reading that image. Try again in a moment.")

async def handle_voice(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = str(update.effective_user.id)
    await update.message.chat.send_action("typing")
    status_msg = await update.message.reply_text("🎧 *Listening to your voice...*", parse_mode="Markdown")
    voice = update.message.voice
    file = await context.bot.get_file(voice.file_id)
    file_bytes = await file.download_as_bytearray()
    try:
        text = await analyze_voice(user_id, file_bytes, voice.mime_type or "audio/ogg")
        await status_msg.delete()
        if text:
            content = f"[Voice Message] {text}"
            await process_message(update, user_id, content, False)
    except Exception as e:
        logger.error(f"Voice handler error for {user_id}: {e}", exc_info=True)
        await status_msg.edit_text("I had trouble listening to your voice message. Can you type it?")

async def handle_document(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = str(update.effective_user.id)
    document = update.message.document
    mime = document.mime_type or ""
    caption = update.message.caption or "Analyze this file and tell me what you think."
    await update.message.chat.send_action("typing")
    status_msg = await update.message.reply_text("📄 *Reading document...*", parse_mode="Markdown")
    file = await context.bot.get_file(document.file_id)
    file_bytes = await file.download_as_bytearray()
    try:
        text = await analyze_document(user_id, file_bytes, mime, document.file_name, caption)
        if not text:
            await status_msg.edit_text("I can't read that file type yet. Try .txt, .pdf, or an image.")
            return
        profile = get_profile(user_id)
        asyncio.create_task(extract_and_save_profile(user_id, f"file:{document.file_name}", caption, text, profile))
        asyncio.create_task(extract_episodes_from_content(user_id, f"File: {caption}", text))
        asyncio.create_task(extract_people(user_id, f"File: {document.file_name} - {caption}", text))
        await status_msg.edit_text(text)
    except Exception as e:
        logger.error(f"Document handler error for {user_id}: {e}", exc_info=True)
        await status_msg.edit_text("I had trouble reading that file. Try again in a moment.")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# MESSAGE HANDLER
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

async def process_message(update: Update, user_id: str, content: str, is_new_session: bool):
    await update.message.chat.send_action("typing")

    try:
        profile = get_profile(user_id)

        # Check if first message of the day
        last = profile.get("last_conversation", "")
        is_first_today = False
        if last:
            try:
                last_dt = datetime.strptime(last, "%Y-%m-%d %H:%M")
                is_first_today = last_dt.date() < datetime.now().date()
            except:
                pass

        text = await get_response(user_id, content, is_new_session)
        
        lower_text = text.lower()
        mood_triggers = [
            "how are you feeling", "cómo te sientes", "how do you feel", 
            "cómo estás", "how are you today", "del 1 al 10",
            "cómo te encuentras", "how are things today", "qué tal tu día", "how was your day"
        ]
        if any(kw in lower_text for kw in mood_triggers) and "?" in text:
            await update.message.reply_text(text, reply_markup=get_mood_keyboard())
        else:
            await update.message.reply_text(text)

        if is_first_today:
            # Daily summary of yesterday
            messages = await asyncio.to_thread(get_messages, user_id, 20)
            if messages:
                asyncio.create_task(generate_daily_summary(user_id, profile, messages))

        # Learn from message
        asyncio.create_task(extract_episodes_from_content(user_id, content, text))
        asyncio.create_task(extract_people(user_id, content, text))

        # Weekly summary on Sundays
        if datetime.now().weekday() == 6 and is_first_today:
            messages = await asyncio.to_thread(get_messages, user_id, 40)
            episodes = await asyncio.to_thread(get_episodes, user_id, limit=20)
            asyncio.create_task(generate_weekly_summary(user_id, profile, messages, episodes))

    except Exception as e:
        logger.error(f"Message processing error for {user_id}: {e}", exc_info=True)
        await update.message.reply_text("I'm having trouble thinking right now. Try again in a moment.")

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = str(update.effective_user.id)
    content = update.message.text.strip()

    if len(content) < MIN_MESSAGE_LENGTH:
        return

    now = time.time()
    last = user_last_message.get(user_id, 0)
    if now - last < COOLDOWN_SECONDS:
        if user_id in user_pending:
            user_pending[user_id].cancel()

    user_last_message[user_id] = now
    is_new_session = user_sessions.pop(user_id, False)

    async def debounced():
        await update.message.chat.send_action("typing")
        # Simulate human reading time based on message length (min 1s, max 4s)
        reading_delay = max(1.0, min(len(content) * 0.02, 4.0))
        await asyncio.sleep(reading_delay)
        await process_message(update, user_id, content, is_new_session)

    task = asyncio.create_task(debounced())
    user_pending[user_id] = task

    try:
        await task
    except asyncio.CancelledError:
        pass
    finally:
        if user_pending.get(user_id) == task:
            del user_pending[user_id]

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# BOT STARTUP
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

async def start_telegram_bot():
    telegram_app = Application.builder().token(TELEGRAM_TOKEN).build()
    telegram_app.add_handler(CommandHandler("start", start))
    telegram_app.add_handler(CallbackQueryHandler(mood_callback, pattern="^mood_"))
    telegram_app.add_handler(CommandHandler("help", help_command))
    telegram_app.add_handler(CommandHandler("profile", profile_command))
    telegram_app.add_handler(CommandHandler("evolution", evolution_command))
    telegram_app.add_handler(CommandHandler("episodes", episodes_command))
    telegram_app.add_handler(CommandHandler("people", people_command))
    telegram_app.add_handler(CommandHandler("reflect", reflect_command))
    telegram_app.add_handler(CommandHandler("flashback", flashback_command))
    telegram_app.add_handler(CommandHandler("week", week_command))
    telegram_app.add_handler(CommandHandler("contract", contract_command))
    telegram_app.add_handler(CommandHandler("reset", reset_command))
    telegram_app.add_handler(CommandHandler("export", export_command))
    telegram_app.add_handler(CommandHandler("mood", mood_command))
    telegram_app.add_handler(CommandHandler("sos", sos_command))
    telegram_app.add_handler(MessageHandler(filters.VOICE, handle_voice))
    telegram_app.add_handler(MessageHandler(filters.PHOTO, handle_image))
    telegram_app.add_handler(MessageHandler(filters.Document.ALL, handle_document))
    telegram_app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    
    # Run the proactive check every 4 hours
    if telegram_app.job_queue:
        telegram_app.job_queue.run_repeating(proactive_check_job, interval=3600 * 4)
        # Run daily maintenance (self-healing memory)
        telegram_app.job_queue.run_repeating(daily_maintenance_job, interval=3600 * 24)
    else:
        logger.warning("JobQueue is None. Proactive background jobs are disabled. Install 'python-telegram-bot[job-queue]'.")
    
    await telegram_app.bot.delete_webhook(drop_pending_updates=True)
    await telegram_app.initialize()
    await telegram_app.start()
    await telegram_app.updater.start_polling()
    logger.info("Telegram bot started successfully in the main event loop.")
