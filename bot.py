import logging
import os
import time
import asyncio
import json
from telegram import Update
from telegram.ext import Application, MessageHandler, CommandHandler, filters, ContextTypes
from dotenv import load_dotenv
from database import get_profile, get_episodes, get_all_messages, get_all_people, get_all_profiles, supabase
from chat import get_response
from analyzer import analyze_image, analyze_document, analyze_voice
from extractor import (
    extract_and_save_profile, extract_episodes_from_content, 
    extract_people, generate_weekly_summary, generate_daily_summary
)
from google import genai
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

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# PROACTIVE ENGAGEMENT (BACKGROUND JOB)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

async def proactive_check_job(context: ContextTypes.DEFAULT_TYPE):
    """Checks if users have been silent for 3+ days and reaches out to them."""
    profiles = get_all_profiles()
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
            if days_silent in [3, 7]:
                prompt = f"The user {data.get('name', '')} hasn't spoken to you in {days_silent} days. Write a very short, warm, pressure-free message checking in. Don't ask for a big update, just let them know you're there if they need to talk."
                response = client.models.generate_content(model="gemini-3.1-flash-lite-preview", contents=prompt)
                await context.bot.send_message(chat_id=user_id, text=response.text.strip())
                # Update last conversation slightly so it doesn't trigger again today
                data["last_conversation"] = now.strftime("%Y-%m-%d %H:%M")
                supabase.table("profile").update({"data": data}).eq("user_id", user_id).execute()
        except Exception as e:
            logger.error(f"Proactive job error for {user_id}: {e}")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# COMMANDS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = str(update.effective_user.id)
    user_sessions[user_id] = True
    supabase.table("messages").delete().eq("user_id", user_id).execute()
    await update.message.reply_text("New session started. I'm here.")

async def profile_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = str(update.effective_user.id)
    profile = get_profile(user_id)
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
    profile = get_profile(user_id)
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
    episodes = get_episodes(user_id, limit=15)
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
    people = get_all_people(user_id)
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
    profile = get_profile(user_id)
    episodes = get_episodes(user_id, limit=20)

    if not profile:
        await update.message.reply_text("I don't know you well enough yet. Talk to me more first.")
        return

    await update.message.chat.send_action("typing")

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
        response = client.models.generate_content(
            model="gemini-3.1-flash-lite-preview",
            contents=prompt
        )
        await update.message.reply_text(response.text)
    except Exception as e:
        logger.error(f"Reflect command error: {e}")
        await update.message.reply_text("I had trouble generating your reflection. Try again in a moment.")

async def week_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = str(update.effective_user.id)
    profile = get_profile(user_id)
    messages = get_all_messages(user_id)
    episodes = get_episodes(user_id, limit=20)

    if not profile and not messages:
        await update.message.reply_text("Not enough data yet. Keep talking to me.")
        return

    await update.message.chat.send_action("typing")

    summary = await generate_weekly_summary(user_id, profile, messages, episodes)
    if summary:
        await update.message.reply_text(summary)
    else:
        await update.message.reply_text("I had trouble generating your weekly summary. Try again in a moment.")

async def checkin_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = str(update.effective_user.id)
    profile = get_profile(user_id)
    name = profile.get("name", "")

    greeting = f"Hey{' ' + name if name else'}'}."

    hour = datetime.now().hour
    if hour < 12:
        time_ref = "this morning"
    elif hour < 18:
        time_ref = "today"
    else:
        time_ref = "tonight"

    await update.message.reply_text(f"{greeting} How are you doing {time_ref}?")

async def contract_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = str(update.effective_user.id)
    profile = get_profile(user_id)
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
    supabase.table("profile").delete().eq("user_id", user_id).execute()
    supabase.table("messages").delete().eq("user_id", user_id).execute()
    await update.message.reply_text("Profile and history cleared. Starting fresh.")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# RELAPSE DETECTION
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def check_relapses(profile: dict) -> list:
    contracts = profile.get("personal_contracts", [])
    if not contracts:
        return []

    last = profile.get("last_conversation", "")
    if not last:
        return []

    try:
        last_dt = datetime.strptime(last, "%Y-%m-%d %H:%M")
        days_since = (datetime.now() - last_dt).days
    except:
        return []

    warnings = []
    gym_keywords = ["gym", "gimnasio", "workout", "ejercicio", "exercise", "training"]
    study_keywords = ["study", "estudiar", "learn", "aprender", "code", "coding"]
    porn_keywords = ["porn", "porno", "nofap", "no fap"]

    contracts_text = " ".join(str(c).lower() for c in contracts)

    if any(kw in contracts_text for kw in gym_keywords) and days_since >= 3:
        warnings.append(f"You committed to the gym but haven't mentioned it in {days_since} days.")
    if any(kw in contracts_text for kw in study_keywords) and days_since >= 3:
        warnings.append(f"You committed to studying but haven't mentioned it in {days_since} days.")
    if any(kw in contracts_text for kw in porn_keywords) and days_since >= 1:
        warnings.append(f"You made a commitment. How is it going?")

    return warnings

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
    try:
        text = await analyze_image(user_id, file_bytes, caption)
        profile = get_profile(user_id)
        asyncio.create_task(extract_and_save_profile(user_id, "image", caption, text, profile))
        await update.message.reply_text(text)
    except Exception as e:
        logger.error(f"Image handler error for {user_id}: {e}", exc_info=True)
        await update.message.reply_text("I had trouble reading that image. Try again in a moment.")

async def handle_voice(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = str(update.effective_user.id)
    await update.message.chat.send_action("typing")
    voice = update.message.voice
    file = await context.bot.get_file(voice.file_id)
    file_bytes = await file.download_as_bytearray()
    try:
        text = await analyze_voice(user_id, file_bytes, voice.mime_type or "audio/ogg")
        if text:
            content = f"[Voice Message] {text}"
            await process_message(update, user_id, content, False)
    except Exception as e:
        logger.error(f"Voice handler error for {user_id}: {e}", exc_info=True)
        await update.message.reply_text("I had trouble listening to your voice message. Can you type it?")

async def handle_document(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = str(update.effective_user.id)
    document = update.message.document
    mime = document.mime_type or ""
    caption = update.message.caption or "Analyze this file and tell me what you think."
    await update.message.chat.send_action("typing")
    file = await context.bot.get_file(document.file_id)
    file_bytes = await file.download_as_bytearray()
    try:
        text = await analyze_document(user_id, file_bytes, mime, document.file_name, caption)
        if not text:
            await update.message.reply_text("I can't read that file type yet. Try .txt, .pdf, or an image.")
            return
        profile = get_profile(user_id)
        asyncio.create_task(extract_and_save_profile(user_id, f"file:{document.file_name}", caption, text, profile))
        asyncio.create_task(extract_episodes_from_content(user_id, f"File: {caption}", text))
        await update.message.reply_text(text)
    except Exception as e:
        logger.error(f"Document handler error for {user_id}: {e}", exc_info=True)
        await update.message.reply_text("I had trouble reading that file. Try again in a moment.")

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
        await update.message.reply_text(text)

        # Relapse check on first message of the day
        if is_first_today:
            warnings = check_relapses(profile)
            if warnings:
                warning_text = "\n".join(f"— {w}" for w in warnings)
                await update.message.reply_text(warning_text)

            # Daily summary of yesterday
            messages = get_all_messages(user_id)
            if messages:
                asyncio.create_task(generate_daily_summary(user_id, profile, messages))

        # Learn from message
        asyncio.create_task(extract_episodes_from_content(user_id, content, text))
        asyncio.create_task(extract_people(user_id, content, text))

        # Weekly summary on Sundays
        if datetime.now().weekday() == 6 and is_first_today:
            messages = get_all_messages(user_id)
            episodes = get_episodes(user_id, limit=20)
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

def run_telegram_bot():
    async def start_bot():
        telegram_app = Application.builder().token(TELEGRAM_TOKEN).build()
        telegram_app.add_handler(CommandHandler("start", start))
        telegram_app.add_handler(CommandHandler("profile", profile_command))
        telegram_app.add_handler(CommandHandler("evolution", evolution_command))
        telegram_app.add_handler(CommandHandler("episodes", episodes_command))
        telegram_app.add_handler(CommandHandler("people", people_command))
        telegram_app.add_handler(CommandHandler("reflect", reflect_command))
        telegram_app.add_handler(CommandHandler("week", week_command))
        telegram_app.add_handler(CommandHandler("checkin", checkin_command))
        telegram_app.add_handler(CommandHandler("contract", contract_command))
        telegram_app.add_handler(CommandHandler("reset", reset_command))
        telegram_app.add_handler(CommandHandler("mood", mood_command))
        telegram_app.add_handler(CommandHandler("sos", sos_command))
        telegram_app.add_handler(MessageHandler(filters.VOICE, handle_voice))
        telegram_app.add_handler(MessageHandler(filters.PHOTO, handle_image))
        telegram_app.add_handler(MessageHandler(filters.Document.ALL, handle_document))
        telegram_app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
        
        # Run the proactive check every 12 hours
        telegram_app.job_queue.run_repeating(proactive_check_job, interval=3600 * 12)
        
        await telegram_app.initialize()
        await telegram_app.start()
        await telegram_app.updater.start_polling()
        await asyncio.Event().wait()

    asyncio.run(start_bot())
