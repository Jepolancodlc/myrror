import logging
import os
import time
import asyncio
import json
import random
import math
from telegram import Update
from telegram.ext import Application, MessageHandler, CommandHandler, CallbackQueryHandler, filters, ContextTypes
from telegram import InlineKeyboardButton, InlineKeyboardMarkup
from dotenv import load_dotenv
from database import get_profile, get_episodes, get_messages, get_all_people, supabase
from chat import get_response
from analyzer import analyze_image, analyze_document, analyze_voice
from extractor import generate_daily_summary, generate_weekly_summary, run_post_analysis_tasks
from bot_commands import (get_mood_keyboard, mood_command, sos_command, export_command, help_command, profile_command, evolution_command, episodes_command, people_command, reflect_command, week_command, mood_callback, contract_command, reset_command, flashback_command, dossier_command, setcompass_command)
from bot_jobs import proactive_check_job, daily_maintenance_job
from google import genai
from google.genai import types
from pydantic import BaseModel, Field
from datetime import datetime

load_dotenv()

logger = logging.getLogger(__name__)
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

user_sessions = {}
user_pending = {}
user_message_buffers = {}
user_quiz_options = {}
telegram_app = None

COOLDOWN_SECONDS = 3
MIN_MESSAGE_LENGTH = 2


class QuizSchema(BaseModel):
    question: str
    options: list[str] = Field(description="Exactly 4 options")

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


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# PSYCHOLOGICAL QUIZ
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

async def quiz_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = str(update.effective_user.id)
    profile = await asyncio.to_thread(get_profile, user_id)
    await update.message.chat.send_action("typing")

    prompt = f"""Generate a deep, engaging psychological multiple-choice question to learn something new about this user.
Make it practical, situational, or metaphoric. Target their blind spots or shadow traits.
User Profile: {json.dumps(profile)}"""
    try:
        response = await client.aio.models.generate_content(
            model="gemini-3.1-flash-lite-preview",
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=QuizSchema
            )
        )
        data = json.loads(response.text)
        
        keyboard = []
        for i, opt in enumerate(data["options"]):
            keyboard.append([InlineKeyboardButton(opt[:50] + ("..." if len(opt)>50 else ""), callback_data=f"quiz_{i}")])
        
        user_quiz_options[user_id] = data["options"]
        await update.message.reply_text(f"🧠 **MYRROR Test**\n\n{data['question']}", reply_markup=InlineKeyboardMarkup(keyboard), parse_mode="Markdown")
    except Exception as e:
        logger.error(f"Quiz error: {e}")
        await update.message.reply_text("I couldn't generate a quiz right now.")

async def quiz_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    user_id = str(query.from_user.id)
    idx = int(query.data.split("_")[1])
    
    options = user_quiz_options.get(user_id, [])
    chosen = options[idx] if idx < len(options) else "An unknown option"
    
    await query.edit_message_text(text=f"{query.message.text}\n\n✅ **You chose:** {chosen}", parse_mode="Markdown")
    
    content = f"[Quiz Answer] I chose: {chosen}"
    await process_message(update, user_id, content, False)

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
        profile = await asyncio.to_thread(get_profile, user_id)
        asyncio.create_task(run_post_analysis_tasks(user_id, "image", f"Image: {caption}", text, profile))
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
        profile = await asyncio.to_thread(get_profile, user_id)
        asyncio.create_task(run_post_analysis_tasks(user_id, f"file:{document.file_name}", f"File: {caption}", text, profile))
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
        profile = await asyncio.to_thread(get_profile, user_id)

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

    # Add message to the user's temporary burst buffer
    if user_id not in user_message_buffers:
        user_message_buffers[user_id] = []
    user_message_buffers[user_id].append(content)

    if user_id in user_pending:
        user_pending[user_id].cancel()

    async def debounced():
        try:
            await asyncio.sleep(COOLDOWN_SECONDS)
            
            await update.message.chat.send_action("typing")
            
            is_new_session = user_sessions.pop(user_id, False)
            combined_content = "\n".join(user_message_buffers[user_id])
            is_burst = len(user_message_buffers[user_id]) > 2
            user_message_buffers[user_id] = [] # Limpiar el búfer
            
            final_content = combined_content
            if is_burst:
                final_content = f"[RAPID BURST OF MESSAGES - VENTING DETECTED]\n{combined_content}"
            
            await process_message(update, user_id, final_content, is_new_session)
        except asyncio.CancelledError:
            pass
        finally:
            if user_pending.get(user_id) == asyncio.current_task():
                del user_pending[user_id]

    user_pending[user_id] = asyncio.create_task(debounced())

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# BOT STARTUP
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

async def start_telegram_bot():
    global telegram_app
    telegram_app = Application.builder().token(TELEGRAM_TOKEN).build()
    telegram_app.add_handler(CommandHandler("start", start))
    telegram_app.add_handler(CommandHandler("quiz", quiz_command))
    telegram_app.add_handler(CallbackQueryHandler(quiz_callback, pattern="^quiz_"))
    telegram_app.add_handler(CallbackQueryHandler(mood_callback, pattern="^mood_"))
    telegram_app.add_handler(CommandHandler("help", help_command))
    telegram_app.add_handler(CommandHandler("dossier", dossier_command))
    telegram_app.add_handler(CommandHandler("profile", profile_command))
    telegram_app.add_handler(CommandHandler("evolution", evolution_command))
    telegram_app.add_handler(CommandHandler("episodes", episodes_command))
    telegram_app.add_handler(CommandHandler("people", people_command))
    telegram_app.add_handler(CommandHandler("reflect", reflect_command))
    telegram_app.add_handler(CommandHandler("setcompass", setcompass_command))
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

async def stop_telegram_bot():
    global telegram_app
    if telegram_app:
        if telegram_app.updater:
            await telegram_app.updater.stop()
        await telegram_app.stop()
        await telegram_app.shutdown()
