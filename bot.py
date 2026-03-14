import logging
import os
import time
import asyncio
from telegram import Update
from telegram.ext import Application, MessageHandler, CommandHandler, filters, ContextTypes
from dotenv import load_dotenv
from database import get_profile, supabase
from chat import get_response
from analyzer import analyze_image, analyze_document
from extractor import extract_and_save_profile, extract_episodes_from_content

load_dotenv()

logger = logging.getLogger(__name__)
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")

user_sessions = {}
user_last_message = {}
user_pending = {}

COOLDOWN_SECONDS = 3
MIN_MESSAGE_LENGTH = 2
TYPING_DELAY = 1.5

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
        text = get_response(user_id, content, is_new_session)
        await update.message.reply_text(text)
        asyncio.create_task(extract_and_save_profile(user_id, "message", content, text, profile))
        asyncio.create_task(extract_episodes_from_content(user_id, content, text))
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
        await asyncio.sleep(TYPING_DELAY)
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
        telegram_app.add_handler(CommandHandler("contract", contract_command))
        telegram_app.add_handler(CommandHandler("reset", reset_command))
        telegram_app.add_handler(MessageHandler(filters.PHOTO, handle_image))
        telegram_app.add_handler(MessageHandler(filters.Document.ALL, handle_document))
        telegram_app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
        await telegram_app.initialize()
        await telegram_app.start()
        await telegram_app.updater.start_polling()
        await asyncio.Event().wait()

    asyncio.run(start_bot())
