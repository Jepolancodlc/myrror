from telegram import Update
from telegram.ext import Application, MessageHandler, CommandHandler, filters, ContextTypes
from google import genai
from dotenv import load_dotenv
from prompt import SYSTEM_PROMPT
from database import get_profile, save_profile, get_messages, save_message
from extractor import extract_profile
from datetime import datetime
import os

load_dotenv()

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

# Track new sessions in memory
user_sessions = {}

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = str(update.effective_user.id)
    user_sessions[user_id] = True
    from database import supabase
    supabase.table("messages").delete().eq("user_id", user_id).execute()
    await update.message.reply_text("New session started. I'm here.")

async def profile_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = str(update.effective_user.id)
    profile = get_profile(user_id)
    if not profile:
        await update.message.reply_text("I don't know anything about you yet. Talk to me first.")
        return
    
    lines = ["Here's what I know about you:\n"]
    for key, value in profile.items():
        if key not in ["last_conversation", "total_conversations"]:
            lines.append(f"• {key}: {value}")
    
    if "last_conversation" in profile:
        lines.append(f"\nLast conversation: {profile['last_conversation']}")
    if "total_conversations" in profile:
        lines.append(f"Total conversations: {profile['total_conversations']}")
    
    await update.message.reply_text("\n".join(lines))

async def contract_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = str(update.effective_user.id)
    profile = get_profile(user_id)
    contracts = profile.get("personal_contracts", None)
    
    if not contracts:
        await update.message.reply_text(
            "You have no personal contracts yet.\n\n"
            "Tell me something like:\n"
            "'Don't let me justify skipping the gym'\n"
            "'Call me out if I haven't studied in 3 days'\n\n"
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

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = str(update.effective_user.id)
    content = update.message.text
    is_new_session = user_sessions.pop(user_id, False)

    # 1. Fetch profile and history
    profile = get_profile(user_id)
    history = get_messages(user_id)

    # 2. Build context
    ctx = SYSTEM_PROMPT
    ctx += f"\n\nCURRENT DATE AND TIME: {datetime.now().strftime('%Y-%m-%d %H:%M')}"

    if is_new_session:
        ctx += "\n\nNEW SESSION — start fresh. No tension from before."

    if profile:
        ctx += f"\n\nWHAT YOU KNOW ABOUT THIS USER:\n{profile}"

    if history:
        ctx += "\n\nRECENT HISTORY:\n"
        for msg in history:
            role = "User" if msg["role"] == "user" else "MYRROR"
            ctx += f"{role}: {msg['content']}\n"

    ctx += f"\nUser: {content}\nMYRROR:"

    # 3. Show typing indicator
    await update.message.chat.send_action("typing")

    # 4. Call Gemini with error handling
    try:
        response = client.models.generate_content(
            model="gemini-3.1-flash-lite-preview",
            contents=ctx
        )
        text = response.text
    except Exception as e:
        print(f"Gemini error: {e}")
        await update.message.reply_text(
            "I'm having trouble thinking right now. Give me a moment and try again."
        )
        return

    # 5. Extract and save profile
    try:
        updated_profile = extract_profile(profile, content, text)
        save_profile(user_id, updated_profile)
    except Exception as e:
        print(f"Profile extraction error: {e}")

    # 6. Save messages
    try:
        save_message(user_id, "user", content)
        save_message(user_id, "assistant", text)
    except Exception as e:
        print(f"Message save error: {e}")

    await update.message.reply_text(text)

def main():
    app = Application.builder().token(TELEGRAM_TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("profile", profile_command))
    app.add_handler(CommandHandler("contract", contract_command))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    print("MYRROR Telegram bot running...")
    app.run_polling()

if __name__ == "__main__":
    main()
