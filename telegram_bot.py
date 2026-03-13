from telegram import Update
from telegram.ext import Application, MessageHandler, filters, ContextTypes
from dotenv import load_dotenv
import httpx
import os

load_dotenv()

TOKEN = os.getenv("TELEGRAM_TOKEN")
MYRROR_URL = os.getenv("MYRROR_URL", "https://myrror.onrender.com")

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = str(update.effective_user.id)
    content = update.message.text

    async with httpx.AsyncClient(timeout=60) as client:
        response = await client.post(
            f"{MYRROR_URL}/chat",
            json={
                "user_id": user_id,
                "content": content,
                "new_session": False
            }
        )
        data = response.json()
        await update.message.reply_text(data["response"])

def main():
    app = Application.builder().token(TOKEN).build()
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    print("MYRROR Telegram bot running...")
    app.run_polling()

if __name__ == "__main__":
    main()