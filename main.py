import logging
import threading
import asyncio
import os
from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv
from chat import get_response
from bot import run_telegram_bot
from keepalive import keep_alive

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)

app = FastAPI()

class Message(BaseModel):
    user_id: str
    content: str
    new_session: bool = False

@app.get("/")
def home():
    return {"message": "MYRROR online"}

@app.get("/health")
def health():
    return {"status": "alive"}

@app.post("/chat")
async def chat(message: Message):
    text = await get_response(message.user_id, message.content, message.new_session)
    return {"response": text}

@app.on_event("startup")
async def startup_event():
    if os.getenv("TELEGRAM_TOKEN"):
        thread = threading.Thread(target=run_telegram_bot, daemon=True)
        thread.start()
        logger.info("Telegram bot started.")
    asyncio.create_task(keep_alive())
