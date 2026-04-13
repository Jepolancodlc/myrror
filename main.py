import logging
import asyncio
import os
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from logging.handlers import RotatingFileHandler
from dotenv import load_dotenv
from app.services.chat import get_response
from app.bot.bot import start_telegram_bot, stop_telegram_bot
from app.core.keepalive import keep_alive
from app.db.database import get_profile
from contextlib import asynccontextmanager

load_dotenv()

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ADVANCED LOGGING SYSTEM (ROTATING FILES)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
log_formatter = logging.Formatter("%(asctime)s | %(name)s | %(levelname)s | %(message)s")

file_handler = RotatingFileHandler("myrror_system.log", maxBytes=10*1024*1024, backupCount=5, encoding="utf-8")
file_handler.setFormatter(log_formatter)

console_handler = logging.StreamHandler()
console_handler.setFormatter(log_formatter)

logging.basicConfig(level=logging.INFO, handlers=[file_handler, console_handler])
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    if os.getenv("TELEGRAM_TOKEN"):
        # Awaiting ensures FastAPI waits for the bot to be fully connected before accepting web requests
        await start_telegram_bot()
    keepalive_task = asyncio.create_task(keep_alive())
    yield
    if os.getenv("TELEGRAM_TOKEN"):
        await stop_telegram_bot()
    keepalive_task.cancel()

app = FastAPI(lifespan=lifespan)

# CORS Configuration: Enables secure cross-origin requests for future Dashboard or Web App integrations.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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

@app.get("/dossier/{user_id}")
async def web_dossier(user_id: str):
    try:
        profile = await asyncio.to_thread(get_profile, user_id)
        if not profile:
            return {"error": "Profile not found or insufficient data."}
            
        clean_profile = {k: v for k, v in profile.items() if k not in ["evolution", "confidence"] and v}
        return {"user_id": user_id, "dossier": clean_profile}
    except Exception as e:
        return {"error": str(e)}

@app.post("/chat")
async def chat(message: Message):
    try:
        text = await get_response(message.user_id, message.content, message.new_session)
        return {"response": text}
    except Exception as e:
        return {"response": "I'm experiencing a temporary cognitive lapse. Please try again."}
