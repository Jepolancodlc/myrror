"""FastAPI Entry Point: Serves as the web server for Render and manages the Telegram Bot lifecycle."""
import logging
import os
import asyncio
from contextlib import asynccontextmanager
from fastapi import FastAPI
from app.bot.bot import start_telegram_bot, stop_telegram_bot
from app.core.keepalive import keep_alive

# Configuración general de Logs para producción
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Controla el ciclo de vida de la app: Inicia el bot al arrancar y lo limpia al apagar."""
    logger.info("Starting MYRROR ecosystem...")
    await start_telegram_bot()
    
    # Inicia el latido del corazón para evitar que Render hiberne el bot
    ping_task = asyncio.create_task(keep_alive())
    yield
    logger.info("Shutting down MYRROR ecosystem...")
    ping_task.cancel()
    await stop_telegram_bot()

app = FastAPI(title="MYRROR API", lifespan=lifespan)

@app.get("/")
async def root():
    return {"status": "online", "message": "MYRROR Telegram Bot is running in the background."}

@app.get("/health")
async def health_check():
    return {"status": "ok"}

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)