"""FastAPI Entry Point: Serves as the web server for Render and manages the Telegram Bot lifecycle."""
import logging
import os
import asyncio
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, Response
from app.bot.bot import start_telegram_bot, stop_telegram_bot
import app.bot.bot as bot_module
from app.core.keepalive import keep_alive
from telegram import Update

# Configuración general de Logs para producción
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)

# Añadir un manejador específico para guardar solo errores en un archivo local
file_handler = logging.FileHandler("myrror_errors.log", encoding="utf-8")
file_handler.setLevel(logging.ERROR)  # Solo capturará WARNING, ERROR y CRITICAL
file_handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
logging.getLogger().addHandler(file_handler)

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

@app.post("/webhook")
async def telegram_webhook(request: Request):
    """Endpoint to receive updates from Telegram via Webhook in production."""
    if bot_module.telegram_app:
        update = Update.de_json(await request.json(), bot_module.telegram_app.bot)
        await bot_module.telegram_app.process_update(update)
    return Response(status_code=200)

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)