import httpx
import asyncio
import logging
import os

logger = logging.getLogger(__name__)

# Render inyecta automáticamente RENDER_EXTERNAL_URL en producción
MYRROR_URL = os.getenv("RENDER_EXTERNAL_URL", os.getenv("MYRROR_URL", "http://localhost:8000"))

async def keep_alive():
    while True:
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                response = await client.get(f"{MYRROR_URL}/health")
                logger.debug(f"Self-ping successful: {response.status_code}")
        except Exception as e:
            logger.warning(f"Self-ping failed (normal during startup): {e}")
        
        # 840 segundos = 14 minutos. Engaña al timeout de 15 min de Render.
        await asyncio.sleep(840)