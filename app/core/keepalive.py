import httpx
import asyncio
import logging
import os

logger = logging.getLogger(__name__)
MYRROR_URL = os.getenv("MYRROR_URL", "https://myrror.onrender.com")

async def keep_alive():
    while True:
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                response = await client.get(f"{MYRROR_URL}/health")
                logger.info(f"Keep-alive ping: {response.status_code}")
        except Exception as e:
            logger.error(f"Keep-alive error: {e}")
        await asyncio.sleep(600)