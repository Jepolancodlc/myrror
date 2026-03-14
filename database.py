import logging
from supabase import create_client
from dotenv import load_dotenv
import os

load_dotenv()

logger = logging.getLogger(__name__)

supabase = create_client(
    os.getenv("SUPABASE_URL"),
    os.getenv("SUPABASE_KEY")
)

def get_profile(user_id: str) -> dict:
    try:
        result = supabase.table("profile").select("*").eq("user_id", user_id).execute()
        if result.data:
            return result.data[0]["data"]
        return {}
    except Exception as e:
        logger.error(f"get_profile error for {user_id}: {e}", exc_info=True)
        return {}

def save_profile(user_id: str, data: dict):
    try:
        existing = supabase.table("profile").select("*").eq("user_id", user_id).execute()
        if existing.data:
            supabase.table("profile").update({"data": data}).eq("user_id", user_id).execute()
        else:
            supabase.table("profile").insert({"user_id": user_id, "data": data}).execute()
    except Exception as e:
        logger.error(f"save_profile error for {user_id}: {e}", exc_info=True)

def get_messages(user_id: str, limit: int = 20) -> list:
    try:
        result = supabase.table("messages").select("*").eq("user_id", user_id).order("created_at").limit(limit).execute()
        return result.data or []
    except Exception as e:
        logger.error(f"get_messages error for {user_id}: {e}", exc_info=True)
        return []

def save_message(user_id: str, role: str, content: str):
    try:
        supabase.table("messages").insert({
            "user_id": user_id,
            "role": role,
            "content": content
        }).execute()
    except Exception as e:
        logger.error(f"save_message error for {user_id}: {e}", exc_info=True)

def save_episode(user_id: str, event: str, domain: str = None, impact: str = None):
    try:
        supabase.table("episodes").insert({
            "user_id": user_id,
            "event": event,
            "domain": domain,
            "impact": impact,
            "verified": False
        }).execute()
    except Exception as e:
        logger.error(f"save_episode error for {user_id}: {e}", exc_info=True)

def get_episodes(user_id: str, limit: int = 10) -> list:
    try:
        result = supabase.table("episodes").select("*").eq("user_id", user_id).order("created_at", desc=True).limit(limit).execute()
        return result.data or []
    except Exception as e:
        logger.error(f"get_episodes error for {user_id}: {e}", exc_info=True)
        return []