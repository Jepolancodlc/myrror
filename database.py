import logging
from supabase import create_client
from dotenv import load_dotenv
import os

load_dotenv()

logger = logging.getLogger(__name__)

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

if not SUPABASE_URL or not SUPABASE_KEY:
    logger.error("CRITICAL: SUPABASE_URL or SUPABASE_KEY is missing in environment variables. Database will fail.")
    # Prevent crash on import, but queries will fail gracefully.
    supabase = None
else:
    supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

def get_profile(user_id: str) -> dict:
    try:
        if not supabase: return {}
        result = supabase.table("profile").select("*").eq("user_id", user_id).execute()
        if result.data:
            return result.data[0]["data"]
        return {}
    except Exception as e:
        logger.error(f"get_profile error for {user_id}: {e}", exc_info=True)
        return {}

def save_profile(user_id: str, data: dict):
    try:
        if not supabase: return
        existing = supabase.table("profile").select("*").eq("user_id", user_id).execute()
        if existing.data:
            supabase.table("profile").update({"data": data}).eq("user_id", user_id).execute()
        else:
            supabase.table("profile").insert({"user_id": user_id, "data": data}).execute()
    except Exception as e:
        logger.error(f"save_profile error for {user_id}: {e}", exc_info=True)

def get_all_profiles() -> list:
    """Fetch all profiles to check for inactive users or scheduled events."""
    try:
        if not supabase: return []
        result = supabase.table("profile").select("user_id", "data").execute()
        return result.data or []
    except Exception as e:
        logger.error(f"get_all_profiles error: {e}", exc_info=True)
        return []

def get_messages(user_id: str, limit: int = 10) -> list:
    try:
        if not supabase: return []
        # Ordenar descendente para obtener los últimos, y luego revertir para mantener el orden cronológico
        result = supabase.table("messages").select("*").eq("user_id", user_id).order("created_at", desc=True).limit(limit).execute()
        return list(reversed(result.data)) if result.data else []
    except Exception as e:
        logger.error(f"get_messages error for {user_id}: {e}", exc_info=True)
        return []

def get_all_messages(user_id: str) -> list:
    try:
        if not supabase: return []
        result = supabase.table("messages").select("*").eq("user_id", user_id).order("created_at").execute()
        return result.data or []
    except Exception as e:
        logger.error(f"get_all_messages error for {user_id}: {e}", exc_info=True)
        return []

def save_message(user_id: str, role: str, content: str):
    try:
        if not supabase: return
        supabase.table("messages").insert({
            "user_id": user_id,
            "role": role,
            "content": content
        }).execute()
    except Exception as e:
        logger.error(f"save_message error for {user_id}: {e}", exc_info=True)

def save_episode(user_id: str, event: str, domain: str = None, impact: str = None, embedding: list = None):
    try:
        if not supabase: return
        data = {
            "user_id": user_id,
            "event": event,
            "domain": domain,
            "impact": impact,
            "verified": False
        }
        if embedding:
            # Forzar el formato string para pgvector: "[0.1,0.2,...]"
            data["embedding"] = f"[{','.join(str(x) for x in embedding)}]"
        supabase.table("episodes").insert(data).execute()
    except Exception as e:
        logger.error(f"save_episode error for {user_id}: {e}", exc_info=True)

def get_episodes(user_id: str, limit: int = 10) -> list:
    try:
        if not supabase: return []
        result = supabase.table("episodes").select("*").eq("user_id", user_id).order("created_at", desc=True).limit(limit).execute()
        return result.data or []
    except Exception as e:
        logger.error(f"get_episodes error for {user_id}: {e}", exc_info=True)
        return []

def search_similar_episodes(user_id: str, query_embedding: list, limit: int = 3) -> list:
    try:
        if not supabase: return []
        response = supabase.rpc(
            "match_episodes",
            {
                # Forzar el formato string para la llamada RPC
                "query_embedding": f"[{','.join(str(x) for x in query_embedding)}]",
                "match_threshold": 0.6, # Solo recuerdos relevantes (60% de similitud o más)
                "match_count": limit,
                "p_user_id": user_id
            }
        ).execute()
        return response.data or []
    except Exception as e:
        logger.error(f"Vector search error for {user_id}: {e}", exc_info=True)
        return []

def get_person(user_id: str, name: str) -> dict:
    try:
        if not supabase: return {}
        result = supabase.table("people").select("*").eq("user_id", user_id).ilike("name", f"%{name}%").execute()
        if result.data:
            return result.data[0]
        return {}
    except Exception as e:
        logger.error(f"get_person error for {user_id}: {e}", exc_info=True)
        return {}

def get_all_people(user_id: str) -> list:
    try:
        if not supabase: return []
        result = supabase.table("people").select("*").eq("user_id", user_id).execute()
        return result.data or []
    except Exception as e:
        logger.error(f"get_all_people error for {user_id}: {e}", exc_info=True)
        return []

def get_null_episodes(user_id: str) -> list:
    try:
        if not supabase: return []
        result = supabase.table("episodes").select("*").eq("user_id", user_id).is_("embedding", "null").execute()
        return result.data or []
    except Exception as e:
        logger.error(f"get_null_episodes error for {user_id}: {e}", exc_info=True)
        return []

def update_episode_embedding(episode_id: str, embedding: list):
    try:
        if not supabase: return
        emb_str = f"[{','.join(str(x) for x in embedding)}]"
        supabase.table("episodes").update({"embedding": emb_str}).eq("id", episode_id).execute()
    except Exception as e:
        logger.error(f"update_episode_embedding error for {episode_id}: {e}", exc_info=True)

def save_person(user_id: str, name: str, relationship: str = None, notes: dict = {}):
    try:
        if not supabase: return
        existing = supabase.table("people").select("*").eq("user_id", user_id).ilike("name", name).execute()
        if existing.data:
            current_notes = existing.data[0].get("notes", {})
            current_notes.update(notes)
            supabase.table("people").update({
                "relationship": relationship,
                "notes": current_notes,
                "updated_at": "now()"
            }).eq("id", existing.data[0]["id"]).execute()
        else:
            supabase.table("people").insert({
                "user_id": user_id,
                "name": name,
                "relationship": relationship,
                "notes": notes
            }).execute()
    except Exception as e:
        logger.error(f"save_person error for {user_id}: {e}", exc_info=True)