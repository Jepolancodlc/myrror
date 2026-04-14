import logging
import asyncio
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

# Global asyncio.Lock registry to prevent Race Conditions during rapid sequential messages.
_user_locks = {}

def get_user_lock(user_id: str) -> asyncio.Lock:
    """Returns a user-specific asyncio.Lock for thread-safe updates, preventing Race Conditions."""
    if user_id not in _user_locks:
        _user_locks[user_id] = asyncio.Lock()
    return _user_locks[user_id]

def get_profile(user_id: str) -> dict:
    """Retrieves user profile JSON. Note: Supabase is synchronous; wrap in `asyncio.to_thread` to avoid blocking FastAPI."""
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
    """Creates or updates the complete profile of a user in the database."""
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
    """Fetches ALL registered user profiles. Used by Background Jobs to identify inactive users or events."""
    try:
        if not supabase: return []
        result = supabase.table("profile").select("user_id", "data").execute()
        return result.data or []
    except Exception as e:
        logger.error(f"get_all_profiles error: {e}", exc_info=True)
        return []

def get_messages(user_id: str, limit: int = 10) -> list:
    """Retrieves recent message history for short-term conversational context."""
    try:
        if not supabase: return []
        # Sort descending to grab recent items, then reverse to restore chronological order
        result = supabase.table("messages").select("*").eq("user_id", user_id).order("created_at", desc=True).limit(limit).execute()
        return list(reversed(result.data)) if result.data else []
    except Exception as e:
        logger.error(f"get_messages error for {user_id}: {e}", exc_info=True)
        return []

def get_all_messages(user_id: str) -> list:
    """Retrieves the complete chronological conversation history for a specific user."""
    try:
        if not supabase: return []
        result = supabase.table("messages").select("*").eq("user_id", user_id).order("created_at").execute()
        return result.data or []
    except Exception as e:
        logger.error(f"get_all_messages error for {user_id}: {e}", exc_info=True)
        return []

def save_message(user_id: str, role: str, content: str):
    """Saves an individual message into the database (Chat History)."""
    try:
        if not supabase: return
        supabase.table("messages").insert({
            "user_id": user_id,
            "role": role,
            "content": content
        }).execute()
    except Exception as e:
        logger.error(f"save_message error for {user_id}: {e}", exc_info=True)

def delete_user_messages(user_id: str):
    """Safely clears a user's short-term message history."""
    try:
        if not supabase: return
        supabase.table("messages").delete().eq("user_id", user_id).execute()
    except Exception as e:
        logger.error(f"delete_user_messages error for {user_id}: {e}", exc_info=True)

def save_episode(user_id: str, event: str, domain: str = None, impact: str = None, embedding: list = None):
    """Saves an autobiographical event ("Episode") and its mathematical vector into the Semantic RAG memory."""
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
            # Coerce list into string format required by PostgreSQL pgvector: "[0.1, 0.2, ...]"
            data["embedding"] = f"[{','.join(str(x) for x in embedding)}]"
        supabase.table("episodes").insert(data).execute()
    except Exception as e:
        logger.error(f"save_episode error for {user_id}: {e}", exc_info=True)

def get_episodes(user_id: str, limit: int = 10) -> list:
    """Retrieves the most recent biographical episodes chronologically (used for /episodes and Weekly Summary)."""
    try:
        if not supabase: return []
        result = supabase.table("episodes").select("*").eq("user_id", user_id).order("created_at", desc=True).limit(limit).execute()
        return result.data or []
    except Exception as e:
        logger.error(f"get_episodes error for {user_id}: {e}", exc_info=True)
        return []

def search_similar_episodes(user_id: str, query_embedding: list, limit: int = 3) -> list:
    """Semantic RAG Retrieval: Calls a Supabase RPC to find past episodes mathematically related via Cosine Similarity."""
    try:
        if not supabase: return []
        response = supabase.rpc(
            "match_episodes",
            {
                # Force string array format for pgvector RPC
                "query_embedding": f"[{','.join(str(x) for x in query_embedding)}]",
                "match_threshold": 0.6, # Return memories with >= 60% relevance
                "match_count": limit,
                "p_user_id": user_id
            }
        ).execute()
        return response.data or []
    except Exception as e:
        logger.error(f"Vector search error for {user_id}: {e}", exc_info=True)
        return []

def get_person(user_id: str, name: str) -> dict:
    """Searches for a specific person in the database by name (case-insensitive)."""
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
    """Fetches the list of all registered people in the user's life."""
    try:
        if not supabase: return []
        result = supabase.table("people").select("*").eq("user_id", user_id).execute()
        return result.data or []
    except Exception as e:
        logger.error(f"get_all_people error for {user_id}: {e}", exc_info=True)
        return []

def get_null_episodes(user_id: str) -> list:
    """Finds episodes missing their semantic vector (embedding=null) for auto-healing background jobs."""
    try:
        if not supabase: return []
        result = supabase.table("episodes").select("*").eq("user_id", user_id).is_("embedding", "null").execute()
        return result.data or []
    except Exception as e:
        logger.error(f"get_null_episodes error for {user_id}: {e}", exc_info=True)
        return []

def update_episode_embedding(episode_id: str, embedding: list):
    """Updates the semantic vector (embedding) of an existing episode (auto-maintenance)."""
    try:
        if not supabase: return
        emb_str = f"[{','.join(str(x) for x in embedding)}]"
        supabase.table("episodes").update({"embedding": emb_str}).eq("id", episode_id).execute()
    except Exception as e:
        logger.error(f"update_episode_embedding error for {episode_id}: {e}", exc_info=True)

def save_person(user_id: str, name: str, relationship: str = None, notes: dict = None):
    try:
        data = {
            "user_id": user_id,
            "name": name,
            "relationship": relationship,
            "notes": notes
        }
        supabase.table("people").upsert(
            data, 
            on_conflict="user_id,name"
        ).execute()
    except Exception as e:
        logger.error(f"Error saving person: {e}")

def delete_all_user_data(user_id: str):
    """Hard reset: Erases all profiles, messages, episodes, and people for a given user."""
    try:
        if not supabase: return
        supabase.table("profile").delete().eq("user_id", user_id).execute()
        supabase.table("messages").delete().eq("user_id", user_id).execute()
        supabase.table("episodes").delete().eq("user_id", user_id).execute()
        supabase.table("people").delete().eq("user_id", user_id).execute()
    except Exception as e:
        logger.error(f"delete_all_user_data error for {user_id}: {e}", exc_info=True)