from supabase import create_client
from dotenv import load_dotenv
import os

load_dotenv()

supabase = create_client(
    os.getenv("SUPABASE_URL"),
    os.getenv("SUPABASE_KEY")
)

def get_profile(user_id: str):
    result = supabase.table("profile").select("*").eq("user_id", user_id).execute()
    if result.data:
        return result.data[0]["data"]
    return {}

def save_profile(user_id: str, data: dict):
    existing = supabase.table("profile").select("*").eq("user_id", user_id).execute()
    if existing.data:
        supabase.table("profile").update({"data": data}).eq("user_id", user_id).execute()
    else:
        supabase.table("profile").insert({"user_id": user_id, "data": data}).execute()

def get_messages(user_id: str, limit: int = 20):
    result = supabase.table("messages").select("*").eq("user_id", user_id).order("created_at").limit(limit).execute()
    return result.data or []

def save_message(user_id: str, role: str, content: str):
    supabase.table("messages").insert({
        "user_id": user_id,
        "role": role,
        "content": content
    }).execute()