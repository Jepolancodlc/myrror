from fastapi import FastAPI
from pydantic import BaseModel
from google import genai
from dotenv import load_dotenv
from prompt import SYSTEM_PROMPT
from database import get_profile, save_profile, get_messages, save_message
from extractor import extract_profile
from datetime import datetime
import os

load_dotenv()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

app = FastAPI()

class Message(BaseModel):
    user_id: str
    content: str
    new_session: bool = False

@app.get("/")
def home():
    return {"message": "MYRROR online"}

@app.post("/chat")
def chat(message: Message):

    # 1. Fetch profile and history
    profile = get_profile(message.user_id)
    history = get_messages(message.user_id)

    # 2. Build context for Gemini
    context = SYSTEM_PROMPT
    context += f"\n\nCURRENT DATE AND TIME: {datetime.now().strftime('%Y-%m-%d %H:%M')}"

    if message.new_session:
        context += "\n\nNEW SESSION STARTED — the user is starting fresh today."

    if profile:
        context += f"\n\nWHAT YOU KNOW ABOUT THIS USER:\n{profile}"

    if history:
        context += "\n\nRECENT HISTORY:\n"
        for msg in history:
            role = "User" if msg["role"] == "user" else "MYRROR"
            context += f"{role}: {msg['content']}\n"

    context += f"\nUser: {message.content}\nMYRROR:"

    # 3. Call Gemini
    response = client.models.generate_content(
        model="gemini-3.1-flash-lite-preview",
        contents=context
    )
    text = response.text

    # 4. Extract and save updated profile
    updated_profile = extract_profile(profile, message.content, text)
    save_profile(message.user_id, updated_profile)

    # 5. Save messages
    save_message(message.user_id, "user", message.content)
    save_message(message.user_id, "assistant", text)

    return {"response": text}