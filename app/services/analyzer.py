"""Servicio de análisis multimodal (Imágenes, Documentos, Audio) usando Gemini."""
import logging
import os
import base64
import asyncio
from datetime import datetime
import re
from google import genai
from google.genai import types
from dotenv import load_dotenv
from app.core.prompt import SYSTEM_PROMPT
from app.db.database import get_profile, save_message, get_messages
from app.services.extractor import get_profile_for_context, get_rag_memories_text, SAFETY_SETTINGS

load_dotenv()

logger = logging.getLogger(__name__)
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

THOUGHT_PATTERN = re.compile(r'<thought>.*?</thought>', flags=re.DOTALL)

async def analyze_image(user_id: str, file_bytes: bytearray, caption: str) -> str:
    profile = await asyncio.to_thread(get_profile, user_id) or {}
    profile_ctx = get_profile_for_context(profile, caption)
    image_base64 = base64.b64encode(file_bytes).decode("utf-8")

    recent_history = await asyncio.to_thread(get_messages, user_id, 6)
    history_text = ""
    if recent_history:
        history_text = "\n\nRECENT CONVERSATION HISTORY:\n"
        for msg in recent_history:
            role = "User" if msg["role"] == "user" else "MYRROR"
            safe_content = msg['content'] if len(msg['content']) <= 500 else msg['content'][:500] + "... [truncated]"
            history_text += f"{role}: {safe_content}\n"
        history_text += "\nCRITICAL: The user just sent this image as part of the ongoing conversation above. Respond in a way that naturally continues the flow of the chat."

    if caption and (len(caption.split()) > 3 or len(caption) > 15):
        try:
            eps_text = await get_rag_memories_text(user_id, caption)
            if eps_text:
                history_text += f"\n\nRELEVANT PAST MEMORIES (Triggered by the image caption):\n{eps_text}\nSTEALTH MEMORY: If these memories connect to the image, bring them up naturally like a friend remembering a past detail. Never say 'I found a memory in my database'."
        except Exception as e:
            logger.error(f"RAG search error for {user_id} in image: {e}")

    mood = profile.get("current_mood_score", "unknown")
    try:
        response = await client.aio.models.generate_content(
            model="gemini-3.1-flash-lite-preview",
            contents=[
                {
                    "parts": [
                        {"text": (
                            f"{SYSTEM_PROMPT}\n\nWHAT YOU KNOW ABOUT THIS USER:\n{profile_ctx}{history_text}\n\n"
                            f"USER'S CURRENT MOOD SCORE: {mood}/10. Adapt your emotional tone to respect their current state.\n\n"
                            f"The user sent an image with this message: '{caption}'\n\n"
                            "INSTRUCTIONS:\n"
                            "1. Analyze the image as MYRROR. Look beyond the obvious.\n"
                            "2. Notice the environment, mood, lighting, and hidden details.\n"
                            "3. MUSIC/LYRICS DETECTION: If the image is a screenshot of a music app (Spotify, Apple Music) or contains lyrics, explicitly identify the song and artist. Use the meaning of the song/lyrics as a window into their current emotional state.\n"
                            "4. Connect this visual to their psychological profile. Why did they share this with you right now?\n"
                            "5. Be brutally honest, highly observant, and deeply personal, but use simple, easy-to-understand language.\n"
                            "6. NATURAL REACTION: If the image is casual/everyday, react casually. If it's profound, be profound. DO NOT force a deep Socratic question if it doesn't fit the context naturally.\n"
                            "7. CRITICAL: You MUST reply entirely in the user's language."
                        )},
                        {"inline_data": {"mime_type": "image/jpeg", "data": image_base64}}
                    ]
                }
            ],
            config=types.GenerateContentConfig(safety_settings=SAFETY_SETTINGS)
        )
        
        try:
            text = response.text
        except ValueError as e:
            logger.warning(f"Image analysis blocked by Gemini: {e}")
            text = None
            
        if not text:
            text = "I couldn't analyze the image clearly. Let's talk about something else."
            
        text = THOUGHT_PATTERN.sub('', text).strip()
        if not text: text = "..."
        
        await asyncio.to_thread(save_message, user_id, "user", f"[Image] {caption}")
        await asyncio.to_thread(save_message, user_id, "assistant", text)
        return text
    except Exception as e:
        logger.error(f"Image analysis error for {user_id}: {e}", exc_info=True)
        raise

async def analyze_document(user_id: str, file_bytes: bytearray, mime: str, filename: str, caption: str) -> str | None:
    profile = await asyncio.to_thread(get_profile, user_id) or {}
    profile_ctx = get_profile_for_context(profile, caption)

    recent_history = await asyncio.to_thread(get_messages, user_id, 6)
    history_text = ""
    if recent_history:
        history_text = "\n\nRECENT CONVERSATION HISTORY:\n"
        for msg in recent_history:
            role = "User" if msg["role"] == "user" else "MYRROR"
            safe_content = msg['content'] if len(msg['content']) <= 500 else msg['content'][:500] + "... [truncated]"
            history_text += f"{role}: {safe_content}\n"
        history_text += "\nCRITICAL: The user just sent this file as part of the ongoing conversation above. Respond in a way that naturally continues the flow of the chat."

    query_text = caption if caption and (len(caption.split()) > 3 or len(caption) > 15) else filename
    if query_text:
        try:
            eps_text = await get_rag_memories_text(user_id, query_text)
            if eps_text:
                history_text += f"\n\nRELEVANT PAST MEMORIES (Triggered by the document/caption):\n{eps_text}\nSTEALTH MEMORY: If these memories connect to the document, bring them up naturally like a friend remembering a past detail. Never say 'I found a memory in my database'."
        except Exception as e:
            logger.error(f"RAG search error for {user_id} in document: {e}")

    try:
        if mime.startswith("text/"):
            file_content = file_bytes.decode("utf-8", errors="ignore")
            if len(file_content) > 500000:
                file_content = file_content[:500000] + "\n\n[File truncated]"

            mood = profile.get("current_mood_score", "unknown")
            analysis_prompt = f"""{SYSTEM_PROMPT}

WHAT YOU KNOW ABOUT THIS USER:
{profile_ctx}{history_text}

The user sent a file.

USER'S CURRENT MOOD SCORE: {mood}/10. Adapt your emotional tone to respect their current state.
FILE CONTENT:
{file_content}
User's message: "{caption}"

INSTRUCTIONS:
1. Detect the file type (e.g., chat logs, resume, journal, code, etc.).
2. Read it as MYRROR. Read between the lines, but DO NOT output a clinical report, bullet points, or sections. 
3. Respond as a human conversationalist. If it's a chat, focus gently on how the user feels and what they might be missing. If it's a resume, focus on their potential and fears.
4. Be objective, direct, and personal, but highly empathetic. Keep it conversational and very easy to understand (avoid complex psychological jargon).
5. NATURAL REACTION: Adapt to the document. Do not force a deep question at the end if the file is just practical/administrative.
6. CRITICAL: You MUST reply entirely in the user's language.
"""
            response = await client.aio.models.generate_content(
                model="gemini-3.1-flash-lite-preview",
                contents=analysis_prompt,
                config=types.GenerateContentConfig(safety_settings=SAFETY_SETTINGS)
            )

        elif mime == "application/pdf" or mime.startswith("image/"):
            mood = profile.get("current_mood_score", "unknown")
            file_base64 = base64.b64encode(file_bytes).decode("utf-8")
            response = await client.aio.models.generate_content(
                model="gemini-3.1-flash-lite-preview",
                contents=[
                    {
                        "parts": [
                            {"text": (
                                f"{SYSTEM_PROMPT}\n\nWHAT YOU KNOW ABOUT THIS USER:\n{profile_ctx}{history_text}\n\n"
                                f"USER'S CURRENT MOOD SCORE: {mood}/10. Adapt your emotional tone to respect their current state.\n\n"
                                f"User message: '{caption}'\n\n"
                                "INSTRUCTIONS:\n"
                                "1. Detect the document type and read its contents carefully.\n"
                                "2. Do not just summarize. Read between the lines as MYRROR.\n"
                                "3. Connect the insights to their psychological profile.\n"
                                "4. NATURAL REACTION: Talk about it conversationally. Don't sound like an AI assistant summarizing a PDF.\n"
                                "5. CRITICAL: You MUST reply entirely in the user's language."
                            )},
                            {"inline_data": {"mime_type": mime, "data": file_base64}}
                        ]
                    }
            ],
            config=types.GenerateContentConfig(safety_settings=SAFETY_SETTINGS)
            )
        else:
            return None

        
        try:
            text = response.text
        except ValueError as e:
            logger.warning(f"Document analysis blocked by Gemini: {e}")
            text = None
            
        if not text:
            text = "I couldn't read the document properly. Some content might have been blocked."
            
        text = THOUGHT_PATTERN.sub('', text).strip()
        if not text: text = "..."
        
        await asyncio.to_thread(save_message, user_id, "user", f"[File: {filename}] {caption}")
        await asyncio.to_thread(save_message, user_id, "assistant", text)
        return text

    except Exception as e:
        logger.error(f"Document analysis error for {user_id}: {e}", exc_info=True)
        raise

async def analyze_voice(user_id: str, file_bytes: bytearray, mime: str) -> str:
    audio_base64 = base64.b64encode(file_bytes).decode("utf-8")
    try:
        response = await client.aio.models.generate_content(
            model="gemini-3.1-flash-lite-preview",
            contents=[
                {
                    "parts": [
                        {"text": (
                            "Listen carefully to this audio. You are natively multimodal, so pay deep attention to the emotional tone, "
                            "voice cracks, sighing, laughing, breathing, speech pace, and stress levels.\n\n"
                            "Return the response in this EXACT format:\n"
                            "[Voice Analysis: Write a sharp, clinical but empathetic observation of HOW they are speaking (e.g., 'Breathing heavily, speaking fast, sounds anxious but trying to hide it')]\n\n"
                            "<Write the exact transcription of what they said, retaining stutters or pauses, in its original language>"
                        )},
                        {"inline_data": {"mime_type": mime, "data": audio_base64}}
                    ]
                }
            ],
            config=types.GenerateContentConfig(safety_settings=SAFETY_SETTINGS)
        )
        
        try:
            return response.text.strip()
        except ValueError as e:
            logger.warning(f"Voice analysis blocked by Gemini: {e}")
            return "[Voice Analysis: Unable to process]\n\n(I couldn't transcribe this audio.)"
    except Exception as e:
        logger.error(f"Voice analysis error for {user_id}: {e}", exc_info=True)
        raise
