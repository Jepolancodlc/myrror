import logging
import os
import base64
import asyncio
from google import genai
from dotenv import load_dotenv
from prompt import SYSTEM_PROMPT
from database import get_profile, save_message
from extractor import get_profile_for_context

load_dotenv()

logger = logging.getLogger(__name__)
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

async def analyze_image(user_id: str, file_bytes: bytearray, caption: str) -> str:
    profile = await asyncio.to_thread(get_profile, user_id)
    profile_ctx = get_profile_for_context(profile, caption)
    image_base64 = base64.b64encode(file_bytes).decode("utf-8")

    try:
        response = await client.aio.models.generate_content(
            model="gemini-3.1-flash-lite-preview",
            contents=[
                {
                    "parts": [
                        {"text": (
                            f"{SYSTEM_PROMPT}\n\nWHAT YOU KNOW ABOUT THIS USER:\n{profile_ctx}\n\n"
                            f"The user sent an image with this message: '{caption}'\n\n"
                            "INSTRUCTIONS:\n"
                            "1. Analyze the image as MYRROR. Look beyond the obvious.\n"
                            "2. Notice the environment, mood, lighting, and hidden details.\n"
                            "3. Connect this visual to their psychological profile. Why did they share this with you right now?\n"
                            "4. Be brutally honest, highly observant, and deeply personal, but use simple, easy-to-understand language.\n"
                            "5. End with ONE penetrating question.\n"
                            "6. CRITICAL: You MUST reply entirely in the user's language (e.g., Spanish)."
                        )},
                        {"inline_data": {"mime_type": "image/jpeg", "data": image_base64}}
                    ]
                }
            ]
        )
        text = response.text
        await asyncio.to_thread(save_message, user_id, "user", f"[Image] {caption}")
        await asyncio.to_thread(save_message, user_id, "assistant", text)
        return text
    except Exception as e:
        logger.error(f"Image analysis error for {user_id}: {e}", exc_info=True)
        raise

async def analyze_document(user_id: str, file_bytes: bytearray, mime: str, filename: str, caption: str) -> str | None:
    profile = await asyncio.to_thread(get_profile, user_id)
    profile_ctx = get_profile_for_context(profile, caption)

    try:
        if mime.startswith("text/"):
            file_content = file_bytes.decode("utf-8", errors="ignore")
            if len(file_content) > 500000:
                file_content = file_content[:500000] + "\n\n[File truncated]"

            analysis_prompt = f"""{SYSTEM_PROMPT}

WHAT YOU KNOW ABOUT THIS USER:
{profile_ctx}

The user sent a file.

FILE CONTENT:
{file_content}
User's message: "{caption}"

INSTRUCTIONS:
1. Detect the file type (e.g., chat logs, resume, journal, code, etc.).
2. Read it as MYRROR. Read between the lines, but DO NOT output a clinical report, bullet points, or sections. 
3. Respond as a human conversationalist. If it's a chat, focus gently on how the user feels and what they might be missing. If it's a resume, focus on their potential and fears.
4. Be objective, direct, and personal, but highly empathetic. Keep it conversational and very easy to understand (avoid complex psychological jargon).
5. End with ONE powerful, Socratic question that forces them to reflect.
6. CRITICAL: You MUST reply entirely in the user's language (e.g., if the chat or user is in Spanish, reply in Spanish).
"""
            response = await client.aio.models.generate_content(
                model="gemini-3.1-flash-lite-preview",
                contents=analysis_prompt
            )

        elif mime == "application/pdf" or mime.startswith("image/"):
            file_base64 = base64.b64encode(file_bytes).decode("utf-8")
            response = await client.aio.models.generate_content(
                model="gemini-3.1-flash-lite-preview",
                contents=[
                    {
                        "parts": [
                            {"text": (
                                f"{SYSTEM_PROMPT}\n\nWHAT YOU KNOW ABOUT THIS USER:\n{profile_ctx}\n\n"
                                f"User message: '{caption}'\n\n"
                                "INSTRUCTIONS:\n"
                                "1. Detect the document type and read its contents carefully.\n"
                                "2. Do not just summarize. Read between the lines as MYRROR.\n"
                                "3. Connect the insights to their psychological profile.\n"
                                "4. End with ONE powerful, Socratic question, using simple and clear language.\n"
                                "5. CRITICAL: You MUST reply entirely in the user's language."
                            )},
                            {"inline_data": {"mime_type": mime, "data": file_base64}}
                        ]
                    }
                ]
            )
        else:
            return None

        text = response.text
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
            ]
        )
        return response.text.strip()
    except Exception as e:
        logger.error(f"Voice analysis error for {user_id}: {e}", exc_info=True)
        raise
