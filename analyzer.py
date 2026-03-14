import logging
import os
import base64
from google import genai
from dotenv import load_dotenv
from prompt import SYSTEM_PROMPT
from database import get_profile, save_message
from extractor import get_profile_for_context

load_dotenv()

logger = logging.getLogger(__name__)
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

async def analyze_image(user_id: str, file_bytes: bytearray, caption: str) -> str:
    profile = get_profile(user_id)
    profile_ctx = get_profile_for_context(profile, caption)
    image_base64 = base64.b64encode(file_bytes).decode("utf-8")

    try:
        response = client.models.generate_content(
            model="gemini-3.1-flash-lite-preview",
            contents=[
                {
                    "parts": [
                        {"text": f"{SYSTEM_PROMPT}\n\nWHAT YOU KNOW ABOUT THIS USER:\n{profile_ctx}\n\nThe user sent an image with this message: '{caption}'\n\nAnalyze it as MYRROR would. Be honest and personal. Connect to what you know about them. End with one question."},
                        {"inline_data": {"mime_type": "image/jpeg", "data": image_base64}}
                    ]
                }
            ]
        )
        text = response.text
        save_message(user_id, "user", f"[Image] {caption}")
        save_message(user_id, "assistant", text)
        return text
    except Exception as e:
        logger.error(f"Image analysis error for {user_id}: {e}", exc_info=True)
        raise

async def analyze_document(user_id: str, file_bytes: bytearray, mime: str, filename: str, caption: str) -> str | None:
    profile = get_profile(user_id)
    profile_ctx = get_profile_for_context(profile, caption)

    try:
        if mime.startswith("text/"):
            file_content = file_bytes.decode("utf-8", errors="ignore")
            if len(file_content) > 500000:
                file_content = file_content[:500000] + "\n\n[File truncated]"

            analysis_prompt = f"""{SYSTEM_PROMPT}

WHAT YOU KNOW ABOUT THIS USER:
{profile_ctx}

The user sent a file. Detect the type:
- WhatsApp/Telegram chat → analyze communication dynamics, key turning points,
  what the user did well, what went wrong, patterns in how they communicate
- CV/Resume → analyze strengths, gaps, presentation, what to improve
- Contract/document → summarize key points and risks
- Journal/notes → analyze emotional state, thought patterns, growth areas
- Other → adapt intelligently

User's message: "{caption}"

FILE CONTENT:
{file_content}

INSTRUCTIONS:
1. Confirm what you read and date range if applicable.
2. Deep, honest, personal analysis as MYRROR would.
3. Connect findings to what you know about this person.
4. Reference specific moments from the file.
5. End with ONE specific question.
"""
            response = client.models.generate_content(
                model="gemini-3.1-flash-lite-preview",
                contents=analysis_prompt
            )

        elif mime == "application/pdf" or mime.startswith("image/"):
            file_base64 = base64.b64encode(file_bytes).decode("utf-8")
            response = client.models.generate_content(
                model="gemini-3.1-flash-lite-preview",
                contents=[
                    {
                        "parts": [
                            {"text": f"{SYSTEM_PROMPT}\n\nWHAT YOU KNOW ABOUT THIS USER:\n{profile_ctx}\n\nUser message: '{caption}'\n\nDetect the file type and analyze it as MYRROR would. Be specific and personal. End with one question."},
                            {"inline_data": {"mime_type": mime, "data": file_base64}}
                        ]
                    }
                ]
            )
        else:
            return None

        text = response.text
        save_message(user_id, "user", f"[File: {filename}] {caption}")
        save_message(user_id, "assistant", text)
        return text

    except Exception as e:
        logger.error(f"Document analysis error for {user_id}: {e}", exc_info=True)
        raise
