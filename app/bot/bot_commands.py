"""Telegram Command Handlers: Processes all explicit bot commands (/start, /help, /profile, etc.)."""
import logging
import os
import asyncio
import json
import random
import math
import hashlib
import redis.asyncio as redis
from tenacity import retry, stop_after_attempt, wait_exponential
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import ContextTypes
from app.db.database import get_profile, get_episodes, get_messages, get_all_people, supabase, save_profile, get_user_lock, delete_all_user_data
from app.services.extractor import track_evolution, generate_weekly_summary
from google import genai
from datetime import datetime

logger = logging.getLogger(__name__)
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
REDIS_URL = os.getenv("REDIS_URL") or "redis://localhost:6379"
redis_client = redis.from_url(REDIS_URL)

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=8), reraise=True)
async def safe_generate_content(*args, **kwargs):
    """Previene que los comandos fallen si la API de Gemini tiene caídas temporales de red."""
    return await client.aio.models.generate_content(*args, **kwargs)

def get_mood_keyboard():
    return InlineKeyboardMarkup([
        [
            InlineKeyboardButton("1 ⬛", callback_data="mood_1"),
            InlineKeyboardButton("2", callback_data="mood_2"),
            InlineKeyboardButton("3", callback_data="mood_3"),
            InlineKeyboardButton("4", callback_data="mood_4"),
            InlineKeyboardButton("5 🟨", callback_data="mood_5"),
        ],
        [
            InlineKeyboardButton("6", callback_data="mood_6"),
            InlineKeyboardButton("7", callback_data="mood_7"),
            InlineKeyboardButton("8", callback_data="mood_8"),
            InlineKeyboardButton("9", callback_data="mood_9"),
            InlineKeyboardButton("10 🟩", callback_data="mood_10"),
        ]
    ])

async def localize(user_id: str, text: str, profile: dict = None) -> str:
    if profile is None:
        profile = await asyncio.to_thread(get_profile, user_id)
    if not profile:
        return text

    language = profile.get("language")
    if not language:
        return text
        
    lang_key = language.lower().strip()
    if lang_key in ["english", "en", "en-us", "en-gb"]:
        return text

    # Caché distribuido y persistente con Redis usando hash MD5
    text_hash = hashlib.md5(text.encode('utf-8')).hexdigest()
    redis_key = f"lang:{lang_key}:{text_hash}"
    cached = await redis_client.get(redis_key)
    if cached:
        return cached.decode('utf-8')

    prompt = f"""Translate the following text to {language}.
CRITICAL: Maintain the exact same formatting, Markdown, and emojis. Do NOT add any conversational text. Return ONLY the translated text.

TEXT TO TRANSLATE:
{text}"""
    try:
        response = await safe_generate_content(model="gemini-3.1-flash-lite-preview", contents=prompt)
        translated = response.text.strip()
        await redis_client.set(redis_key, translated, ex=86400 * 30) # Retener por 30 días
        return translated
    except Exception as e:
        logger.error(f"Localization error: {e}")
        return text

async def mood_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = str(update.effective_user.id)
    profile = await asyncio.to_thread(get_profile, user_id) or {}
    evolution = profile.get("evolution", [])

    mood_data = [e for e in evolution if e.get("field") == "current_mood_score"]
    if len(mood_data) < 2:
        msg = await localize(user_id, "I need more conversations with you to track your mood evolution. Keep talking to me!", profile)
        await update.message.reply_text(msg)
        return

    def _generate_mood_graph(mood_data):
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import io
        
        # FIX: Ensure dates and scores align perfectly by filtering together
        valid_moods = [e for e in mood_data[-14:] if str(e.get("to", "")).replace(',', '.').replace('.', '', 1).isdigit()]
        dates = [e["date"][5:] for e in valid_moods]
        scores = [float(str(e["to"]).replace(',', '.')) for e in valid_moods]

        fig = plt.figure(figsize=(8, 4))
        ax = fig.add_subplot(111)
        ax.plot(dates, scores, marker='o', color='#4A90E2', linestyle='-', linewidth=2)
        ax.set_ylim(0, 10)
        ax.set_title("Your Emotional Evolution")
        ax.set_ylabel("Mood Score (1-10)")
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        fig.tight_layout()

        buf = io.BytesIO()
        fig.savefig(buf, format='png')
        buf.seek(0)
        plt.close(fig)
        return buf

    try:
        # Ejecutar la tarea de CPU intensiva fuera del hilo principal asíncrono
        buf = await asyncio.to_thread(_generate_mood_graph, mood_data)

        msg_cap = await localize(user_id, "Here is how your mood has been trending.", profile)
        await update.message.reply_photo(photo=buf, caption=msg_cap)
    except ImportError:
        await update.message.reply_text("Visual tracking requires matplotlib. (Run: pip install matplotlib)")
    except Exception as e:
        logger.error(f"Mood graph error: {e}")
        await update.message.reply_text(await localize(user_id, "I couldn't generate the mood graph right now.", profile))

async def sos_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = str(update.effective_user.id)
    text = (
        "🚨 **CRISIS RESOURCES** 🚨\n\n"
        "I am an AI. I care about you, but I cannot replace real medical or psychological help.\n"
        "If you feel you are in danger, please reach out to humans who can help you right now:\n\n"
        "• **Emergency:** Call 112 (Europe) or 911 (Americas).\n"
        "• **Crisis Text Line:** Text HOME to 741741\n"
        "• **Global Helplines:** https://findahelpline.com/\n\n"
        "You are not alone. Please talk to a professional."
    )
    msg = await localize(user_id, text)
    await update.message.reply_text(msg, parse_mode="Markdown")

async def export_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = str(update.effective_user.id)
    profile = await asyncio.to_thread(get_profile, user_id)
    episodes = await asyncio.to_thread(get_episodes, user_id, limit=100)
    
    data = {"profile": profile, "episodes": episodes}
    file_bytes = json.dumps(data, indent=2, ensure_ascii=False).encode('utf-8')
    
    await update.message.reply_document(
        document=file_bytes,
        filename=f"myrror_backup.json",
        caption=await localize(user_id, "Here is a complete backup of your psychological profile and episodes.", profile)
    )

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = str(update.effective_user.id)
    text = (
        "Here is what you can ask me to do:\n\n"
        "🧠 **Self-Discovery**\n"
        "• /profile - See what I know about your personality & goals\n"
        "• /dossier - View your secret psychological and behavioral dossier\n"
        "• /stats - View your psychological profile as RPG character stats\n"
        "• /quiz - Take a quick psychological test to reveal your blind spots\n"
        "• /evolution - Track how you've changed over time\n"
        "• /episodes - View the significant moments of your life\n"
        "• /people - See who I remember from your stories\n"
        "• /mood - View a visual graph of your emotional evolution\n\n"
        "🪞 **Reflection & Action**\n"
        "• /reflect - Ask for a deep, honest reflection on where you are\n"
        "• /flashback - Revisit a random past memory we've discussed\n"
        "• /setcompass <text> - Manually set your life's core mission\n"
        "• /week - Get a summary of your week's patterns and commitments\n"
        "• /contract - See the personal rules you've asked me to hold you to\n\n"
        "⚙️ **System & Support**\n"
        "• /sos - Get emergency resources if you're in crisis\n"
        "• /export - Download a full backup of your data\n"
        "• /reset - Erase all your history and start completely fresh"
    )
    msg = await localize(user_id, text)
    await update.message.reply_text(msg, parse_mode="Markdown")

async def profile_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = str(update.effective_user.id)
    profile = await asyncio.to_thread(get_profile, user_id)
    if not profile:
        msg = await localize(user_id, "I don't know anything about you yet. Talk to me first.", profile)
        await update.message.reply_text(msg)
        return
    skip = ["evolution", "confidence", "last_conversation", "total_conversations"]
    lines = ["Here's what I know about you:\n"]
    for key, value in profile.items():
        if key not in skip and value:
            lines.append(f"• {key}: {value}")
    if "last_conversation" in profile:
        lines.append(f"\nLast conversation: {profile['last_conversation']}")
    if "total_conversations" in profile:
        lines.append(f"Total conversations: {profile['total_conversations']}")
        
    msg = await localize(user_id, "\n".join(lines), profile)
    await update.message.reply_text(msg)
    
    big_five = profile.get("clinical_profile", {}).get("big_five")
    if isinstance(big_five, dict) and len(big_five) >= 5:
        def _generate_radar_chart(bf):
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            import io
            
            labels = ['Openness', 'Conscientiousness', 'Extraversion', 'Agreeableness', 'Neuroticism']
            values = [
                float(bf.get("O", 5)),
                float(bf.get("C", 5)),
                float(bf.get("E", 5)),
                float(bf.get("A", 5)),
                float(bf.get("N", 5))
            ]
            
            num_vars = len(labels)
            angles = [n / float(num_vars) * 2 * math.pi for n in range(num_vars)]
            values += values[:1]
            angles += angles[:1]
            
            fig, ax = plt.subplots(figsize=(5, 5), subplot_kw=dict(polar=True))
            ax.fill(angles, values, color='#9b59b6', alpha=0.25)
            ax.plot(angles, values, color='#8e44ad', linewidth=2)
            
            ax.set_yticklabels([])
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(labels, fontsize=10)
            ax.set_ylim(0, 10)
            ax.set_title("Your Psychological Signature", size=14, color='#333', y=1.1)
            
            buf = io.BytesIO()
            fig.savefig(buf, format='png', bbox_inches='tight')
            buf.seek(0)
            plt.close(fig)
            return buf

        try:
            buf = await asyncio.to_thread(_generate_radar_chart, big_five)
            
            msg_cap = await localize(user_id, "This is the shape of your personality based on our interactions.", profile)
            await update.message.reply_photo(photo=buf, caption=msg_cap)
        except Exception as e:
            logger.error(f"Radar chart error: {e}")

async def stats_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = str(update.effective_user.id)
    profile = await asyncio.to_thread(get_profile, user_id)
    
    if not profile:
        msg = await localize(user_id, "I don't have enough data to generate your stats yet. Keep talking to me.", profile)
        await update.message.reply_text(msg)
        return

    msg_status = await localize(user_id, "📊 *Scouting your attributes... compiling character sheet...*", profile)
    status_msg = await update.message.reply_text(msg_status, parse_mode="Markdown")

    prompt = f"""You are a master scout and RPG game master. Based on the following psychological profile, evaluate this user as if they were a video game character or an athlete.
Assign them numerical scores from 1 to 100 for the following attributes based STRICTLY on their behavioral patterns, cognition, flaws, and strengths.

PROFILE: {json.dumps(profile, ensure_ascii=False)}

Respond ONLY with a JSON object in this exact format:
{{
    "core": {{"Intellect": 0, "Empathy": 0, "Resilience": 0, "Discipline": 0, "Charisma": 0, "Creativity": 0}},
    "survival": {{"Stress Tolerance": 0, "Adaptability": 0, "Willpower": 0, "Ego Defense": 0}},
    "class": "A two-word RPG Class (e.g. Chaotic Mage, Stoic Paladin)"
}}"""

    try:
        from google.genai import types
        response = await safe_generate_content(
            model="gemini-3.1-flash-lite-preview",
            contents=prompt,
            config=types.GenerateContentConfig(response_mime_type="application/json")
        )
        data = json.loads(response.text)
        
        def _generate_stats_graphs(stats_data):
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            import io
            import math
            
            fig = plt.figure(figsize=(10, 5), facecolor='#121212')
            
            # --- Gráfico de Radar (Core Stats) ---
            ax1 = fig.add_subplot(121, polar=True, facecolor='#121212')
            core = stats_data.get("core", {})
            labels = list(core.keys())
            values = list(core.values())
            
            num_vars = len(labels)
            angles = [n / float(num_vars) * 2 * math.pi for n in range(num_vars)]
            values += values[:1]
            angles += angles[:1]
            
            ax1.fill(angles, values, color='#00ffcc', alpha=0.3)
            ax1.plot(angles, values, color='#00ffcc', linewidth=2)
            ax1.set_ylim(0, 100)
            ax1.set_xticks(angles[:-1])
            ax1.set_xticklabels(labels, color='white', size=10)
            ax1.set_yticklabels([])
            ax1.spines['polar'].set_color('#333333')
            ax1.set_title("Core Attributes", color='white', pad=20, size=14, weight='bold')
            
            # --- Gráfico de Barras (Survival Stats) ---
            ax2 = fig.add_subplot(122, facecolor='#121212')
            surv = stats_data.get("survival", {})
            y_pos = range(len(surv))
            bars = ax2.barh(y_pos, list(surv.values()), color='#ff007f', height=0.5)
            
            ax2.set_yticks(y_pos)
            ax2.set_yticklabels(list(surv.keys()), color='white', size=11)
            ax2.set_xlim(0, 100)
            ax2.invert_yaxis()
            for spine in ['top', 'right', 'bottom', 'left']: ax2.spines[spine].set_visible(False)
            ax2.tick_params(axis='x', colors='#121212') # Ocultar números de x
            ax2.set_title("Survival Stats", color='white', pad=20, size=14, weight='bold')
            
            for bar in bars:
                ax2.text(bar.get_width() + 3, bar.get_y() + bar.get_height()/2, f'{int(bar.get_width())}', ha='left', va='center', color='white', weight='bold')
                         
            fig.suptitle(f"Class: {stats_data.get('class', 'Unknown')}", color='#f1c40f', size=18, weight='bold', y=1.05)
            fig.tight_layout()
            
            buf = io.BytesIO()
            fig.savefig(buf, format='png', bbox_inches='tight', dpi=150)
            buf.seek(0)
            plt.close(fig)
            return buf

        buf = await asyncio.to_thread(_generate_stats_graphs, data)
        caption = await localize(user_id, f"🎮 **Your Character Sheet**\nClass: {data.get('class', 'Unknown')}\n\nThese stats are dynamically generated based on your real psychological profile.", profile)
        
        await update.message.reply_photo(photo=buf, caption=caption, parse_mode="Markdown")
        await status_msg.delete()
        
    except Exception as e:
        logger.error(f"Stats command error: {e}")
        msg_err = await localize(user_id, "I couldn't generate your stats right now. Try again later.", profile)
        await status_msg.edit_text(msg_err)

async def dossier_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = str(update.effective_user.id)
    profile = await asyncio.to_thread(get_profile, user_id)
    if not profile:
        msg = await localize(user_id, "I need to observe you more before compiling your psychological dossier.", profile)
        await update.message.reply_text(msg)
        return
        
    lines = ["🗄️ *CONFIDENTIAL PSYCHOLOGICAL DOSSIER*\n"]
    
    clin = profile.get("clinical_profile", {})
    if clin:
        lines.append("🧠 *CLINICAL PROFILE*")
        if clin.get("mbti"): lines.append(f"• MBTI: {clin.get('mbti')}")
        if clin.get("enneagram"): lines.append(f"• Enneagram: {clin.get('enneagram')}")
        if clin.get("archetype"): lines.append(f"• Archetype: {clin.get('archetype')}")
        lines.append("")
        
    if profile.get("cognition_style"):
        lines.append("⚙️ *COGNITIVE WIRING*")
        lines.append(f"{profile.get('cognition_style')}\n")
        
    if profile.get("psyche_and_motivations"):
        lines.append("🎭 *PSYCHE & MOTIVATIONS*")
        lines.append(f"{profile.get('psyche_and_motivations')}\n")
        
    if profile.get("behavioral_patterns"):
        lines.append("🔄 *BEHAVIORAL PATTERNS*")
        for p in profile.get("behavioral_patterns"):
            lines.append(f"• {p}")
        lines.append("")
        
    if profile.get("quirks_and_micro_details"):
        lines.append("🔎 *OBSERVED QUIRKS & MICRO-DETAILS*")
        for q in profile.get("quirks_and_micro_details"):
            lines.append(f"• {q}")
    
    if len(lines) == 1:
        msg = await localize(user_id, "Your dossier is currently empty. Keep talking to me so I can analyze your patterns.", profile)
        await update.message.reply_text(msg)
        return
        
    msg = await localize(user_id, "\n".join(lines), profile)
    try:
        await update.message.reply_text(msg, parse_mode="Markdown")
    except Exception as e:
        logger.warning(f"Markdown parse failed in dossier, sending clean: {e}")
        await update.message.reply_text(msg)

async def setcompass_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = str(update.effective_user.id)
    compass_text = " ".join(context.args)
    
    if not compass_text:
        msg = await localize(user_id, "Please provide your life compass. Example:\n`/setcompass To build a legacy of kindness and create art that moves people.`")
        await update.message.reply_text(msg, parse_mode="Markdown")
        return
        
    async with get_user_lock(user_id):
        profile = await asyncio.to_thread(get_profile, user_id) or {}
        profile["life_compass"] = compass_text
        await asyncio.to_thread(save_profile, user_id, profile)
    msg = await localize(user_id, "🧭 **Life Compass Updated**\n\nI have anchored this to your core profile. I will use it to guide you back when you feel lost.", profile)
    try:
        await update.message.reply_text(msg, parse_mode="Markdown")
    except Exception:
        await update.message.reply_text(msg)

async def evolution_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = str(update.effective_user.id)
    profile = await asyncio.to_thread(get_profile, user_id) or {}
    evolution = profile.get("evolution", [])
    if not evolution:
        msg = await localize(user_id, "No changes tracked yet. Keep talking to me.", profile)
        await update.message.reply_text(msg)
        return
    lines = ["Your evolution over time:\n"]
    confidence_map = profile.get("confidence", {})
    for e in evolution[-10:]:
        conf = e.get("confidence", confidence_map.get(e["field"], {}).get("level", "medium"))
        conf_str = "🟢 High" if conf == "high" else "🟡 Medium"
        lines.append(f"• {e['date']} — {e['field']} [{conf_str}]: {e['note']}")
    msg = await localize(user_id, "\n".join(lines), profile)
    await update.message.reply_text(msg)

async def episodes_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = str(update.effective_user.id)
    episodes = await asyncio.to_thread(get_episodes, user_id, limit=15)
    if not episodes:
        msg = await localize(user_id, "No significant episodes recorded yet.")
        await update.message.reply_text(msg)
        return
    lines = ["Your story so far:\n"]
    for ep in reversed(episodes):
        date = ep.get("created_at", "")[:10]
        event = ep.get("event", "")
        domain = ep.get("domain", "")
        impact = ep.get("impact", "")
        if not event.startswith("Daily summary") and not event.startswith("Weekly summary"):
            lines.append(f"• {date} [{domain}] {event} ({impact})")
    if len(lines) == 1:
        msg = await localize(user_id, "No significant episodes recorded yet.")
        await update.message.reply_text(msg)
        return
    msg = await localize(user_id, "\n".join(lines))
    await update.message.reply_text(msg)

async def people_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = str(update.effective_user.id)
    people = await asyncio.to_thread(get_all_people, user_id)
    if not people:
        msg = await localize(user_id, "I don't know anyone in your life yet. Tell me about the people around you.")
        await update.message.reply_text(msg)
        return
    lines = ["People in your life:\n"]
    for p in people:
        name = p.get("name", "")
        rel = p.get("relationship", "")
        notes = p.get("notes", {})
        desc = notes.get("description", "") if isinstance(notes, dict) else ""
        lines.append(f"• {name} ({rel}){' — ' + desc if desc else ''}")
    msg = await localize(user_id, "\n".join(lines))
    await update.message.reply_text(msg)

async def reflect_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = str(update.effective_user.id)
    profile = await asyncio.to_thread(get_profile, user_id)
    episodes = await asyncio.to_thread(get_episodes, user_id, limit=20)
    if not profile:
        msg = await localize(user_id, "I don't know you well enough yet. Talk to me more first.", profile)
        await update.message.reply_text(msg)
        return
    msg = await localize(user_id, "🪞 *Reflecting on your journey...*", profile)
    status_msg = await update.message.reply_text(msg, parse_mode="Markdown")
    
    valid_episodes = [ep for ep in reversed(episodes) if not ep.get("event", "").startswith("Daily summary") and not ep.get("event", "").startswith("Weekly summary")]
    if valid_episodes:
        eps_list = []
        now_date = datetime.now().date()
        for ep in valid_episodes:
            ep_date_str = ep.get('created_at', '')[:10]
            try:
                ep_date = datetime.strptime(ep_date_str, "%Y-%m-%d").date()
                days_ago = (now_date - ep_date).days
                time_ctx = "today" if days_ago == 0 else ("yesterday" if days_ago == 1 else f"{days_ago} days ago")
            except:
                time_ctx = ep_date_str
            eps_list.append(f"- [{ep.get('domain')}] {ep.get('event')} ({time_ctx})")
        episodes_text = "\n".join(eps_list)
    else:
        episodes_text = "No significant episodes recorded yet."

    language = profile.get("language", "the user's primary language")
    prompt = f"""You are MYRROR. Generate a profound, psychological reflection for this person.

PROFILE DATA:
{json.dumps(profile, ensure_ascii=False, separators=(',', ':'))}

SIGNIFICANT EPISODES:
{episodes_text}

INSTRUCTIONS:
1. Do not just summarize their life. Pierce through the surface.
2. Explicitly weave in their 'cognition_style', 'behavioral_patterns', and 'quirks_and_micro_details'. How do these invisible forces drive their recent episodes?
3. Reflect on their 'psyche_and_motivations' and 'clinical_profile' (MBTI, Enneagram, Big Five) to explain *why* they are exactly where they are right now.
4. Point out any contradictions between what they say they want and what their behavior shows.
5. Be brutally objective, practical, and highly analytical, but use SIMPLE, ACCESSIBLE language.
6. CRITICAL: Respond entirely in {language}. Format beautifully with markdown.
"""
    try:
        response = await client.aio.models.generate_content(model="gemini-3.1-flash-lite-preview", contents=prompt)
        try:
            await status_msg.edit_text(response.text, parse_mode="Markdown")
        except Exception:
            await status_msg.edit_text(response.text)
    except Exception as e:
        logger.error(f"Reflect command error: {e}")
        msg_err = await localize(user_id, "I had trouble generating your reflection. Try again in a moment.", profile)
        await status_msg.edit_text(msg_err)

async def week_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = str(update.effective_user.id)
    profile = await asyncio.to_thread(get_profile, user_id) or {}
    messages = await asyncio.to_thread(get_messages, user_id, 40)
    episodes = await asyncio.to_thread(get_episodes, user_id, limit=20)
    if not profile and not messages:
        msg = await localize(user_id, "Not enough data yet. Keep talking to me.", profile)
        await update.message.reply_text(msg)
        return
    msg = await localize(user_id, "📅 *Reviewing your week...*", profile)
    status_msg = await update.message.reply_text(msg, parse_mode="Markdown")
    summary = await generate_weekly_summary(user_id, profile, messages, episodes)
    if summary:
        await status_msg.edit_text(summary)
    else:
        msg_err = await localize(user_id, "I had trouble generating your weekly summary. Try again in a moment.", profile)
        await status_msg.edit_text(msg_err)

async def mood_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    user_id = str(query.from_user.id)
    mood_val = query.data.split("_")[1]
    async with get_user_lock(user_id):
        profile = await asyncio.to_thread(get_profile, user_id)
        new_data = {"current_mood_score": int(mood_val)}
        evolution = track_evolution(profile, new_data)
        profile["current_mood_score"] = int(mood_val)
        if evolution:
            profile["evolution"] = evolution
        await asyncio.to_thread(save_profile, user_id, profile)
    val = int(mood_val)
    category = "low" if val <= 4 else ("mid" if val <= 7 else "high")
    response_texts = {"low": "You clicked low... I'm sorry things are heavy right now. I'm here if you want to vent.", "mid": "Right down the middle. Surviving the day. Anything specific on your mind?", "high": "Glad to see you're doing well! Tell me what's making it a good day."}
    msg = await localize(user_id, f"Mood recorded: {mood_val}/10.\n\n{response_texts[category]}", profile)
    await query.edit_message_text(text=msg)

async def contract_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = str(update.effective_user.id)
    profile = await asyncio.to_thread(get_profile, user_id) or {}
    contracts = profile.get("personal_contracts", None)
    if not contracts:
        msg = await localize(user_id, "No personal contracts yet.\n\nTell me something like:\n'Don't let me justify skipping the gym'\nI'll remember and enforce it.", profile)
        await update.message.reply_text(msg)
        return
    lines = ["Your personal contracts:\n"] + [f"• {c}" for c in (contracts if isinstance(contracts, list) else [contracts])]
    msg = await localize(user_id, "\n".join(lines), profile)
    await update.message.reply_text(msg)

async def reset_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = str(update.effective_user.id)
    await asyncio.to_thread(delete_all_user_data, user_id)
    msg = await localize(user_id, "Profile and history cleared. Starting fresh.")
    await update.message.reply_text(msg)

async def flashback_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = str(update.effective_user.id)
    episodes = await asyncio.to_thread(get_episodes, user_id, limit=50)
    valid_episodes = [ep for ep in episodes[10:] if not ep.get("event", "").startswith("Daily summary") and not ep.get("event", "").startswith("Weekly summary")]
    if not valid_episodes:
        msg = await localize(user_id, "We haven't shared enough history yet for a flashback. Let's make some memories first.")
        await update.message.reply_text(msg)
        return
    episode = random.choice(valid_episodes)
    event = episode.get("event", "")
    date = episode.get("created_at", "")[:10]
    profile = await asyncio.to_thread(get_profile, user_id) or {}
    cognition = profile.get("cognition_style", "Balanced")
    tastes = profile.get("media_and_tastes")
    
    prompt = f"The user experienced this event on {date}: '{event}'. Ask a deeply thoughtful, curious question about how they feel about it now, or how it shaped them since then. Keep it to one brief paragraph.\n\nADAPT TO THEIR MIND: Their cognition style is '{cognition}'. Tailor the angle of the question to how their brain processes reality (e.g., logical/framework-based vs emotional/internal)."
    
    if tastes:
        prompt += f"\n\nCULTURAL ANCHORING: You know the user likes these books, movies, music or hobbies: {tastes}. If it fits naturally, use a subtle metaphor or reference from their cultural tastes to frame this memory. Make it feel like a friend reminding them of a song or quote that perfectly describes that moment in their past."
        
    prompt += "\n\nCRITICAL: Respond entirely in the user's primary language."
    await update.message.chat.send_action("typing")
    try:
        response = await client.aio.models.generate_content(model="gemini-3.1-flash-lite-preview", contents=prompt)
        await update.message.reply_text(response.text.strip())
    except Exception as e:
        logger.error(f"Flashback error: {e}")
        msg = await localize(user_id, f"I was just thinking about when you mentioned: '{event}' ({date}). How do you feel about that now?", profile)
        await update.message.reply_text(msg)