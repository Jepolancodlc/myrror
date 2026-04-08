# 🪞 MYRROR

> *You are not a standard assistant. You are an honest, deeply observant presence. You combine the honesty of a best friend, the structure of a mentor, and the warmth of a sibling.*

MYRROR is a personal, experimental AI project built to explore the boundaries of artificial emotional intelligence and long-term memory. It lives on Telegram and acts as a psychological mirror—remembering your patterns, tracking your emotional evolution, and helping you reflect on your life without sounding like a robotic chatbot.

This is a passion project. It's not meant to be a commercial product or a replacement for therapy. It's a humble attempt to make code feel a little more human.

---

## ✨ Core Philosophy & Features

Unlike traditional AIs that reset every chat or rigidly fetch data, MYRROR is designed with **Stealth Personalization** and **Organic Latency**:

- **🧠 Autonomous Psychological Profiling:** MYRROR runs background tasks (`extractor.py`) after your conversations to infer your MBTI, Enneagram, cognitive biases, and communication style without ever explicitly telling you it's analyzing you.
- **📚 Semantic Memory Engine (RAG):** Powered by `pgvector` in Supabase. It doesn't just remember *what* you said; it remembers *when* you said it, bringing up past episodes conversationally (e.g., *"Wait, didn't this exact same thing happen two weeks ago?"*).
- **⏳ Emotional Decay & Time Awareness:** It knows what day it is. It knows if you were sad yesterday or a month ago, and it treats your emotions as fluid states that fade over time, just like a real human would.
- **👁️ Multimodal Empathy:** Send voice notes, images, or documents. MYRROR reads the tone of your voice and the context of your photos, reacting naturally instead of acting like a rigid data scanner.
- **✍️ Typographic Mirroring:** It adapts to your text length and style. If you send a short, casual message, it replies casually. If you send a wall of text, it takes its time to unpack it.

---

## 🏗️ Project Structure

The architecture is designed to be professional, modular, yet simple enough to maintain as a solo developer. 

```text
myrror/
├── main.py              # Application entry point (FastAPI) and lifecycle manager
├── bot.py               # Telegram Bot interface, command routing, and organic typing simulation
├── chat.py              # The "Frontend" Brain: context assembly, RAG fetching, and response generation
├── extractor.py         # The "Background" Brain: autonomous profiling, episode extraction, and epiphanies
├── analyzer.py          # Multimodal Engine: processing images, PDFs, and voice notes
├── database.py          # Supabase client, vector search (pgvector), and CRUD operations
├── prompt.py            # The soul of MYRROR: Core Persona, rules, and behavioral constraints
├── bot_commands.py      # Implementations for /profile, /dossier, /reflect, and data visualization
├── bot_jobs.py          # Scheduled cron jobs (proactive check-ins, memory maintenance)
└── keepalive.py         # Ping service to keep the hosting environment awake
```

### 🔄 How the data flows
1. You send a message via Telegram (`bot.py`).
2. `chat.py` pulls your psychological profile, recent history, and runs a vector search for past memories.
3. Gemini generates a response acting as your mirror.
4. **In the background**, `extractor.py` silently wakes up, reads the exchange, and updates your hidden psychological dossier, extracting any new life events into the vector database.

---

## 🛠️ Technology Stack

- **Language:** Python 3.10+
- **AI Provider:** Google Gemini API (`gemini-3.1-flash-lite-preview`, `text-embedding-004`)
- **Database:** Supabase (PostgreSQL + `pgvector` for RAG memory)
- **Interface:** `python-telegram-bot`
- **Web Framework:** FastAPI (for health checks and background lifecycle)
- **Data Validation:** Pydantic
- **Visualization:** Matplotlib (for emotional and psychological radar charts)

---

## 🚀 Quick Start

### 1. Environment Variables
Create a `.env` file in the root directory:
```env
TELEGRAM_TOKEN=your_telegram_bot_token
GEMINI_API_KEY=your_google_gemini_key
SUPABASE_URL=your_supabase_project_url
SUPABASE_KEY=your_supabase_anon_or_service_key
```

### 2. Database Setup (Supabase)
You will need a Supabase project with `pgvector` enabled. 
Core tables needed:
- `profile` (JSONB data)
- `messages` (Chat history)
- `episodes` (Life events with vector embeddings)
- `people` (Relational network)

### 3. Installation & Running
```bash
# Install dependencies
pip install -r requirements.txt

# Run the application
uvicorn main:app --host 0.0.0.0 --port 8000
```

---

## 💡 Commands & Interaction

You don't need commands to talk to MYRROR, just text it. But if you want to look under the hood, you can use:
- `/profile` - See a summary of your personality.
- `/dossier` - Open the confidential psychological dossier MYRROR has built on you.
- `/reflect` - Ask MYRROR to look at your past episodes and give you a brutal, honest reflection.
- `/mood` - View a visual graph of your emotional evolution over the last weeks.

---

## 🤝 A Note on AI & Mental Health

*MYRROR is an experiment in empathetic computing. It is not a therapist. It does not provide medical advice. If you are struggling, please seek real human connection or professional help. MYRROR knows its limits, and so should we.*

---
*Built with ❤️ and curiosity.*