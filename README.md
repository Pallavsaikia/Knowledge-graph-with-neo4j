# 🧠 Neo4j-Powered Knowledge Graph with LlamaIndex RAG, LangGraph Agents & Real-Time Voice Bot

A scalable and modular LLM-based system combining **Neo4j**, **LlamaIndex**, **LangGraph**, and a **FastAPI voice bot** for intelligent, real-time interactions via audio.

---

## 📌 Project Goals

- Structure and store knowledge in a **Neo4j graph database**.
- Enable **semantic and graph-aware retrieval** using **LlamaIndex**.
- Integrate **LangGraph** for multi-agent reasoning and workflow orchestration.
- Build a real-time **audio-based interface** with bot response capabilities.
- Use **silence detection** and **noise filtering** to control when the bot responds.
- Ensure **production-readiness** with modular architecture and async components.

---

## 🗂️ Project Structure

```bash
.
├── audio/                        # Audio processing utilities or samples
├── data/                         # Source datasets for KG population        
├── rag/                          # RAG logic and knowledge graph integration
├── recordings/                   # Saved user call recordings
├── static/
│   └── index.html                # Frontend interface (WebRTC-based)
├── transcription/
│   └── transcriber.py           # Real-time audio transcription & silence detection
├── utils/
│   └── config.py                 # Loads `.env` and shared constants
├── .env                          # Environment variables (Neo4j, API keys, etc.)
├── .gitignore
├── audio.py                      # Audio-related helper logic
├── bot.py                        # Bot logic (connects to calls, responds intelligently)
├── config.py                     # Central config loader
├── main.py                       # FastAPI app with WebSocket & routing logic
├── README.md
└── upload_data.py                # Script to upload data into Neo4j
