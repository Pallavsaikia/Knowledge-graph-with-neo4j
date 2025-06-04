# 🧠 Neo4j-Powered Knowledge Graph with LlamaIndex RAG & LangGraph Agents

A scalable and intelligent pipeline for building **LLM applications** powered by a **Neo4j knowledge graph**, **LlamaIndex** for RAG, and **LangGraph** for agent-based orchestration.

---

## 📌 Project Goals

- Use **Neo4j** to structure and query domain knowledge in graph form.
- Enable **semantic and graph-aware retrieval** using **LlamaIndex**.
- Integrate **LangGraph** for multi-agent orchestration and reasoning.
- Build a flexible and production-ready AI system with memory, tool use, and human-in-the-loop capability.

---

## 🧱 Tech Stack

| Component        | Role                                             |
|------------------|--------------------------------------------------|
| Neo4j            | Graph database for structured knowledge storage |
| LlamaIndex       | Connects the KG with LLMs for intelligent RAG   |
| LangGraph        | Agent orchestration for reasoning workflows     |
| OpenAI / Azure   | Backend LLMs (e.g. GPT-4, GPT-4o, GPT-4-turbo)   |
| Python + FastAPI | Core implementation and API integration         |

---

## 🗂 Project Structure

```bash
.
├── data/
│   └── healthcare_dataset.csv
├── graph/
│   └── neo4j_loader.py        # Loads CSV into Neo4j
├── llm/
│   ├── llamaindex_rag.py      # RAG pipeline using LlamaIndex + Neo4j
│   └── langgraph_agents.py    # LangGraph workflow logic
├── utils/
│   └── config.py              # Load .env and shared configs
├── .env
├── requirements.txt
└── README.md
