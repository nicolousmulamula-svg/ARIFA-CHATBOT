"""ARIFA chatbot API — BURE kwa Groq.
Run: set GROQ_API_KEY=gsk_... && uvicorn app:app --port 8000
POST /chat {"message": "...", "history": [{"role":"user|assistant","content":"..."}]}
"""
import os
import chromadb
from groq import Groq
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

# ============================================================
# CONFIG
# ============================================================
MODEL = os.getenv("LLM_MODEL", "openai/gpt-oss-120b")
MIN_SCORE = float(os.getenv("MIN_SCORE", "0.35"))
TOP_K = 5
SYSTEM = open("system_prompt.txt", encoding="utf-8").read()

# ============================================================
# LOAD MODELS
# ============================================================
embedder = SentenceTransformer("BAAI/bge-m3")
col = chromadb.PersistentClient(path="./arifa_index").get_collection("arifa")
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

app = FastAPI(title="ARIFA Assistant")

ALLOWED_ORIGINS = os.getenv(
    "ALLOWED_ORIGINS",
    "https://arifa.org,https://www.arifa.org"
).split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["POST"],
    allow_headers=["*"],
)

MAX_CHARS = 500

# ============================================================
# MODELS
# ============================================================
class Turn(BaseModel):
    role: str
    content: str

class Chat(BaseModel):
    message: str
    history: list[Turn] = []

OFF_TOPIC = {
    "en": "I can only help with questions about ARIFA - our research, training, events, careers and contacts. What would you like to know?",
    "sw": "Ninaweza kusaidia maswali kuhusu ARIFA tu - utafiti, mafunzo, matukio, ajira na mawasiliano. Ungependa kujua nini?",
}
OFF_TOPIC_ESCALATE = {
    "en": " I can also see this may be frustrating - if it's urgent, please email info@arifa.org and our team will help directly.",
    "sw": " Naona hii inaweza kukukera - kama ni jambo la haraka, tafadhali tuandikie info@arifa.org na timu yetu itakusaidia moja kwa moja.",
}

# ============================================================
# SENTIMENT
# ============================================================
try:
    from sentiment import detect as detect_sentiment
    HAS_SENTIMENT = True
except ImportError:
    HAS_SENTIMENT = False
    def detect_sentiment(text):
        return {"label": "neutral", "escalate": False, "urgent": False, "language": "en"}

# ============================================================
# LANGUAGE
# ============================================================
def detect(text):
    try:
        from langdetect import detect as d
        return "sw" if d(text) == "sw" else "en"
    except Exception:
        return "en"

# ============================================================
# CHAT ENDPOINT
# ============================================================
@app.post("/chat")
def chat(req: Chat):
    if not req.message.strip() or len(req.message) > MAX_CHARS:
        raise HTTPException(400, f"Message must be 1-{MAX_CHARS} characters.")

    lang = detect(req.message)
    sentiment = detect_sentiment(req.message)

    # --- Search ---
    q = embedder.encode([req.message], normalize_embeddings=True).tolist()
    res = col.query(query_embeddings=q, n_results=TOP_K * 2)

    hits, seen = [], set()
    for doc, meta, dist in zip(
        res["documents"][0], res["metadatas"][0], res["distances"][0]
    ):
        score = 1 - dist
        if meta["id"] in seen:
            continue
        seen.add(meta["id"])
        hits.append((score, doc, meta))
    hits = hits[:TOP_K]

    # --- Out of scope ---
    if not hits or hits[0][0] < MIN_SCORE:
        answer = OFF_TOPIC[lang] + (
            OFF_TOPIC_ESCALATE[lang] if sentiment.get("escalate") else ""
        )
        return {"answer": answer, "sources": [], "sentiment": sentiment["label"]}

    # --- Build context ---
    context = "\n\n".join(
        f"[{m['source_url']}]\n{d}" + (
            f"\n(NOTE: {m['review_flag']})" if m["review_flag"] else ""
        )
        for _, d, m in hits
    )

    hint = (
        f"SENTIMENT_HINT: label={sentiment['label']}, "
        f"urgent={sentiment['urgent']}, "
        f"escalate={sentiment['escalate']}"
    )

    # --- Build messages for Groq ---
    messages = [{"role": "system", "content": SYSTEM}]
    for t in req.history[-6:]:
        messages.append({"role": t.role, "content": t.content})
    messages.append({
        "role": "user",
        "content": f"{hint}\n\nCONTEXT:\n{context}\n\nUSER QUESTION:\n{req.message}",
    })

    # --- Call Groq (BURE) ---
    out = groq_client.chat.completions.create(
        model=MODEL,
        messages=messages,
        temperature=0.2,
        max_tokens=500,
    )
    answer = out.choices[0].message.content

    return {
        "answer": answer,
        "sources": sorted({m["source_url"] for _, _, m in hits}),
        "sentiment": sentiment["label"],
    }