"""
ARIFA Chatbot — Streamlit + Groq (BURE)
========================================
Kwa testing ya haraka ya ARIFA chatbot.

Run:
    streamlit run streamlit_app.py

Inahitaji:
    - arifa_index/ folda (kutoka ingest.py)
    - system_prompt.txt
    - sentiment.py (optional)
    - GROQ_API_KEY environment variable
"""
import os
import streamlit as st
import chromadb
from groq import Groq
from sentence_transformers import SentenceTransformer

# ============================================================
# 1. PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="ARIFA Assistant",
    page_icon="🤖",
    layout="centered",
    initial_sidebar_state="expanded",
)

# ============================================================
# 2. CONFIG (environment)
# ============================================================
MODEL = os.getenv("LLM_MODEL", "llama-3.1-8b-instant")
DEFAULT_MIN_SCORE = float(os.getenv("MIN_SCORE", "0.35"))
DEFAULT_TOP_K = 5

# ============================================================
# 3. LOAD MODELS (cached — only once)
# ============================================================
@st.cache_resource(show_spinner="Loading models... / Inapakia modeli...")
def load_models():
    """Load embedder, collection, Groq client, and system prompt (once)."""
    embedder = SentenceTransformer("BAAI/bge-m3")
    col = chromadb.PersistentClient(path="./arifa_index").get_collection("arifa")
    groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    with open("system_prompt.txt", encoding="utf-8") as f:
        system = f.read()
    return embedder, col, groq_client, system

try:
    embedder, col, groq_client, SYSTEM = load_models()
except Exception as e:
    st.error(f"❌ Imeshindwa kupakia modeli / Failed to load models: {e}")
    st.info("💡 Hakikisha umefanya: `python ingest.py` kwanza, na `GROQ_API_KEY` imewekwa.")
    st.stop()

# ============================================================
# 4. SENTIMENT (optional)
# ============================================================
try:
    from sentiment import detect as detect_sentiment
    HAS_SENTIMENT = True
except ImportError:
    HAS_SENTIMENT = False
    def detect_sentiment(text):
        return {"label": "neutral", "escalate": False, "urgent": False, "language": "en"}

# ============================================================
# 5. SIDEBAR — Settings
# ============================================================
with st.sidebar:
    st.header("⚙️ Mipangilio / Settings")
    
    min_score = st.slider(
        "MIN_SCORE (ukali wa kukataa)",
        0.0, 1.0, DEFAULT_MIN_SCORE, 0.05,
        help="Juu = makini zaidi (hukataa maswali mengi). Chini = rahisi kujibu.",
    )
    
    top_k = st.slider(
        "TOP_K (vipande vya kutafuta)",
        1, 10, DEFAULT_TOP_K,
        help="Idadi ya vipande vya data vya kutumia kwa jibu.",
    )
    
    show_mood = st.checkbox("Onyesha hisia / Show sentiment", value=True)
    show_sources = st.checkbox("Onyesha sources", value=True)
    
    if st.button("🗑️ Futa mazungumzo / Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()
    
    st.divider()
    st.caption("**Hali / Status:**")
    st.caption(f"✅ Embedder: `BAAI/bge-m3`")
    st.caption(f"✅ Collection: `arifa`")
    st.caption(f"{'✅' if HAS_SENTIMENT else '❌'} Sentiment: `sentiment.py`")
    st.caption(f"✅ LLM: `{MODEL}` (Groq — BURE)")
    st.caption(f"{'✅' if os.getenv('GROQ_API_KEY') else '❌'} API Key: `{'set' if os.getenv('GROQ_API_KEY') else 'MISSING'}`")

# ============================================================
# 6. HEADER
# ============================================================
st.title(" ARIFA Assistant")
st.caption(
    "Uliza kuhusu ARIFA kwa Kiingereza au Kiswahili — "
    "utafiti, mafunzo, matukio, ajira na mawasiliano. | "
    "Ask about ARIFA in English or Kiswahili."
)

# ============================================================
# 7. CHAT HISTORY
# ============================================================
if "messages" not in st.session_state:
    st.session_state.messages = []

# Welcome message
if not st.session_state.messages:
    with st.chat_message("assistant"):
        st.write(
            "👋 **Habari! / Hello!**\n\n"
            "Mimi ni **ARIFA Assistant**. Naweza kukusaidia na maswali kuhusu:\n"
            "- 🎓 Mafunzo na vyeti (certifications)\n"
            "- 📅 Matukio (ICAFoW, AI Marathon)\n"
            "- 🔬 Utafiti na miradi\n"
            "- 💼 Ajira na ushirikiano\n"
            "- 📞 Mawasiliano\n\n"
            "Uliza kwa Kiingereza au Kiswahili!"
        )

# Render history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        
        if msg.get("sources") and show_sources:
            with st.expander(f"📚 Sources ({len(msg['sources'])})"):
                for s in msg["sources"]:
                    st.markdown(f"- {s}")
        
        if msg.get("sentiment") and show_mood and msg["role"] == "assistant":
            mood_emoji = {
                "positive": "😊",
                "negative": "😟",
                "neutral": "😐",
            }.get(msg["sentiment"], "😐")
            st.caption(f"{mood_emoji} Hisia / Mood: `{msg['sentiment']}`")

# ============================================================
# 8. CHAT INPUT
# ============================================================
if prompt := st.chat_input("Uliza kuhusu ARIFA... / Ask about ARIFA..."):

    # --- Show user message ---
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # --- Generate assistant response ---
    with st.chat_message("assistant"):
        with st.spinner("Nafikiria... / Thinking..."):
            try:
                # 1. Sentiment
                sent = detect_sentiment(prompt)
                lang = sent.get("language", "en")

                # 2. Embed & search
                q = embedder.encode([prompt], normalize_embeddings=True).tolist()
                res = col.query(query_embeddings=q, n_results=top_k * 2)

                # 3. Deduplicate (EN/SW twins)
                hits, seen = [], set()
                for doc, meta, dist in zip(
                    res["documents"][0],
                    res["metadatas"][0],
                    res["distances"][0],
                ):
                    score = 1 - dist
                    if meta["id"] in seen:
                        continue
                    seen.add(meta["id"])
                    hits.append((score, doc, meta))
                hits = hits[:top_k]

                # 4. Out-of-scope check
                if not hits or hits[0][0] < min_score:
                    if lang == "sw":
                        answer = (
                            "Ninaweza kusaidia maswali kuhusu ARIFA tu — "
                            "utafiti, mafunzo, matukio, ajira na mawasiliano. "
                            "Ungependa kujua nini?"
                        )
                    else:
                        answer = (
                            "I can only help with questions about ARIFA — "
                            "our research, training, events, careers and contacts. "
                            "What would you like to know?"
                        )
                    st.markdown(answer)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "sources": [],
                        "sentiment": sent.get("label", "neutral"),
                    })
                    st.stop()

                # 5. Build context
                context = "\n\n".join(
                    f"[{m['source_url']}]\n{d}" for _, d, m in hits
                )

                # 6. Sentiment hint (for LLM)
                hint = (
                    f"SENTIMENT_HINT: label={sent.get('label')}, "
                    f"escalate={sent.get('escalate')}, "
                    f"urgent={sent.get('urgent')}"
                )

                # 7. Build messages (history + current)
                messages = [{"role": "system", "content": SYSTEM}]
                for m in st.session_state.messages[-6:]:
                    if m["role"] in ("user", "assistant"):
                        messages.append({"role": m["role"], "content": m["content"]})

                messages.append({
                    "role": "user",
                    "content": (
                        f"{hint}\n\n"
                        f"CONTEXT:\n{context}\n\n"
                        f"USER QUESTION:\n{prompt}"
                    ),
                })

                # 8. Call Groq (BURE)
                out = groq_client.chat.completions.create(
                    model=MODEL,
                    messages=messages,
                    temperature=0.2,
                    max_tokens=500,
                )
                answer = out.choices[0].message.content
                sources = sorted({m["source_url"] for _, _, m in hits})

                # 9. Display
                st.markdown(answer)

                if sources and show_sources:
                    with st.expander(f"📚 Sources ({len(sources)})"):
                        for s in sources:
                            st.markdown(f"- {s}")

                if show_mood:
                    mood_emoji = {
                        "positive": "😊",
                        "negative": "😟",
                        "neutral": "😐",
                    }.get(sent.get("label", "neutral"), "😐")
                    st.caption(
                        f"{mood_emoji} Hisia / Mood: `{sent.get('label', 'neutral')}` "
                        f"| escalate: `{sent.get('escalate', False)}`"
                    )

                # 10. Save
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "sources": sources,
                    "sentiment": sent.get("label", "neutral"),
                })

            except Exception as e:
                error_msg = f"❌ Kosa / Error: `{type(e).__name__}: {e}`"
                st.error(error_msg)
                st.info(
                    "💡 Angalia:\n"
                    "- `GROQ_API_KEY` imewekwa?\n"
                    "- `arifa_index/` ipo? (`python ingest.py`)\n"
                    "- Mtandao unafanya kazi?"
                )

# ============================================================
# 9. FOOTER
# ============================================================
st.divider()
st.caption(
    "🤖 ARIFA Assistant (Groq — BURE) — AI inaweza kukosea. Thibitisha kwenye "
    "[arifa.org](https://arifa.org) au info@arifa.org. | "
    "AI can make mistakes — verify on [arifa.org](https://arifa.org)."
)
