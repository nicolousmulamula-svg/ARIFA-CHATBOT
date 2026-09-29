"""Build the vector index from the bilingual knowledge base.
Run: python ingest.py
Each record becomes two chunks (English + Kiswahili) so either language retrieves the same fact.
"""
import json, chromadb
from sentence_transformers import SentenceTransformer

KB = "ARIFA_chatbot_knowledge_base_bilingual_v2.jsonl"
model = SentenceTransformer("BAAI/bge-m3")          # multilingual, handles English + Kiswahili
client = chromadb.PersistentClient(path="./arifa_index")
try: client.delete_collection("arifa")
except Exception: pass
col = client.create_collection("arifa", metadata={"hnsw:space": "cosine"})

ids, docs, metas = [], [], []
for line in open(KB, encoding="utf-8"):
    r = json.loads(line)
    for lang in ("en", "sw"):
        text = (r.get(f"content_{lang}") or "").strip()
        if not text: continue
        ids.append(f'{r["id"]}_{lang}')
        docs.append(f'{r["title"]}. {text}')
        metas.append({"id": r["id"], "lang": lang, "category": r["category"], "title": r["title"],
                      "source_url": r["source_url"], "volatility": r.get("volatility", ""),
                      "review_flag": r.get("review_flag", "")})
emb = model.encode(docs, normalize_embeddings=True, batch_size=16, show_progress_bar=True).tolist()
col.add(ids=ids, documents=docs, embeddings=emb, metadatas=metas)
print("Indexed", len(ids), "chunks")
