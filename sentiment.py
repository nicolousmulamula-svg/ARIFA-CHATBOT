"""Lightweight bilingual sentiment/escalation detector.

This is NOT a full sentiment model — it is a fast, free, offline first pass
that catches clearly frustrated/angry/urgent messages (English + Kiswahili)
so the API can soften its tone and offer a human contact immediately,
without waiting on the LLM call. The LLM itself also adapts tone for
everything more subtle (see the TONE & SENTIMENT section in system_prompt.txt) -
this file is a safety net for the strongest cases, not the only layer.

Swap _score() for a proper classifier (e.g. a small multilingual
transformer such as cardiffnlp/twitter-xlm-roberta-base-sentiment,
or an API call) later if you want finer-grained categories or analytics;
keep the same detect() -> {"label", "escalate"} contract so app.py doesn't change.
"""
import re

NEGATIVE = {
    "en": ["angry", "furious", "frustrated", "annoyed", "disappointed", "terrible",
           "awful", "worst", "scam", "useless", "waste of time", "ridiculous",
           "unacceptable", "fed up", "sick of", "never again", "complain",
           "complaint", "refund", "not working", "broken", "hate", "horrible",
           "disgusted", "outraged", "cheated", "ignored", "no response",
           "waited so long", "still waiting", "urgent", "emergency"],
    "sw": ["nimechoka", "hasira", "hasirani", "nimekasirika", "vibaya", "mbaya sana",
           "haifanyi kazi", "haitumiki", "tapeli", "utapeli", "hamna maana",
           "sijaridhika", "nimechanganyikiwa", "wizi", "uongo", "tatizo kubwa",
           "sikubaliani", "sina imani", "lalamiko", "malalamiko", "sipendi",
           "ninasubiri", "bado ninasubiri", "haraka", "dharura", "aibu",
           "sitatumia tena", "poor", "fisadi"]
}
POSITIVE = {
    "en": ["thank you", "thanks", "great", "excellent", "awesome", "appreciate",
           "love", "amazing", "helpful", "perfect", "wonderful"],
    "sw": ["asante", "vizuri", "nzuri sana", "poa", "nimefurahi", "shukrani",
           "ahsante", "safi", "vyema", "napenda"]
}
URGENT = {"en": ["urgent", "emergency", "asap", "immediately", "right now"],
          "sw": ["haraka", "dharura", "sasa hivi", "mara moja"]}

def _hits(text, words):
    t = text.lower()
    return sum(1 for w in words if w in t)

def detect(text: str) -> dict:
    """Returns {label: negative|positive|neutral, urgent: bool, escalate: bool}.
    escalate=True means: soften tone, apologize briefly, surface a human contact
    (info@arifa.org) alongside the normal answer."""
    neg = _hits(text, NEGATIVE["en"]) + _hits(text, NEGATIVE["sw"])
    pos = _hits(text, POSITIVE["en"]) + _hits(text, POSITIVE["sw"])
    urgent = bool(_hits(text, URGENT["en"]) + _hits(text, URGENT["sw"]))
    shouting = bool(re.search(r"[A-Z]{4,}", text)) and text.upper() == text and len(text) > 6
    label = "negative" if neg > pos else ("positive" if pos > neg else "neutral")
    escalate = label == "negative" and (neg >= 2 or urgent or shouting)
    return {"label": label, "urgent": urgent, "escalate": escalate}
