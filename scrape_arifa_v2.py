"""ARIFA crawler v2 (improvements over v1):
- also crawls official subdomains (aiconference, ijait, aimarathon)
- removes nav/footer/header boilerplate so every page doesn't repeat the same footer text
- keeps query strings intact, skips login/register/pdf/image links
- saves heading-aware sections (better for chunking) + a content hash to detect changes on re-crawl
Run: pip install requests beautifulsoup4 ; python scrape_arifa_v2.py   -> arifa_raw_pages_v2.jsonl
"""
import json, re, time, hashlib
from urllib.parse import urljoin, urlparse, urldefrag
from urllib.robotparser import RobotFileParser
import requests
from bs4 import BeautifulSoup

START = ["https://arifa.org/", "https://aiconference.arifa.org/", "https://ijait.arifa.org/", "https://aimarathon.arifa.org/"]
ALLOWED = {"arifa.org", "www.arifa.org", "aiconference.arifa.org", "ijait.arifa.org", "aimarathon.arifa.org"}
SKIP = re.compile(r"(login|register|my-account|\.(pdf|jpg|jpeg|png|webp|svg|zip)$)", re.I)
UA = {"User-Agent": "ARIFA-Knowledge-Crawler/2.0 (official chatbot knowledge base)"}
OUT = "arifa_raw_pages_v2.jsonl"
s = requests.Session(); s.headers.update(UA)
_robots = {}

def allowed(url):
    host = urlparse(url).netloc
    if host not in _robots:
        rp = RobotFileParser(f"https://{host}/robots.txt")
        try: rp.read()
        except Exception: rp = None
        _robots[host] = rp
    rp = _robots[host]
    return True if rp is None else rp.can_fetch(UA["User-Agent"], url)

def extract(soup):
    for t in soup(["script", "style", "noscript", "svg", "nav", "footer", "header", "form"]):
        t.decompose()
    root = soup.find("main") or soup.body or soup
    sections, cur = [], {"heading": "", "text": []}
    for el in root.find_all(["h1", "h2", "h3", "h4", "p", "li", "td"]):
        txt = re.sub(r"\s+", " ", el.get_text(" ", strip=True))
        if not txt: continue
        if el.name in ("h1", "h2", "h3", "h4"):
            if cur["text"]: sections.append(cur)
            cur = {"heading": txt, "text": []}
        else:
            cur["text"].append(txt)
    if cur["text"]: sections.append(cur)
    return [{"heading": x["heading"], "text": " ".join(x["text"])} for x in sections]

seen, queue, rows = set(), list(START), []
while queue:
    url, _ = urldefrag(queue.pop(0))
    if url in seen: continue
    seen.add(url)
    p = urlparse(url)
    if p.netloc not in ALLOWED or SKIP.search(p.path) or not allowed(url): continue
    try:
        r = s.get(url, timeout=25)
        if r.status_code != 200 or "text/html" not in r.headers.get("content-type", ""): continue
        soup = BeautifulSoup(r.text, "html.parser")
        title = soup.title.get_text(" ", strip=True) if soup.title else ""
        for a in soup.find_all("a", href=True):
            nxt, _ = urldefrag(urljoin(url, a["href"]))
            if urlparse(nxt).netloc in ALLOWED and nxt not in seen: queue.append(nxt)
        secs = extract(soup)
        full = " ".join(x["text"] for x in secs)
        if len(full) < 80: continue
        rows.append({"url": url, "title": title, "sections": secs,
                     "hash": hashlib.sha256(full.encode()).hexdigest()[:16],
                     "retrieved_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
        print("OK", url); time.sleep(1.0)
    except Exception as e:
        print("ERR", url, e)
with open(OUT, "w", encoding="utf-8") as f:
    for r in rows: f.write(json.dumps(r, ensure_ascii=False) + "\n")
print(f"Saved {len(rows)} pages -> {OUT}")
