"""Run the eval set against the local API and print answers for human review.
python run_eval.py   (start app.py first)"""
import json, requests
for line in open("eval_set.jsonl", encoding="utf-8"):
    e = json.loads(line)
    r = requests.post("http://localhost:8000/chat", json={"message": e["q"]}).json()
    ok_src = (not e["must_cite"]) or any(e["must_cite"] in s for s in r["sources"]) or e["must_cite"] in r["answer"]
    print(f"\nQ: {e['q']}\nEXPECT: {e['expect']}\nA: {r['answer']}\nSOURCE OK: {ok_src}")
