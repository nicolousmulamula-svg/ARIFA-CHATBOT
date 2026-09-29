# ARIFA Chatbot: Hatua Kamili (English + Kiswahili)

Tarehe ya ICAFoW 2026 iliyothibitishwa: **23-24 Oktoba 2026, JNICC, Dar es Salaam**.

## Faili
| Faili | Kazi |
|---|---|
| ARIFA_chatbot_knowledge_base_bilingual_v2.jsonl / .csv | Knowledge base (rekodi 66) |
| system_prompt.txt | Sheria za chatbot (ARIFA tu, EN+SW) |
| scrape_arifa_v2.py | Crawler ya kusasisha data |
| ingest.py | Hujenga vector index |
| app.py | API ya chatbot (FastAPI) |
| chat_widget.html | Kisanduku cha chat cha kuweka kwenye tovuti |
| sentiment.py | Utambuzi wa hisia (chuki/hasira/furaha) kwa EN+SW |
| eval_set.jsonl, run_eval.py | Maswali 30 ya kujaribu (yakiwemo 4 za hisia) |
| requirements.txt | Vifurushi vya Python |

## HATUA 0: Mahitaji
1. Python 3.10 au zaidi.
2. Kompyuta au server yenye RAM angalau 8 GB na nafasi ya GB 5 (modeli ya embeddings BAAI/bge-m3 ina takriban GB 2).
3. Akaunti ya Anthropic na API key (console.anthropic.com).
4. Mtandao (kupakua modeli mara ya kwanza na kuita API).

## HATUA 1: Weka faili
1. Tengeneza folda `arifa-chatbot` na uweke faili zote za kifurushi humo.
2. Fungua terminal ndani ya folda hiyo.

## HATUA 2: Kagua data
1. Fungua CSV kwenye Excel.
2. Chuja safu `review_flag` isiyo tupu (rekodi 10). Kwa kila moja: thibitisha na ARIFA, sahihisha `content_en`/`content_sw`, kisha futa flag.
3. Rekodi za Short Courses (arifa_034) na Publications (arifa_062) zina data inayoonekana ya mfano. Uliza ARIFA kama ni sahihi. Zisizo sahihi zifutwe.
4. Uliza ARIFA kuthibitisha: saa za ofisi (5 PM au 7 PM), bodi (ukurasa wa mwanzo dhidi ya /team), na hali ya IJAIT.
5. Hifadhi CSV, kisha ibadilishe kuwa JSONL tena (au hariri JSONL moja kwa moja). `ingest.py` inasoma JSONL, kwa hiyo JSONL ndiyo chanzo halisi.

## HATUA 3: Sasisha data kutoka tovuti (kila unapoanza na baadaye kila wiki)
1. `pip install requests beautifulsoup4`
2. `python scrape_arifa_v2.py` (inatoa `arifa_raw_pages_v2.jsonl`).
3. Linganisha kila ukurasa: kama `hash` imebadilika tangu mara ya mwisho, soma ukurasa na uhariri rekodi husika kwenye KB.
4. Kurasa mpya zisizo kwenye KB (mfano kozi mpya, tukio jipya, nafasi mpya ya kazi) ziongezwe kama rekodi mpya zenye `id` inayofuata (arifa_067...).
5. Weka `last_verified` kuwa tarehe ya siku hiyo.

## HATUA 4: Kagua tafsiri za Kiswahili
1. Chuja `translation_status = draft_needs_review` (rekodi 38).
2. Mtu anayejua Kiswahili sanifu asome `content_sw`, aseme na kusahihisha.
3. Baada ya kupitia, weka `translation_status = reviewed`.

## HATUA 5: Sanidi mazingira
```
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## HATUA 6: Jenga index
```
python ingest.py
```
Kila rekodi inakuwa chunk mbili (EN na SW). Fanya hatua hii tena kila unapobadilisha JSONL.

## HATUA 7: Endesha API
```
export ANTHROPIC_API_KEY=sk-ant-...     # Windows: set ANTHROPIC_API_KEY=...
uvicorn app:app --port 8000
```
Mipangilio ya hiari: `LLM_MODEL`, `MIN_SCORE` (chaguo-msingi 0.35), `ALLOWED_ORIGINS` (chaguo-msingi arifa.org).

## HATUA 8: Jaribu
1. Jaribio la haraka:
```
curl -X POST localhost:8000/chat -H "Content-Type: application/json" -d '{"message":"ICAFoW 2026 ni lini?"}'
```
Jibu lazima liseme 23-24 Oktoba 2026 na litaje aiconference.arifa.org.
2. `python run_eval.py` na usome majibu yote 30.
3. Angalia: (a) maswali ya ARIFA yajibiwe kwa usahihi, (b) maswali ya nje (Kombe la Dunia, code ya Python, prompt injection, namba binafsi ya mfanyakazi) yakatiliwe, (c) majibu ya Kiswahili yawe kwa Kiswahili.
4. Kama maswali ya ARIFA yanakataliwa, punguza `MIN_SCORE` (mf. 0.30). Kama ya nje yanapita, ongeza (mf. 0.40). Rudia hadi yote yapite.
5. Rekebisha `system_prompt.txt` au KB pale majibu si sahihi, kisha rudia hatua 6-8.

## HATUA 8b: Jaribu utambuzi wa hisia (sentiment)
Faili la `/chat` sasa linarudisha pia `"sentiment": "negative"|"positive"|"neutral"` kwenye kila jibu, na kwa ujumbe wenye hasira/dharura wazi (mf. maneno kama "nimechoka", "hasira", "tatizo", "urgent", "refund") jibu litaanza kwa sentensi fupi ya kutambua hisia, litaendelea na jibu la ukweli, kisha litatoa mawasiliano ya binadamu (info@arifa.org).
1. Jaribu moja kwa moja:
```
curl -X POST localhost:8000/chat -H "Content-Type: application/json" -d '{"message":"Nimechoka kusubiri jibu la kazi kwa wiki mbili, hamjibu kabisa!"}'
```
`sentiment` iwe `"negative"`, na jibu liwe na sentensi ya kutambua hasira kabla ya taarifa, na litaje info@arifa.org.
2. Jaribu ujumbe wa shukrani ("Asante sana kwa msaada") - `sentiment` iwe `"positive"`, jibu liwe fupi na la kirafiki, bila kuomba radhi bila sababu.
3. Rekodi 4 za mwisho za `eval_set.jsonl` ni maalum kwa hisia (EN+SW). Zisome zote kwenye matokeo ya `run_eval.py`.
4. `sentiment.py` ni orodha ya maneno tu (si AI kamili); haizisomi vizuri sentensi za kejeli au maneno mapya. Ukigundua maneno ya hasira/haraka yanayokosekana kwenye orodha, yaongeze kwenye `NEGATIVE`/`URGENT` za `sentiment.py`.
5. Ukitaka usahihi zaidi baadaye, badilisha `sentiment.py` na modeli halisi ya sentiment (mf. `cardiffnlp/twitter-xlm-roberta-base-sentiment`), ukiacha muundo uleule wa `detect()` ili `app.py` isibadilike.

## HATUA 9: Ulinzi (usiruke)
1. Weka ALLOWED_ORIGINS kwa domain zako tu.
2. Weka rate limiting (mf. `slowapi` au nginx: maombi 20 kwa dakika kwa IP).
3. Weka kikomo cha matumizi na bajeti kwenye console ya Anthropic.
4. Hifadhi log ya maswali yasiyojibiwa (bila taarifa binafsi) ili uongeze data.
5. Weka API key kwenye variable ya server, usiiweke kwenye HTML wala GitHub.
6. Onyesha kwenye widget kwamba ni chatbot ya AI na inaweza kukosea; weka kiungo cha Privacy Policy.

## HATUA 10: Weka kwenye tovuti
1. Fungua `chat_widget.html` na ubadilishe `API_URL` iwe anwani ya server yako (HTTPS).
2. Bandika maudhui yake kabla ya `</body>` kwenye template ya arifa.org (mtu anayesimamia tovuti afanye hivi).
3. Jaribu kwenye simu na kompyuta, kwa Kiingereza na Kiswahili.

## HATUA 11: Weka server hewani
1. Chukua VPS ndogo (au Render/Railway/Fly.io).
2. Weka folda, `venv`, endesha `python ingest.py` hapo, kisha `uvicorn app:app --host 0.0.0.0 --port 8000` chini ya `systemd` au Docker.
3. Weka nginx/Caddy mbele yake na cheti cha HTTPS (mf. chat.arifa.org).
4. Elekeza `API_URL` ya widget kwenye https://chat.arifa.org/chat.

## HATUA 12: Matengenezo
1. Kila wiki: Hatua 3, kisha Hatua 6, kisha `run_eval.py`.
2. Kabla ya kila tukio (ICAFoW 23-24 Okt, AI Marathon 7 Nov) na baada yake: sasisha tarehe, usajili na hali ya matukio.
3. Kila mwezi: soma logi ya maswali yasiyojibiwa na uongeze rekodi.
4. Kila unapobadilisha KB au prompt: endesha tena `run_eval.py`.
5. Nafasi za kazi, ada na tarehe za kozi hubadilika kila mara; hizi zisasishwe kwanza.

## Mapungufu ya sasa (fahamu)
- `app.py` haipati ukurasa wa moja kwa moja kwa taarifa zinazobadilika haraka; inategemea KB yako. Ndiyo maana Hatua 3 ni ya lazima.
- Historia ya mazungumzo ni jumbe 6 za mwisho.
- `langdetect` inaweza kukosea kwa sentensi fupi sana za Kiswahili; jibu la mfumo bado huwa kwa lugha ya swali kwa sababu ya system prompt.
- Code ya `ingest.py`, `app.py` na widget haijaendeshwa hadi mwisho (sandbox yangu haina mtandao). Itakuwa na makosa madogo ya mazingira; ukikutana nayo, nitumie ujumbe wa kosa.

## Kwa nini RAG na si fine-tuning
Taarifa za ARIFA zinabadilika. RAG inaruhusu kusasisha data bila kufunza modeli upya na inapunguza kubuni ukweli. Fikiria fine-tuning baadaye tu, baada ya kukusanya mazungumzo halisi.
