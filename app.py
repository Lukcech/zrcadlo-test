# -*- coding: utf-8 -*-
import streamlit as st
import requests
import json
import sqlite3
import uuid
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct

API_KEY = "AQ.Ab8RN6LcRITU_9D9RMh8jYDkFLPLLgVantVqKGzDhnpmxrzZbw"
DB_PATH = "./qdrant_db"
GRAPH_DB_PATH = "znalostni_graf.db"

st.set_page_config(page_title="Projekt Zrcadlo", page_icon="🪞", layout="wide")

# --- DB INICIALIZACE A MIGRACE ---
conn = sqlite3.connect(GRAPH_DB_PATH)
cursor = conn.cursor()
cursor.execute("PRAGMA table_info(triples)")
columns = [col[1] for col in cursor.fetchall()]

if not columns:
    cursor.execute('''
        CREATE TABLE triples (
            user_id TEXT DEFAULT 'karel',
            subject TEXT, relation TEXT, object TEXT,
            UNIQUE(user_id, subject, relation, object)
        )
    ''')
elif "user_id" not in columns:
    cursor.execute("ALTER TABLE triples ADD COLUMN user_id TEXT DEFAULT 'karel'")

conn.commit()
conn.close()

# --- BOČNÍ PANEL ---
with st.sidebar:
    st.header("👤 Profil & Paměť")
    active_user = st.text_input("Aktivní Uživatel (user_id):", value="karel").strip().lower()
    st.divider()
    st.caption(f"Živý náhled paměti pro: **{active_user}**")

    # Qdrant počítadlo
    try:
        client = QdrantClient(path=DB_PATH)
        all_points = client.scroll(collection_name="zrcadlo_pamet", limit=200)[0]
        user_points = [p for p in all_points if p.payload.get("user_id", "karel") == active_user]
        pocet_vektoru = len(user_points)
        client.close()
    except Exception:
        pocet_vektoru = 0

    st.metric("Vektory uživatele", pocet_vektoru)

    # SQLite počítadlo
    conn = sqlite3.connect(GRAPH_DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT subject, relation, object FROM triples WHERE user_id = ?", (active_user,))
    uzivatelske_trojice = cursor.fetchall()
    conn.close()

    st.metric("Grafové vazby", len(uzivatelske_trojice))
    
    with st.expander("🕸️ Znalostní graf uživatele"):
        if uzivatelske_trojice:
            for t in uzivatelske_trojice:
                st.write(f"**{t[0]}** `-{t[1]}->` **{t[2]}**")
        else:
            st.info("Tento uživatel nemá v grafu žádné vazby.")

# --- POMOCNÉ FUNKCE PRO AUTOMATICKOU PAMĚŤ ---
def ziskej_embedding(text):
    """Stabilní získání vektoru s ošetřením chyb."""
    url_emb = f"https://generativelanguage.googleapis.com/v1beta/models/text-embedding-004:embedContent?key={API_KEY}"
    try:
        r = requests.post(url_emb, json={"model": "models/text-embedding-004", "content": {"parts": [{"text": text}]}}, headers={"Content-Type": "application/json"})
        res = r.json()
        if "embedding" in res and "values" in res["embedding"]:
            return res["embedding"]["values"]
    except Exception:
        pass
    return None

def ziskej_kontext(dotaz, user_id):
    vektor = ziskej_embedding(dotaz)
    vektory_text = []
    
    if vektor:
        client = QdrantClient(path=DB_PATH)
        try:
            q_res = client.query_points(collection_name="zrcadlo_pamet", query=vektor, limit=10).points
            vektory_text = [hit.payload.get('text', '') for hit in q_res if hit.payload.get('user_id', 'karel') == user_id][:3]
        except Exception:
            vektory_text = []
        client.close()

    conn = sqlite3.connect(GRAPH_DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT subject, relation, object FROM triples WHERE user_id = ?", (user_id,))
    graf_res = [f"[{r[0]}] -> ({r[1]}) -> [{r[2]}]" for r in cursor.fetchall()]
    conn.close()

    return vektory_text, graf_res

def uc_se_z_zpravy(zprava, user_id):
    url_gen = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-3.1-flash-lite:generateContent?key={API_KEY}"
    prompt = f"""
Pokud zpráva obsahuje trvalé osobní fakta, preference, plánované akce nebo tělesné/psychické prožitky, vyextrahuj je.
Pokud zpráva neobsahuje žádná nová fakta, vrať prázdná pole.

Vrať JSON:
- facts: pole věcných tvrzení o uživateli (3. osoba)
- triples: pole objektů {{"subject": "...", "relation": "...", "object": "..."}} (max 3 slova na entitu)

Zpráva:
{zprava}
"""
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "responseMimeType": "application/json",
            "responseSchema": {
                "type": "OBJECT",
                "properties": {
                    "facts": {"type": "ARRAY", "items": {"type": "STRING"}},
                    "triples": {
                        "type": "ARRAY",
                        "items": {
                            "type": "OBJECT",
                            "properties": {
                                "subject": {"type": "STRING"},
                                "relation": {"type": "STRING"},
                                "object": {"type": "STRING"}
                            },
                            "required": ["subject", "relation", "object"]
                        }
                    }
                },
                "required": ["facts", "triples"]
            }
        }
    }
    try:
        r = requests.post(url_gen, json=payload, headers={"Content-Type": "application/json"})
        if r.status_code != 200:
            return

        data = json.loads(r.json()['candidates'][0]['content']['parts'][0]['text'])
        fakta = data.get("facts", [])
        trojice = data.get("triples", [])

        # 1. Uložení vektorů do Qdrant
        if fakta:
            client = QdrantClient(path=DB_PATH)
            for f in fakta:
                v = ziskej_embedding(f)
                if v:
                    client.upsert(
                        collection_name="zrcadlo_pamet",
                        points=[PointStruct(id=str(uuid.uuid4()), vector=v, payload={"text": f, "user_id": user_id})]
                    )
            client.close()

        # 2. Uložení grafu do SQLite
        if trojice:
            conn = sqlite3.connect(GRAPH_DB_PATH)
            cursor = conn.cursor()
            for t in trojice:
                sub, rel, obj = t.get('subject','').lower().strip(), t.get('relation','').lower().strip(), t.get('object','').lower().strip()
                if sub and rel and obj:
                    try:
                        cursor.execute("INSERT INTO triples (user_id, subject, relation, object) VALUES (?, ?, ?, ?)", (user_id, sub, rel, obj))
                    except sqlite3.IntegrityError:
                        pass
            conn.commit()
            conn.close()
    except Exception:
        pass

def generuj_odpoved(dotaz, vektory, graf, user_id):
    prompt = f"""
Jsi Zrcadlo, empatický AI průvodce uživatele '{user_id}'. Odpovídáš na základě jeho vzpomínek.

DOTAZ UŽIVATELE ({user_id}):
{dotaz}

VEKTOROVÉ VZPOMÍNKY:
{chr(10).join(vektory) if vektory else "Žádné předchozí vzpomínky."}

GRAFOVÉ VAZBY:
{chr(10).join(graf) if graf else "Žádné předchozí grafové vazby."}

Odpověz přátelsky, přímo a s pochopením.
"""
    url_gen = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-3.1-flash-lite:generateContent?key={API_KEY}"
    r = requests.post(url_gen, json={"contents": [{"parts": [{"text": prompt}]}]}, headers={"Content-Type": "application/json"})
    return r.json()['candidates'][0]['content']['parts'][0]['text']

# --- HLAVNÍ CHAT ROZHRANÍ ---
st.title("🪞 Zrcadlo")
st.subheader(f"Konverzace pro uživatele: :blue[{active_user}]")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

if user_input := st.chat_input(f"Napiš zprávu pro Zrcadlo..."):
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.write(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Zrcadlo přemýšlí a ukládá poznatky..."):
            vektory, graf = ziskej_kontext(user_input, active_user)
            odpoved = generuj_odpoved(user_input, vektory, graf, active_user)
            uc_se_z_zpravy(user_input, active_user)
            st.write(odpoved)

    st.session_state.messages.append({"role": "assistant", "content": odpoved})
    st.rerun()
