# -*- coding: utf-8 -*-
import json
import os
import sqlite3
import uuid
from datetime import datetime

import streamlit as st
from dotenv import load_dotenv
from google import genai
from google.genai import types
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

load_dotenv()

DB_PATH = "./qdrant_db"
GRAPH_DB_PATH = "znalostni_graf.db"
GEN_MODEL = "gemini-3.1-flash-lite"
EMBED_MODEL = "gemini-embedding-001"
QDRANT_COLLECTION = "zrcadlo_pamet"
EMBED_DIM = 768


def nacti_api_klic():
    """Načte API klíč ze st.secrets (Cloud) nebo z .env / prostředí (lokálně)."""
    try:
        if "API_KEY" in st.secrets:
            return st.secrets["API_KEY"]
    except Exception:
        pass

    klic = os.getenv("API_KEY")
    if klic:
        return klic

    raise ValueError(
        "API_KEY nebyl nalezen. Nastav ho v .streamlit/secrets.toml "
        "nebo v souboru .env (API_KEY=...)."
    )


def vytvor_qdrant_client() -> QdrantClient:
    """Připojení k Qdrant Cloudu ze secrets, jinak lokální ./qdrant_db."""
    try:
        if "QDRANT_URL" in st.secrets and "QDRANT_API_KEY" in st.secrets:
            return QdrantClient(
                url=st.secrets["QDRANT_URL"],
                api_key=st.secrets["QDRANT_API_KEY"],
            )
    except Exception:
        pass

    return QdrantClient(path=DB_PATH)


def zajisti_kolekci(client: QdrantClient) -> None:
    """Vytvoří kolekci zrcadlo_pamet, pokud ještě neexistuje."""
    existuje = False
    try:
        if hasattr(client, "collection_exists"):
            existuje = client.collection_exists(collection_name=QDRANT_COLLECTION)
        else:
            jmena = [c.name for c in client.get_collections().collections]
            existuje = QDRANT_COLLECTION in jmena
    except Exception:
        existuje = False

    if not existuje:
        client.create_collection(
            collection_name=QDRANT_COLLECTION,
            vectors_config=VectorParams(size=EMBED_DIM, distance=Distance.COSINE),
        )


def je_chyba_404(exc: Exception) -> bool:
    """True, pokud výjimka vypadá jako chybějící kolekce (404 / Not Found)."""
    kod = getattr(exc, "status_code", None) or getattr(exc, "code", None)
    text = str(exc).lower()
    return kod == 404 or "404" in text or "not found" in text or "doesn't exist" in text


def qdrant_operace(fn):
    """
    Spustí operaci nad Qdrantem: předem zajistí kolekci,
    při 404 ji znovu vytvoří a operaci jednou zopakuje.
    """
    client = vytvor_qdrant_client()
    try:
        zajisti_kolekci(client)
        try:
            return fn(client)
        except Exception as e:
            if je_chyba_404(e):
                zajisti_kolekci(client)
                return fn(client)
            raise
    finally:
        client.close()


def aktualni_razitko() -> str:
    """Aktuální datum a čas ve formátu DD.MM.YYYY HH:MM."""
    return datetime.now().strftime("%d.%m.%Y %H:%M")


def formatuj_historii_chatu(zpravy, limit=5) -> str:
    """Sestaví text posledních zpráv z aktuálního chatu."""
    if not zpravy:
        return "Žádná předchozí konverzace v této relaci."
    casti = []
    for msg in zpravy[-limit:]:
        role = "Uživatel" if msg.get("role") == "user" else "Zrcadlo"
        casti.append(f"{role}: {msg.get('content', '')}")
    return "\n".join(casti)


def formatuj_chybu(exc: Exception) -> str:
    """Sestaví čitelnou chybovou hlášku včetně kódu, pokud je dostupný."""
    kod = getattr(exc, "code", None) or getattr(exc, "status_code", None)
    if kod is None:
        details = getattr(exc, "details", None)
        if isinstance(details, dict):
            kod = details.get("code") or details.get("status")
    cast = f"{type(exc).__name__}: {exc}"
    if kod is not None:
        return f"Chyba API [{kod}]: {cast}"
    return f"Chyba API: {cast}"


st.set_page_config(page_title="Projekt Zrcadlo", page_icon="🪞", layout="wide")

try:
    API_KEY = nacti_api_klic()
    genai_client = genai.Client(api_key=API_KEY)
except Exception as e:
    st.error(formatuj_chybu(e))
    st.stop()

# --- DB INICIALIZACE A MIGRACE ---
conn = sqlite3.connect(GRAPH_DB_PATH)
cursor = conn.cursor()
cursor.execute("PRAGMA table_info(triples)")
columns = [col[1] for col in cursor.fetchall()]

if not columns:
    cursor.execute(
        """
        CREATE TABLE triples (
            user_id TEXT DEFAULT 'karel',
            subject TEXT, relation TEXT, object TEXT,
            UNIQUE(user_id, subject, relation, object)
        )
        """
    )
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

    try:
        def _nacti_body(client):
            return client.scroll(collection_name=QDRANT_COLLECTION, limit=200)[0]

        all_points = qdrant_operace(_nacti_body)
        user_points = [
            p for p in all_points if p.payload.get("user_id", "karel") == active_user
        ]
        pocet_vektoru = len(user_points)
    except Exception:
        pocet_vektoru = 0

    st.metric("Vektory uživatele", pocet_vektoru)

    conn = sqlite3.connect(GRAPH_DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        "SELECT subject, relation, object FROM triples WHERE user_id = ?",
        (active_user,),
    )
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
    try:
        vysledek = genai_client.models.embed_content(
            model=EMBED_MODEL,
            contents=text,
            config=types.EmbedContentConfig(output_dimensionality=EMBED_DIM),
        )
        return vysledek.embeddings[0].values
    except Exception as e:
        raise RuntimeError(formatuj_chybu(e)) from e


def ziskej_kontext(dotaz, user_id):
    try:
        vektor = ziskej_embedding(dotaz)
    except Exception as e:
        raise RuntimeError(formatuj_chybu(e)) from e

    try:
        def _hledej(client):
            return client.query_points(
                collection_name=QDRANT_COLLECTION,
                query=vektor,
                limit=10,
            ).points

        q_res = qdrant_operace(_hledej)
        vektory_text = [
            hit.payload.get("text", "")
            for hit in q_res
            if hit.payload.get("user_id", "karel") == user_id
        ][:3]
    except Exception:
        vektory_text = []

    conn = sqlite3.connect(GRAPH_DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        "SELECT subject, relation, object FROM triples WHERE user_id = ?",
        (user_id,),
    )
    graf_res = [f"[{r[0]}] -> ({r[1]}) -> [{r[2]}]" for r in cursor.fetchall()]
    conn.close()

    return vektory_text, graf_res


def uc_se_z_zpravy(zprava, user_id):
    """Extrahujeme fakta a graf na pozadí chatu a ukládáme nové vzpomínky."""
    prompt = f"""
Pokud zpráva obsahuje trvalé osobní fakta, preference, vztahy nebo tělesné/psychické prožitky, vyextrahuj je.
Pokud zpráva neobsahuje žádná nová fakta (jen pozdrav, dotaz apod.), vrať prázdná pole.

Vrať JSON:
- facts: pole věcných tvrzení o uživateli (3. osoba)
- triples: pole objektů {{"subject": "...", "relation": "...", "object": "..."}} (max 3 slova na entitu)

Zpráva:
{zprava}
"""
    schema = {
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
                        "object": {"type": "STRING"},
                    },
                    "required": ["subject", "relation", "object"],
                },
            },
        },
        "required": ["facts", "triples"],
    }

    try:
        odpoved = genai_client.models.generate_content(
            model=GEN_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=schema,
            ),
        )
        data = json.loads(odpoved.text)
    except Exception as e:
        raise RuntimeError(formatuj_chybu(e)) from e

    fakta = data.get("facts", [])
    trojice = data.get("triples", [])

    if fakta:
        razitko = aktualni_razitko()
        body = []
        for f in fakta:
            text_s_casem = f"[{razitko}] {f}"
            v = ziskej_embedding(text_s_casem)
            body.append(
                PointStruct(
                    id=str(uuid.uuid4()),
                    vector=v,
                    payload={"text": text_s_casem, "user_id": user_id},
                )
            )

        def _uloz(client):
            client.upsert(collection_name=QDRANT_COLLECTION, points=body)

        qdrant_operace(_uloz)

    if trojice:
        conn = sqlite3.connect(GRAPH_DB_PATH)
        cursor = conn.cursor()
        for t in trojice:
            sub = t.get("subject", "").lower().strip()
            rel = t.get("relation", "").lower().strip()
            obj = t.get("object", "").lower().strip()
            if sub and rel and obj:
                try:
                    cursor.execute(
                        "INSERT INTO triples (user_id, subject, relation, object) VALUES (?, ?, ?, ?)",
                        (user_id, sub, rel, obj),
                    )
                except sqlite3.IntegrityError:
                    pass
        conn.commit()
        conn.close()


def generuj_odpoved(dotaz, vektory, graf, user_id, historie_chatu):
    prompt = f"""
Jsi Zrcadlo, empatický AI průvodce uživatele '{user_id}'.

PRAVIDLA:
- Vycházej primárně ze zadaných faktů: vektorových vzpomínek, grafových vazeb a historie chatu.
- Nevymýšlej si nepodložené detaily, události, jména ani pocity, které v podkladech nejsou.
- Pokud něco nevíš, otevřeně to řekni. Nehalucinuj.
- Na začátku vzpomínek může být časové razítko [DD.MM.YYYY HH:MM] — ber ho v potaz.
- Odpovídej přátelsky, přímo a s pochopením a plynule navazuj na krátkodobou historii chatu.

DOTAZ UŽIVATELE ({user_id}):
{dotaz}

KRÁTKODOBÁ PAMĚŤ (poslední zprávy tohoto chatu):
{formatuj_historii_chatu(historie_chatu, limit=5)}

VEKTOROVÉ VZPOMÍNKY:
{chr(10).join(vektory) if vektory else "Žádné předchozí vzpomínky."}

GRAFOVÉ VAZBY:
{chr(10).join(graf) if graf else "Žádné předchozí grafové vazby."}
"""
    try:
        odpoved = genai_client.models.generate_content(
            model=GEN_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(temperature=0.2),
        )
        return odpoved.text
    except Exception as e:
        raise RuntimeError(formatuj_chybu(e)) from e


# --- HLAVNÍ CHAT ROZHRANÍ ---
st.title("🪞 Zrcadlo")
st.subheader(f"Konverzace pro uživatele: :blue[{active_user}]")

if "chat_historie" not in st.session_state:
    st.session_state.chat_historie = {}
if active_user not in st.session_state.chat_historie:
    st.session_state.chat_historie[active_user] = []

zpravy = st.session_state.chat_historie[active_user]

for msg in zpravy:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

if user_input := st.chat_input("Napiš zprávu pro Zrcadlo..."):
    zpravy.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.write(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Zrcadlo přemýšlí a ukládá poznatky..."):
            try:
                vektory, graf = ziskej_kontext(user_input, active_user)
                historie_pro_prompt = zpravy[:-1]
                odpoved = generuj_odpoved(
                    user_input,
                    vektory,
                    graf,
                    active_user,
                    historie_pro_prompt,
                )
                try:
                    uc_se_z_zpravy(user_input, active_user)
                except Exception as e:
                    uceni_chyba = formatuj_chybu(e)
                    odpoved = f"{odpoved}\n\n⚠️ Učení z paměti selhalo: {uceni_chyba}"
                st.write(odpoved)
            except Exception as e:
                odpoved = formatuj_chybu(e)
                st.error(odpoved)

    zpravy.append({"role": "assistant", "content": odpoved})
    st.rerun()
