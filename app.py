import os
import streamlit as st
import google.generativeai as genai
from qdrant_client import QdrantClient
from qdrant_client.http import models

# Konfigurace aplikace Streamlit
st.set_page_config(page_title="Zrcadlo — AI Paměť", page_icon="🪞", layout="wide")

# Načtení API klíče z st.secrets (Streamlit Cloud) nebo os.getenv (lokálně)
API_KEY = st.secrets.get("GEMINI_API_KEY") or os.getenv("GEMINI_API_KEY")
if API_KEY:
    genai.configure(api_key=API_KEY)

EMBED_MODEL = "gemini-embedding-001"
EMBED_DIM = 3072
COLLECTION_NAME = "zrcadlo_pamet"
PRIMARY_MODEL = "gemini-3.5-flash-lite"
FALLBACK_MODEL = "gemini-3.5-flash"

# Inicializace databáze v paměti (:memory:) pro bezproblémové spuštění na Streamlit Cloud
@st.cache_resource
def get_qdrant_client():
    client = QdrantClient(":memory:")
    collections = [c.name for c in client.get_collections().collections]
    if COLLECTION_NAME not in collections:
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=models.VectorParams(size=EMBED_DIM, distance=models.Distance.COSINE)
        )
    return client

qdrant = get_qdrant_client()

def get_embedding(text: str):
    try:
        res = genai.embed_content(
            model=EMBED_MODEL,
            content=text,
            task_type="retrieval_document"
        )
        return res["embedding"]
    except Exception:
        return None

def store_memory(user_id: str, text: str):
    vector = get_embedding(text)
    if not vector:
        return False
    point_id = abs(hash(f"{user_id}_{text}")) % (10 ** 12)
    qdrant.upsert(
        collection_name=COLLECTION_NAME,
        points=[
            models.PointStruct(
                id=point_id,
                vector=vector,
                payload={"user_id": user_id, "text": text}
            )
        ]
    )
    return True

def search_memories(user_id: str, query: str, limit: int = 3):
    vector = get_embedding(query)
    if not vector:
        return []
    try:
        results = qdrant.search(
            collection_name=COLLECTION_NAME,
            query_vector=vector,
            query_filter=models.Filter(
                must=[models.FieldCondition(key="user_id", match=models.MatchValue(value=user_id))]
            ),
            limit=limit
        )
        return [hit.payload["text"] for hit in results]
    except Exception:
        return []

def get_all_memories(user_id: str):
    try:
        results = qdrant.scroll(
            collection_name=COLLECTION_NAME,
            scroll_filter=models.Filter(
                must=[models.FieldCondition(key="user_id", match=models.MatchValue(value=user_id))]
            ),
            limit=100
        )[0]
        return [hit.payload["text"] for hit in results]
    except Exception:
        return []

# Uživatelské rozhraní
st.title("🪞 Zrcadlo — AI Paměť")

st.sidebar.header("Nastavení")
user_id = st.sidebar.text_input("USER ID", value="karel")
persona = st.sidebar.radio("Režim odpovídání", ["Kreativní parťák", "Striktní Trezor"])

tab1, tab2 = st.tabs(["💬 Chat", "🧠 Správa paměti"])

with tab1:
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    if prompt := st.chat_input("Napiš Zrcadlu..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Zrcadlo přemýšlí..."):
                memories = search_memories(user_id, prompt)
                context_str = "\n".join([f"- {m}" for m in memories]) if memories else "Žádné předchozí vzpomínky."

                system_instruction = f"""Jsi Zrcadlo, empatický AI průvodce uživatele '{user_id}'.
Styl komunikace: {persona}.
Zde jsou známá fakta o uživateli z trvalé paměti:
{context_str}

Odpovídej přirozeně, lidsky a využívej znalosti z paměti, pokud se hodí."""

                reply_text = None
                for m_name in [PRIMARY_MODEL, FALLBACK_MODEL]:
                    try:
                        model = genai.GenerativeModel(model_name=m_name, system_instruction=system_instruction)
                        res = model.generate_content(prompt)
                        reply_text = res.text
                        break
                    except Exception:
                        continue

                if not reply_text:
                    reply_text = "Omlouvám se, služba Gemini je momentálně přetížená. Zkus to prosím za okamžik."

                st.write(reply_text)
                st.session_state.messages.append({"role": "assistant", "content": reply_text})

                # Extrakce paměti (při chybě API přeskočí, aniž by shodila chat)
                try:
                    extractor_prompt = f"Z následující zprávy uživatele extrahuj pouze nová trvalá fakta o něm (např. koníčky, práce, preference). Pokud žádná nová fakta nejsou, napiš 'NIC'. Zpráva: '{prompt}'"
                    extractor_model = genai.GenerativeModel(PRIMARY_MODEL)
                    extracted = extractor_model.generate_content(extractor_prompt).text.strip()
                    if extracted and "NIC" not in extracted.upper():
                        store_memory(user_id, extracted)
                except Exception:
                    pass

with tab2:
    st.header(f"Vzpomínky uživatele: {user_id}")
    if st.button("Obnovit paměť"):
        st.rerun()
    
    user_mems = get_all_memories(user_id)
    if user_mems:
        for m in user_mems:
            st.info(f"📌 {m}")
    else:
        st.write("Pro tohoto uživatele zatím nejsou žádné uložené vzpomínky.")
