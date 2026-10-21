# -*- coding: utf-8 -*-
"""Práce s Qdrantem, SQLite grafem a pamětí uživatele."""

from __future__ import annotations

from dotenv import load_dotenv
import os
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(_ROOT / ".env")
load_dotenv()

import sqlite3
import threading
import time
import uuid
from datetime import datetime
from typing import Any, Callable, Optional

from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.models import Distance, PointStruct, VectorParams

ROOT = _ROOT
DB_PATH = str(ROOT / "qdrant_db")
GRAPH_DB_PATH = str(ROOT / "znalostni_graf.db")
QDRANT_COLLECTION = "zrcadlo_pamet"
# Ověřeno test_step_by_step: gemini-embedding-001 → dim 3072
EMBED_MODEL = "gemini-embedding-001"
EMBED_DIM = 3072

DB_MAX_POKUSU = 3
DB_CEKANI_S = 2

# Jeden sdílený klient — lokální path mode nesnese více instancí najednou.
_qdrant_client: Optional[QdrantClient] = None
_qdrant_client_lock = threading.Lock()


def vytvor_qdrant_client() -> QdrantClient:
    """
    Připojení k Qdrantu:
    - Pokud je nastaveno QDRANT_URL → cloud/remote (api_key z QDRANT_API_KEY, může být prázdný).
    - Jinak lokální diskové úložiště (./qdrant_db) — lokální běh beze změny.
    """
    raw_url = os.getenv("QDRANT_URL") or ""
    raw_key = os.getenv("QDRANT_API_KEY") or ""
    url = raw_url.strip().strip("'\"")
    api_key = raw_key.strip().strip("'\"")

    # Placeholder hodnoty z šablon nepovažuj za platné připojení
    placeholders = {
        "",
        "tvoje_qdrant_url",
        "tvuj_qdrant_api_klic",
        "your_qdrant_url",
        "your_qdrant_api_key",
    }

    if url and url not in placeholders:
        print(f"[Qdrant] Remote připojení: {url}")
        if api_key and api_key not in placeholders:
            return QdrantClient(url=url, api_key=api_key)
        return QdrantClient(url=url)

    print(f"[Qdrant] Lokální úložiště: {DB_PATH}")
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
            vectors_config=VectorParams(size=3072, distance=Distance.COSINE),
        )
        print(f"[Qdrant] Kolekce '{QDRANT_COLLECTION}' vytvořena (dim=3072, COSINE).")


def ensure_collection_exists(
    client: QdrantClient, collection_name: str = QDRANT_COLLECTION
) -> None:
    """Alias pro zajisti_kolekci."""
    zajisti_kolekci(client)


def _je_prechodna_db_chyba(exc: Exception) -> bool:
    text = str(exc).lower()
    nazev = type(exc).__name__.lower()
    kod = getattr(exc, "code", None) or getattr(exc, "status_code", None)
    return (
        kod in (429, 503)
        or "429" in text
        or "503" in text
        or "resourceexhausted" in nazev
        or "resource_exhausted" in text
        or "rate limit" in text
        or "quota" in text
        or "timeout" in text
        or "unavailable" in text
    )


def s_opakovanim(fn: Callable, popisek: str = "operace"):
    """Až 3 pokusy s pauzou 2 s při přechodných chybách."""
    posledni = None
    for pokus in range(1, DB_MAX_POKUSU + 1):
        try:
            return fn()
        except Exception as e:
            posledni = e
            if _je_prechodna_db_chyba(e) and pokus < DB_MAX_POKUSU:
                print(
                    f"[DB] {popisek}: přechodná chyba "
                    f"(pokus {pokus}/{DB_MAX_POKUSU}), čekám {DB_CEKANI_S}s"
                )
                time.sleep(DB_CEKANI_S)
                continue
            raise
    raise posledni  # type: ignore[misc]


def je_chyba_404(exc: Exception) -> bool:
    kod = getattr(exc, "status_code", None) or getattr(exc, "code", None)
    text = str(exc).lower()
    return kod == 404 or "404" in text or "not found" in text or "doesn't exist" in text


def get_qdrant_client() -> QdrantClient:
    """Vrátí sdílenou instanci QdrantClient (bez opakovaného open/close path DB)."""
    global _qdrant_client
    with _qdrant_client_lock:
        if _qdrant_client is None:
            _qdrant_client = vytvor_qdrant_client()
        return _qdrant_client


def qdrant_operace(fn: Callable[[QdrantClient], Any]) -> Any:
    """Spustí operaci nad Qdrantem; při 404 znovu vytvoří kolekci a zopakuje."""
    client = get_qdrant_client()
    zajisti_kolekci(client)
    try:
        return fn(client)
    except Exception as e:
        if je_chyba_404(e):
            zajisti_kolekci(client)
            return fn(client)
        raise


def init_graph_db() -> None:
    """Inicializace / migrace SQLite znalostního grafu."""
    conn = sqlite3.connect(GRAPH_DB_PATH)
    cursor = conn.cursor()
    cursor.execute("PRAGMA table_info(triples)")
    columns = [col[1] for col in cursor.fetchall()]

    if not columns:
        cursor.execute(
            """
            CREATE TABLE triples (
                user_id TEXT DEFAULT '',
                subject TEXT, relation TEXT, object TEXT,
                UNIQUE(user_id, subject, relation, object)
            )
            """
        )
    elif "user_id" not in columns:
        cursor.execute("ALTER TABLE triples ADD COLUMN user_id TEXT DEFAULT ''")

    conn.commit()
    conn.close()


def aktualni_razitko() -> str:
    return datetime.now().strftime("%d.%m.%Y %H:%M")


def rozdel_razitko_a_text(text: str) -> tuple[str, str]:
    if text.startswith("[") and "]" in text:
        konec = text.find("]")
        return text[1:konec], text[konec + 1 :].lstrip()
    return "—", text


def _scroll_filter_user_id(user_id: str) -> Optional[models.Filter]:
    """Stejný filtr user_id jako u chatového hledej_vzpominky."""
    uid = (user_id or "").strip().lower()
    if not uid:
        return None
    return models.Filter(
        must=[
            models.FieldCondition(
                key="user_id",
                match=models.MatchValue(value=uid),
            )
        ]
    )


def _scroll_vsechny_body(
    client: QdrantClient, user_id: str = ""
) -> list:
    """
    Načte body z kolekce přes stránkovaný scroll.
    Pokud je user_id, filtruje stejně jako chat (payload.user_id).
    """
    body = []
    offset = None
    scroll_filter = _scroll_filter_user_id(user_id)
    while True:
        points, next_offset = client.scroll(
            collection_name=QDRANT_COLLECTION,
            scroll_filter=scroll_filter,
            limit=100,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        body.extend(points)
        if next_offset is None:
            break
        offset = next_offset
    return body


def _payload_user_id(payload: dict) -> str:
    if not payload:
        return ""
    val = payload.get("user_id", payload.get("user", ""))
    return str(val).strip().lower() if val is not None else ""


def _payload_text(payload: dict) -> str:
    """Vytáhne textový obsah z běžných klíčů payloadu."""
    if not payload:
        return ""
    for key in (
        "text",
        "content",
        "memory",
        "message",
        "msg",
        "body",
        "value",
        "fact",
    ):
        val = payload.get(key)
        if val is not None and str(val).strip():
            return str(val)
    # Fallback: první stringová hodnota v payloadu
    for val in payload.values():
        if isinstance(val, str) and val.strip():
            return val
    return ""


def _payload_timestamp(payload: dict, text_val: str) -> str:
    if payload:
        ts = payload.get("timestamp") or payload.get("created_at") or payload.get("date")
        if ts:
            return str(ts)
    razitko, _ = rozdel_razitko_a_text(text_val)
    if razitko and razitko != "—":
        return razitko
    return "Neznámé datum"


def vypis_kolekce_qdrant(client: QdrantClient) -> list[str]:
    """Vypíše a vrátí jména všech kolekcí v Qdrantu."""
    try:
        kolekce = [c.name for c in client.get_collections().collections]
    except Exception as e:
        print(f"[Qdrant] Nepodařilo se načíst kolekce: {e}")
        kolekce = []
    print(f"[Qdrant] Dostupné kolekce ({len(kolekce)}): {kolekce}")
    return kolekce


def nacti_vzpominky_uzivatele(user_id: str) -> list[dict]:
    """
    Načte vektorové vzpomínky z Qdrantu pro dané user_id.
    Stejný zdroj, ze kterého chat bere uvítací přehled a (přes search) kontext.
    """
    uid = (user_id or "").strip().lower()
    if not uid:
        return []

    def _scroll_all(client: QdrantClient):
        points = _scroll_vsechny_body(client, user_id=uid)
        # Pojistka pro starší body / odlišný casing v payloadu
        points = [
            p for p in points if _payload_user_id(p.payload or {}) == uid
        ]
        zaznamy = []
        for p in points:
            payload = p.payload or {}
            text_val = _payload_text(payload)
            if not text_val:
                continue
            zaznamy.append(
                {
                    "id": str(p.id),
                    "text": text_val,
                    "timestamp": _payload_timestamp(payload, text_val),
                    "user_id": _payload_user_id(payload) or uid,
                }
            )
        return zaznamy

    try:
        return qdrant_operace(_scroll_all)
    except Exception as e:
        print(f"[Qdrant] nacti_vzpominky_uzivatele selhalo: {e}")
        return []


def smaz_grafove_vazby_uzivatele(user_id: str) -> int:
    """Smaže všechny triples daného user_id ze SQLite grafu. Vrátí počet smazaných řádků."""
    uid = (user_id or "").strip().lower()
    if not uid:
        return 0
    conn = sqlite3.connect(GRAPH_DB_PATH)
    cursor = conn.cursor()
    cursor.execute("DELETE FROM triples WHERE user_id = ?", (uid,))
    smazano = cursor.rowcount
    conn.commit()
    conn.close()
    print(f"[Graf] Smazáno {smazano} vazeb pro user_id={uid!r}")
    return smazano


def smaz_grafove_vazby_pro_text(user_id: str, text: str) -> int:
    """
    Smaže triples uživatele, jejichž subject/object se vyskytuje v textu vzpomínky.
    Pokud text chybí, smaže celý graf uživatele.
    """
    uid = (user_id or "").strip().lower()
    if not uid:
        return 0
    raw = (text or "").lower()
    if not raw.strip():
        return smaz_grafove_vazby_uzivatele(uid)

    conn = sqlite3.connect(GRAPH_DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        "SELECT rowid, subject, relation, object FROM triples WHERE user_id = ?",
        (uid,),
    )
    radky = cursor.fetchall()
    smazane_ids = []
    for rowid, sub, rel, obj in radky:
        tokeny = [t for t in (sub, obj) if t and len(str(t).strip()) >= 3]
        if any(str(t).lower() in raw for t in tokeny):
            smazane_ids.append(rowid)
    if smazane_ids:
        cursor.executemany(
            "DELETE FROM triples WHERE rowid = ?",
            [(rid,) for rid in smazane_ids],
        )
    conn.commit()
    conn.close()
    print(
        f"[Graf] Smazáno {len(smazane_ids)} souvisejících vazeb "
        f"pro user_id={uid!r} (textové mazání)"
    )
    return len(smazane_ids)


def smaz_vzpominku(point_id, user_id: str = "") -> None:
    """
    Smaže vektor z Qdrantu (nebo řádek grafu při id graph-<rowid>)
    a vyčistí související vazby ve znalostním grafu (SQLite).
    """
    pid = str(point_id)
    if pid.startswith("graph-"):
        try:
            rowid = int(pid.split("-", 1)[1])
        except ValueError as e:
            raise ValueError(f"Neplatné graph id: {pid}") from e
        conn = sqlite3.connect(GRAPH_DB_PATH)
        cursor = conn.cursor()
        cursor.execute("DELETE FROM triples WHERE rowid = ?", (rowid,))
        smazano = cursor.rowcount
        conn.commit()
        conn.close()
        print(f"[Graf] Smazán řádek rowid={rowid} (smazano={smazano})")
        return

    nacteny_text = ""
    nacteny_user = (user_id or "").strip().lower()

    def _retrieve_and_delete(client: QdrantClient):
        nonlocal nacteny_text, nacteny_user
        try:
            body = client.retrieve(
                collection_name=QDRANT_COLLECTION,
                ids=[point_id],
                with_payload=True,
                with_vectors=False,
            )
            if body:
                payload = body[0].payload or {}
                nacteny_text = _payload_text(payload)
                nacteny_user = _payload_user_id(payload) or nacteny_user
        except Exception as e:
            print(f"[Qdrant] retrieve před smazáním selhalo: {e}")

        client.delete(
            collection_name=QDRANT_COLLECTION,
            points_selector=models.PointIdsList(points=[point_id]),
        )

    qdrant_operace(_retrieve_and_delete)

    if nacteny_user:
        smaz_grafove_vazby_pro_text(nacteny_user, nacteny_text)
    else:
        print("[Graf] user_id neznámý — graf nepročištěn")



def _format_vzpominka_pro_frontend(zaznam: dict) -> dict:
    """Formát {id, text, timestamp} očekávaný MemoryTab."""
    text_val = zaznam.get("text", "") or ""
    razitko, text_bez = rozdel_razitko_a_text(text_val)
    display_text = text_bez if razitko != "—" else text_val
    return {
        "id": str(zaznam.get("id", "")),
        "text": display_text,
        "timestamp": zaznam.get("timestamp")
        or _payload_timestamp({}, text_val)
        or "Neznámé datum",
    }


def nacti_graf_jako_vzpominky(user_id: str) -> list[dict]:
    """
    Grafové vazby ze SQLite ve formátu paměti pro frontend.
    Chat je čte přes nacti_graf — sem je dáme, aby Správa paměti viděla totéž.
    """
    uid = (user_id or "").strip().lower()
    if not uid:
        return []
    conn = sqlite3.connect(GRAPH_DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        "SELECT rowid, subject, relation, object FROM triples WHERE user_id = ?",
        (uid,),
    )
    radky = cursor.fetchall()
    conn.close()
    return [
        {
            "id": f"graph-{rowid}",
            "text": f"[{sub}] -> ({rel}) -> [{obj}]",
            "timestamp": "Graf",
        }
        for rowid, sub, rel, obj in radky
    ]


def seznam_vzpominek_api(user_id: str = "") -> list[dict]:
    """
    GET /api/memories — stejné zdroje jako chat:
    1) Qdrant body pro user_id (vektorové vzpomínky)
    2) SQLite triples (grafové vazby)
    Vrací [{id, text, timestamp}] pro frontend.
    """
    uid = (user_id or "").strip().lower()
    if not uid:
        return []

    print(
        f"[Paměť] GET memories (Qdrant+graf) kolekce={QDRANT_COLLECTION!r} "
        f"user_id={uid!r}"
    )
    zaznamy = nacti_vzpominky_uzivatele(uid)
    vysledek = [_format_vzpominka_pro_frontend(z) for z in zaznamy]
    grafove = nacti_graf_jako_vzpominky(uid)
    vysledek.extend(grafove)
    print(
        f"[Paměť] Memories pro frontend: {len(vysledek)} "
        f"(qdrant={len(zaznamy)}, graf={len(grafove)})"
    )
    return vysledek


def export_vzpominek(user_id: str) -> dict:
    """JSON kontejner se všemi vzpomínkami a metadaty uživatele."""
    uid = (user_id or "").strip().lower()
    memories = seznam_vzpominek_api(uid)
    return {
        "user_id": uid,
        "exported_at": datetime.now().isoformat(timespec="seconds"),
        "count": len(memories),
        "memories": memories,
    }


def uloz_vektory(body: list) -> None:
    """Uloží vektory; při rate-limitu / chybě jen zaloguje (chat nesmí spadnout)."""
    if not body:
        return

    def _uloz(client: QdrantClient):
        ensure_collection_exists(client, QDRANT_COLLECTION)
        client.upsert(collection_name=QDRANT_COLLECTION, points=body)

    try:
        s_opakovanim(lambda: qdrant_operace(_uloz), popisek="uložení vektorů")
    except Exception as e:
        print(f"[Qdrant] Uložení vzpomínek selhalo (soft): {e}")


def hledej_vzpominky(vektor: list[float], user_id: str, limit: int = 10) -> list[str]:
    """Sémantické vyhledávání vzpomínek pro user_id. Při chybě vrátí []."""

    def _hledej(client: QdrantClient):
        return client.query_points(
            collection_name=QDRANT_COLLECTION,
            query=vektor,
            limit=limit,
            query_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="user_id",
                        match=models.MatchValue(value=user_id),
                    )
                ]
            ),
        ).points

    try:
        q_res = s_opakovanim(
            lambda: qdrant_operace(_hledej),
            popisek="vyhledávání vzpomínek",
        )
        return [hit.payload.get("text", "") for hit in q_res if hit.payload]
    except Exception as e:
        print(f"[Qdrant] Vyhledávání selhalo (soft): {e}")
        return []


def nacti_graf(user_id: str) -> list[str]:
    conn = sqlite3.connect(GRAPH_DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        "SELECT subject, relation, object FROM triples WHERE user_id = ?",
        (user_id,),
    )
    graf_res = [f"[{r[0]}] -> ({r[1]}) -> [{r[2]}]" for r in cursor.fetchall()]
    conn.close()
    return graf_res


def uloz_trojice(user_id: str, trojice: list[dict]) -> None:
    if not trojice:
        return
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


def posledni_vzpominky_texty(user_id: str, limit: int = 10) -> list[str]:
    vse = nacti_vzpominky_uzivatele(user_id)

    def _sort_key(zaznam):
        razitko, _ = rozdel_razitko_a_text(zaznam.get("text", ""))
        try:
            return datetime.strptime(razitko, "%d.%m.%Y %H:%M")
        except Exception:
            return datetime.min

    serazene = sorted(vse, key=_sort_key)
    return [z.get("text", "") for z in serazene[-limit:] if z.get("text")]


def vytvor_point(vektor: list[float], text: str, user_id: str):
    return models.PointStruct(
        id=str(uuid.uuid4()),
        vector=vektor,
        payload={"text": text, "user_id": user_id},
    )
