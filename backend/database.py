# -*- coding: utf-8 -*-
"""Práce s Qdrantem, SQLite grafem a pamětí uživatele."""

from __future__ import annotations

from dotenv import load_dotenv
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env", override=True)
load_dotenv()

import sqlite3
import time
import uuid
from datetime import datetime
from typing import Any, Callable

from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.models import Distance, VectorParams

try:
    from pydantic import ValidationError
except ImportError:  # pragma: no cover
    ValidationError = Exception  # type: ignore[misc, assignment]

DB_PATH = str(ROOT / "qdrant_db")
GRAPH_DB_PATH = str(ROOT / "znalostni_graf.db")
QDRANT_COLLECTION = "zrcadlo_pamet"
# Ověřeno test_step_by_step / list_models: embedContent → gemini-embedding-001, dim 3072
EMBED_MODEL = "gemini-embedding-001"
EMBED_DIM = 3072


def nacti_gemini_api_klic() -> str:
    """Stejný klíč jako ai_engine — GEMINI_API_KEY z .env (pro konzistenci)."""
    raw = os.getenv("GEMINI_API_KEY") or os.getenv("API_KEY") or ""
    key = raw.strip().strip("'\"")
    if not key:
        raise ValueError("GEMINI_API_KEY chybí v .env")
    return key

DB_MAX_POKUSU = 3
DB_CEKANI_S = 2


def vytvor_qdrant_client() -> QdrantClient:
    """Připojení k Qdrant Cloudu z env, jinak lokální úložiště."""
    url = os.getenv("QDRANT_URL")
    api_key = os.getenv("QDRANT_API_KEY")
    placeholder = {"tvoje_qdrant_url", "tvuj_qdrant_api_klic", ""}
    if url and api_key and url not in placeholder and api_key not in placeholder:
        return QdrantClient(url=url, api_key=api_key)
    return QdrantClient(path=DB_PATH)


def ensure_collection_exists(
    client: QdrantClient,
    collection_name: str = QDRANT_COLLECTION,
) -> None:
    """
    Bezpečně zajistí existenci kolekce s VectorParams(size=EMBED_DIM, COSINE).
    Nikdy nemaže ani nerecreatuje existující kolekci (ochrana lokálních dat).
    """
    existuje = False
    try:
        if hasattr(client, "collection_exists"):
            existuje = bool(client.collection_exists(collection_name=collection_name))
        else:
            jmena = [c.name for c in client.get_collections().collections]
            existuje = collection_name in jmena
    except Exception as e:
        print(f"[Qdrant] Kontrola existence kolekce '{collection_name}' selhala: {e}")
        existuje = False

    if existuje:
        print(
            f"[Qdrant] Kolekce '{collection_name}' už existuje "
            f"(očekávaná dimenze EMBED_DIM={EMBED_DIM}, COSINE) — nepřetvářím."
        )
        return

    try:
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=3072, distance=Distance.COSINE),
        )
        print(
            f"[Qdrant] Vytvořena kolekce '{collection_name}' "
            f"(dim=3072, distance=COSINE)."
        )
    except ValidationError as e:
        print(
            f"[Qdrant] ValidationError při create_collection('{collection_name}'): {e} "
            "— kolekci nepřetvářím."
        )
    except Exception as e:
        text = str(e).lower()
        if "already" in text or "exists" in text or "409" in text:
            print(
                f"[Qdrant] Kolekce '{collection_name}' už existuje (create race) — OK."
            )
        else:
            print(f"[Qdrant] create_collection('{collection_name}') selhalo: {e}")


def zajisti_kolekci(client: QdrantClient) -> None:
    """Alias / tenký obal nad ensure_collection_exists pro výchozí kolekci."""
    ensure_collection_exists(client, QDRANT_COLLECTION)


def _je_prechodna_db_chyba(exc: Exception) -> bool:
    """True, pokud výjimka vypadá jako dočasná (retry dává smysl)."""
    text = str(exc).lower()
    klicova = (
        "timeout",
        "timed out",
        "connection",
        "temporarily",
        "unavailable",
        "rate limit",
        "too many requests",
        "429",
        "502",
        "503",
        "504",
        "locked",
        "busy",
        "retry",
        "broken pipe",
        "reset by peer",
        "resource temporarily",
        "deadline",
    )
    return any(k in text for k in klicova)


def s_opakovanim(fn: Callable[[], Any], popisek: str = "operace") -> Any:
    """Spustí fn až DB_MAX_POKUSU×; při přechodné chybě čeká DB_CEKANI_S sekund."""
    posledni: Exception | None = None
    for pokus in range(1, DB_MAX_POKUSU + 1):
        try:
            return fn()
        except Exception as e:
            posledni = e
            if not _je_prechodna_db_chyba(e) or pokus >= DB_MAX_POKUSU:
                raise
            print(
                f"[DB] {popisek}: přechodná chyba "
                f"(pokus {pokus}/{DB_MAX_POKUSU}): {e}; "
                f"čekám {DB_CEKANI_S}s…"
            )
            time.sleep(DB_CEKANI_S)
    assert posledni is not None
    raise posledni


def je_chyba_404(exc: Exception) -> bool:
    """True, pokud výjimka vypadá jako chybějící kolekce (404 / Not Found)."""
    kod = getattr(exc, "status_code", None) or getattr(exc, "code", None)
    text = str(exc).lower()
    return (
        kod == 404
        or "404" in text
        or "not found" in text
        or "doesn't exist" in text
        or "does not exist" in text
    )


def qdrant_operace(fn: Callable[[QdrantClient], Any]) -> Any:
    """Spustí operaci nad Qdrantem; při 404 znovu zajistí kolekci a zopakuje."""
    client = vytvor_qdrant_client()
    try:
        ensure_collection_exists(client, QDRANT_COLLECTION)
        try:
            return fn(client)
        except Exception as e:
            if je_chyba_404(e):
                ensure_collection_exists(client, QDRANT_COLLECTION)
                return fn(client)
            raise
    finally:
        client.close()


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


def _scroll_vsechny_body(client: QdrantClient) -> list:
    """Načte všechny body z kolekce přes stránkovaný scroll."""
    body = []
    offset = None
    while True:
        points, next_offset = client.scroll(
            collection_name=QDRANT_COLLECTION,
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
    """Načte paměťové záznamy POUZE pro dané user_id (bez fallbacku na všechny body)."""

    def _scroll_all(client: QdrantClient):
        points = _scroll_vsechny_body(client)
        uid = (user_id or "").strip().lower()
        if uid:
            points = [
                p for p in points if _payload_user_id(p.payload or {}) == uid
            ]
        else:
            points = []
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
    except Exception:
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
    Smaže vektor z Qdrantu a vyčistí související vazby ve znalostním grafu (SQLite).
    """
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


def seznam_vzpominek_api(user_id: str = "") -> list[dict]:
    """
    GET /api/memories — načte VŠECHNY body bez filtru user_id,
    vypíše debug info do konzole a vrátí [{id, text, timestamp}].
    """

    def _nacti_vse(client: QdrantClient):
        vypis_kolekce_qdrant(client)
        print(
            f"[Qdrant] Scroll všech bodů z kolekce '{QDRANT_COLLECTION}' "
            f"(bez filtru user_id; požadavek user_id={user_id!r})"
        )
        points = _scroll_vsechny_body(client)
        print(f"[Qdrant] Načteno bodů: {len(points)}")

        vysledek = []
        for p in points:
            payload = p.payload or {}
            print(f"Point {p.id}: {payload}")
            text_val = _payload_text(payload)
            if not text_val:
                continue
            razitko, text_bez = rozdel_razitko_a_text(text_val)
            timestamp = _payload_timestamp(payload, text_val)
            display_text = text_bez if razitko != "—" else text_val
            vysledek.append(
                {
                    "id": str(p.id),
                    "text": display_text,
                    "timestamp": timestamp,
                }
            )
        print(f"[Qdrant] Bodů s textovým obsahem pro frontend: {len(vysledek)}")
        return vysledek

    try:
        return qdrant_operace(_nacti_vse)
    except Exception as e:
        print(f"[Qdrant] Chyba při načítání vzpomínek: {e}")
        return []


def export_vzpominek(user_id: str) -> dict:
    """JSON kontejner se všemi vzpomínkami a metadaty uživatele."""
    memories = seznam_vzpominek_api(user_id)
    return {
        "user_id": user_id,
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
