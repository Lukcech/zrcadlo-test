# -*- coding: utf-8 -*-
"""Práce s Qdrantem (vektory + graf) — produkční paměť žije v Qdrant Cloud."""

from __future__ import annotations

from dotenv import load_dotenv
import os
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(_ROOT / ".env")
load_dotenv()

import threading
import time
import uuid
from datetime import datetime
from typing import Any, Callable, Optional

from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.models import Distance, VectorParams

ROOT = _ROOT
DB_PATH = str(ROOT / "qdrant_db")
QDRANT_COLLECTION = "zrcadlo_pamet"
# Ověřeno test_step_by_step: gemini-embedding-001 → dim 3072
EMBED_MODEL = "gemini-embedding-001"
EMBED_DIM = 3072

MEMORY_TYPE_VECTOR = "vector"
MEMORY_TYPE_GRAPH = "graph"

DB_MAX_POKUSU = 3
DB_CEKANI_S = 2

# Jeden sdílený klient — lokální path mode nesnese více instancí najednou.
_qdrant_client: Optional[QdrantClient] = None
_qdrant_client_lock = threading.Lock()
_qdrant_mode: Optional[str] = None  # "cloud" | "local"


def _env_qdrant() -> tuple[str, str]:
    raw_url = os.getenv("QDRANT_URL") or ""
    raw_key = os.getenv("QDRANT_API_KEY") or ""
    url = raw_url.strip().strip("'\"")
    api_key = raw_key.strip().strip("'\"")
    placeholders = {
        "",
        "tvoje_qdrant_url",
        "tvuj_qdrant_api_klic",
        "your_qdrant_url",
        "your_qdrant_api_key",
    }
    if url in placeholders:
        url = ""
    if api_key in placeholders:
        api_key = ""
    return url, api_key


def normalizuj_user_id(user_id: str) -> str:
    return (user_id or "").strip().lower()


def vytvor_qdrant_client() -> QdrantClient:
    """
    Připojení k Qdrantu:
    - QDRANT_URL nastaveno → VÝHRADNĚ cloud/remote (bez fallbacku na ./qdrant_db).
    - Jinak lokální diskové úložiště (./qdrant_db) pro vývoj.
    """
    global _qdrant_mode
    url, api_key = _env_qdrant()

    if url:
        print(f"[Qdrant] Cloud/remote připojení: {url}")
        try:
            if api_key:
                client = QdrantClient(url=url, api_key=api_key)
            else:
                print("[Qdrant] Varování: QDRANT_API_KEY prázdný — zkouším bez klíče")
                client = QdrantClient(url=url)
            # Ověření spojení
            kolekce = [c.name for c in client.get_collections().collections]
            print(f"[Qdrant] Cloud OK — kolekce ({len(kolekce)}): {kolekce}")
            _qdrant_mode = "cloud"
            return client
        except Exception as e:
            print(
                f"[Qdrant] CLOUD PŘIPOJENÍ SELHALO url={url!r}: "
                f"{type(e).__name__}: {e}"
            )
            raise RuntimeError(
                f"Nepodařilo se připojit k Qdrant Cloud ({url}): {e}"
            ) from e

    print(f"[Qdrant] Lokální úložiště (dev): {DB_PATH}")
    _qdrant_mode = "local"
    return QdrantClient(path=DB_PATH)


def zajisti_payload_indexy(client: QdrantClient) -> None:
    """KEYWORD indexy pro spolehlivý filtr na Qdrant Cloud."""
    for field in ("user_id", "memory_type"):
        try:
            client.create_payload_index(
                collection_name=QDRANT_COLLECTION,
                field_name=field,
                field_schema=models.PayloadSchemaType.KEYWORD,
            )
            print(f"[Qdrant] Payload index {field} vytvořen/ověřen.")
        except Exception as e:
            text = str(e).lower()
            if "already" in text or "exists" in text or "409" in text:
                continue
            print(f"[Qdrant] Payload index {field} (info): {type(e).__name__}: {e}")


def zajisti_kolekci(client: QdrantClient) -> None:
    """Vytvoří kolekci zrcadlo_pamet, pokud ještě neexistuje."""
    existuje = False
    try:
        if hasattr(client, "collection_exists"):
            existuje = client.collection_exists(collection_name=QDRANT_COLLECTION)
        else:
            jmena = [c.name for c in client.get_collections().collections]
            existuje = QDRANT_COLLECTION in jmena
    except Exception as e:
        print(f"[Qdrant] Kontrola kolekce selhala: {type(e).__name__}: {e}")
        existuje = False

    if not existuje:
        client.create_collection(
            collection_name=QDRANT_COLLECTION,
            vectors_config=VectorParams(size=EMBED_DIM, distance=Distance.COSINE),
        )
        print(
            f"[Qdrant] Kolekce '{QDRANT_COLLECTION}' vytvořena "
            f"(dim={EMBED_DIM}, COSINE)."
        )

    zajisti_payload_indexy(client)


def ensure_collection_exists(
    client: QdrantClient, collection_name: str = QDRANT_COLLECTION
) -> None:
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
    """Vrátí sdílenou instanci QdrantClient."""
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
    """
    Startup: připraví Qdrant (cloud nebo local).
    Graf už nežije v SQLite — vše je v Qdrant payloadu (memory_type).
    """
    try:
        client = get_qdrant_client()
        zajisti_kolekci(client)
        mode = _qdrant_mode or "?"
        print(f"[Paměť] init OK mode={mode} collection={QDRANT_COLLECTION}")
    except Exception as e:
        print(f"[Paměť] init SELHAL: {type(e).__name__}: {e}")
        raise


def aktualni_razitko() -> str:
    return datetime.now().strftime("%d.%m.%Y %H:%M")


def rozdel_razitko_a_text(text: str) -> tuple[str, str]:
    if text.startswith("[") and "]" in text:
        konec = text.find("]")
        return text[1:konec], text[konec + 1 :].lstrip()
    return "—", text


def _placeholder_vector() -> list[float]:
    """Stabilní ne-nulový vektor pro graph body (nepoužívají se v sémantickém search)."""
    v = [0.0] * EMBED_DIM
    v[0] = 1.0
    return v


def _scroll_filter(
    user_id: str = "",
    memory_type: Optional[str] = None,
    exclude_graph: bool = False,
) -> Optional[models.Filter]:
    must = []
    must_not = []
    uid = normalizuj_user_id(user_id)
    if uid:
        must.append(
            models.FieldCondition(
                key="user_id",
                match=models.MatchValue(value=uid),
            )
        )
    if memory_type:
        must.append(
            models.FieldCondition(
                key="memory_type",
                match=models.MatchValue(value=memory_type),
            )
        )
    if exclude_graph:
        must_not.append(
            models.FieldCondition(
                key="memory_type",
                match=models.MatchValue(value=MEMORY_TYPE_GRAPH),
            )
        )
    if not must and not must_not:
        return None
    return models.Filter(
        must=must or None,
        must_not=must_not or None,
    )


def _scroll_body(
    client: QdrantClient,
    user_id: str = "",
    memory_type: Optional[str] = None,
    exclude_graph: bool = False,
) -> list:
    body = []
    offset = None
    scroll_filter = _scroll_filter(user_id, memory_type, exclude_graph)
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


def _payload_memory_type(payload: dict) -> str:
    if not payload:
        return MEMORY_TYPE_VECTOR
    val = payload.get("memory_type")
    if val in (MEMORY_TYPE_GRAPH, MEMORY_TYPE_VECTOR):
        return str(val)
    return MEMORY_TYPE_VECTOR


def _payload_text(payload: dict) -> str:
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
    # Graph bez textu — sestav z triple
    sub = payload.get("subject")
    rel = payload.get("relation")
    obj = payload.get("object")
    if sub and rel and obj:
        return f"[{sub}] -> ({rel}) -> [{obj}]"
    for val in payload.values():
        if isinstance(val, str) and val.strip():
            return val
    return ""


def _payload_timestamp(payload: dict, text_val: str) -> str:
    if payload:
        mtype = _payload_memory_type(payload)
        if mtype == MEMORY_TYPE_GRAPH:
            return "Graf"
        ts = payload.get("timestamp") or payload.get("created_at") or payload.get("date")
        if ts:
            return str(ts)
    razitko, _ = rozdel_razitko_a_text(text_val)
    if razitko and razitko != "—":
        return razitko
    return "Neznámé datum"


def _points_na_zaznamy(points: list, uid: str) -> list[dict]:
    zaznamy = []
    for p in points:
        payload = p.payload or {}
        if _payload_user_id(payload) != uid:
            continue
        text_val = _payload_text(payload)
        if not text_val:
            continue
        mtype = _payload_memory_type(payload)
        zaznamy.append(
            {
                "id": str(p.id),
                "text": text_val,
                "timestamp": _payload_timestamp(payload, text_val),
                "user_id": _payload_user_id(payload) or uid,
                "memory_type": mtype,
            }
        )
    return zaznamy


def nacti_vzpominky_uzivatele(user_id: str) -> list[dict]:
    """Načte VŠECHNY paměťové body (vector + graph) pro user_id z Qdrantu."""
    uid = normalizuj_user_id(user_id)
    if not uid:
        return []

    def _scroll_all(client: QdrantClient):
        points = _scroll_body(client, user_id=uid)
        zaznamy = _points_na_zaznamy(points, uid)
        if zaznamy:
            print(
                f"[Qdrant] nacti_vzpominky filtr OK user_id={uid!r} "
                f"count={len(zaznamy)} mode={_qdrant_mode}"
            )
            return zaznamy

        print(
            f"[Qdrant] Filtrovaný scroll = 0 pro user_id={uid!r} — "
            f"zkouším klientský filtr"
        )
        vse = _scroll_body(client)
        print(f"[Qdrant] Celkem bodů v kolekci: {len(vse)}")
        for p in vse[:20]:
            print(f"[Qdrant] sample id={p.id} payload={p.payload}")
        zaznamy = _points_na_zaznamy(vse, uid)
        print(
            f"[Qdrant] nacti_vzpominky po klientském filtru "
            f"user_id={uid!r} count={len(zaznamy)}"
        )
        return zaznamy

    try:
        return qdrant_operace(_scroll_all)
    except Exception as e:
        print(
            f"[Qdrant] nacti_vzpominky_uzivatele SELHALO user_id={uid!r}: "
            f"{type(e).__name__}: {e}"
        )
        return []


def smaz_vzpominku(point_id, user_id: str = "") -> None:
    """Smaže bod přímo z Qdrantu (vector i graph)."""
    uid = normalizuj_user_id(user_id)
    pid = point_id
    # Legacy graph-<rowid> z SQLite už nepodporujeme jako SQLite delete
    if isinstance(pid, str) and pid.startswith("graph-"):
        raise ValueError(
            f"Legacy graph id {pid!r} už není v SQLite — "
            "smaž položku s UUID z Qdrantu."
        )

    def _delete(client: QdrantClient):
        try:
            body = client.retrieve(
                collection_name=QDRANT_COLLECTION,
                ids=[pid],
                with_payload=True,
                with_vectors=False,
            )
            if body:
                print(
                    f"[Qdrant] DELETE id={pid} payload={body[0].payload} "
                    f"req_user={uid!r}"
                )
        except Exception as e:
            print(f"[Qdrant] retrieve před smazáním: {type(e).__name__}: {e}")

        client.delete(
            collection_name=QDRANT_COLLECTION,
            points_selector=models.PointIdsList(points=[pid]),
        )
        print(f"[Qdrant] DELETE OK id={pid}")

    qdrant_operace(_delete)


def _format_vzpominka_pro_frontend(zaznam: dict) -> dict:
    """Formát {id, text, timestamp} očekávaný MemoryTab."""
    text_val = zaznam.get("text", "") or ""
    mtype = zaznam.get("memory_type") or MEMORY_TYPE_VECTOR
    if mtype == MEMORY_TYPE_GRAPH:
        display_text = text_val
        timestamp = "Graf"
    else:
        razitko, text_bez = rozdel_razitko_a_text(text_val)
        display_text = text_bez if razitko != "—" else text_val
        timestamp = (
            zaznam.get("timestamp")
            or _payload_timestamp({}, text_val)
            or "Neznámé datum"
        )
    return {
        "id": str(zaznam.get("id", "")),
        "text": display_text,
        "timestamp": timestamp,
    }


def seznam_vzpominek_api(user_id: str = "") -> list[dict]:
    """
    GET /api/memories — vše z Qdrantu pro user_id (vector + graph).
    Vrací [{id, text, timestamp}] pro frontend.
    """
    uid = normalizuj_user_id(user_id)
    if not uid:
        return []

    print(
        f"[Paměť] GET memories z Qdrantu kolekce={QDRANT_COLLECTION!r} "
        f"user_id={uid!r} mode={_qdrant_mode}"
    )
    zaznamy = nacti_vzpominky_uzivatele(uid)
    vysledek = [_format_vzpominka_pro_frontend(z) for z in zaznamy]
    n_vec = sum(1 for z in zaznamy if z.get("memory_type") != MEMORY_TYPE_GRAPH)
    n_graf = sum(1 for z in zaznamy if z.get("memory_type") == MEMORY_TYPE_GRAPH)
    print(
        f"[Paměť] Memories pro frontend: {len(vysledek)} "
        f"(vector={n_vec}, graph={n_graf})"
    )
    return vysledek


def export_vzpominek(user_id: str) -> dict:
    uid = normalizuj_user_id(user_id)
    memories = seznam_vzpominek_api(uid)
    return {
        "user_id": uid,
        "exported_at": datetime.now().isoformat(timespec="seconds"),
        "count": len(memories),
        "memories": memories,
    }


def uloz_vektory(body: list, user_id: str = "") -> bool:
    """Uloží body do Qdrantu. Vrací True při úspěchu."""
    uid = normalizuj_user_id(user_id)
    if not body:
        print(f"[Qdrant] uloz_vektory: prázdný seznam bodů user_id={uid!r}")
        return False

    point_ids = [str(getattr(p, "id", "?")) for p in body]
    types = []
    for p in body:
        pl = getattr(p, "payload", None) or {}
        types.append(pl.get("memory_type", MEMORY_TYPE_VECTOR))
    print(
        f"[Qdrant] UPSERT start user_id={uid!r} mode={_qdrant_mode} "
        f"kolekce={QDRANT_COLLECTION!r} bodů={len(body)} "
        f"types={types} ids={point_ids}"
    )

    def _uloz(client: QdrantClient):
        ensure_collection_exists(client, QDRANT_COLLECTION)
        client.upsert(collection_name=QDRANT_COLLECTION, points=body)
        try:
            overeni = client.retrieve(
                collection_name=QDRANT_COLLECTION,
                ids=[body[0].id],
                with_payload=True,
                with_vectors=False,
            )
            if overeni:
                print(
                    f"[Qdrant] UPSERT ověřeno retrieve id={body[0].id} "
                    f"payload={overeni[0].payload}"
                )
            else:
                print(
                    f"[Qdrant] UPSERT varování: retrieve prázdný "
                    f"id={body[0].id} user_id={uid!r}"
                )
        except Exception as e:
            print(
                f"[Qdrant] UPSERT retrieve-ověření selhalo user_id={uid!r}: "
                f"{type(e).__name__}: {e}"
            )
        return True

    try:
        s_opakovanim(lambda: qdrant_operace(_uloz), popisek="uložení vektorů")
        print(f"[Qdrant] UPSERT ÚSPĚCH user_id={uid!r} bodů={len(body)}")
        return True
    except Exception as e:
        print(
            f"[Qdrant] UPSERT SELHAL user_id={uid!r} bodů={len(body)}: "
            f"{type(e).__name__}: {e}"
        )
        return False


def hledej_vzpominky(vektor: list[float], user_id: str, limit: int = 10) -> list[str]:
    """Sémantické vyhledávání — jen vector (ne graph) pro user_id."""
    uid = normalizuj_user_id(user_id)

    def _hledej(client: QdrantClient):
        return client.query_points(
            collection_name=QDRANT_COLLECTION,
            query=vektor,
            limit=limit,
            query_filter=_scroll_filter(user_id=uid, exclude_graph=True),
        ).points

    try:
        q_res = s_opakovanim(
            lambda: qdrant_operace(_hledej),
            popisek="vyhledávání vzpomínek",
        )
        texts = [
            _payload_text(hit.payload or {})
            for hit in q_res
            if hit.payload and _payload_memory_type(hit.payload) != MEMORY_TYPE_GRAPH
        ]
        texts = [t for t in texts if t]
        print(f"[Qdrant] hledej user_id={uid!r} hits={len(texts)}")
        return texts
    except Exception as e:
        print(
            f"[Qdrant] Vyhledávání SELHALO user_id={uid!r}: "
            f"{type(e).__name__}: {e}"
        )
        return []


def nacti_graf(user_id: str) -> list[str]:
    """Grafové vazby z Qdrantu (memory_type=graph) pro chat prompt."""
    uid = normalizuj_user_id(user_id)
    if not uid:
        return []

    def _nacti(client: QdrantClient):
        points = _scroll_body(client, user_id=uid, memory_type=MEMORY_TYPE_GRAPH)
        if not points:
            # Fallback: klientský filtr (index / starší body)
            vse = _scroll_body(client, user_id=uid)
            points = [
                p
                for p in vse
                if _payload_memory_type(p.payload or {}) == MEMORY_TYPE_GRAPH
            ]
        texts = []
        for p in points:
            t = _payload_text(p.payload or {})
            if t:
                texts.append(t)
        print(f"[Qdrant] nacti_graf user_id={uid!r} count={len(texts)}")
        return texts

    try:
        return qdrant_operace(_nacti)
    except Exception as e:
        print(f"[Qdrant] nacti_graf SELHAL user_id={uid!r}: {type(e).__name__}: {e}")
        return []


def uloz_trojice(user_id: str, trojice: list[dict]) -> bool:
    """
    Uloží grafové trojice do Qdrantu jako body s memory_type=graph.
    (Žádný SQLite — přežije restart Renderu.)
    """
    uid = normalizuj_user_id(user_id)
    if not trojice:
        return True

    body = []
    for t in trojice:
        sub = (t.get("subject") or "").lower().strip()
        rel = (t.get("relation") or "").lower().strip()
        obj = (t.get("object") or "").lower().strip()
        if not (sub and rel and obj):
            continue
        body.append(
            vytvor_graph_point(
                vektor=_placeholder_vector(),
                user_id=uid,
                subject=sub,
                relation=rel,
                object_=obj,
            )
        )

    if not body:
        print(f"[Graf] žádné platné triples user_id={uid!r}")
        return False

    ok = uloz_vektory(body, user_id=uid)
    print(
        f"[Graf] Qdrant graph upsert user_id={uid!r} "
        f"triples={len(body)} ok={ok}"
    )
    return ok


def posledni_vzpominky_texty(user_id: str, limit: int = 10) -> list[str]:
    """Poslední vektorové vzpomínky (bez graph) pro uvítací kontext."""
    vse = nacti_vzpominky_uzivatele(user_id)
    vektory = [z for z in vse if z.get("memory_type") != MEMORY_TYPE_GRAPH]

    def _sort_key(zaznam):
        razitko, _ = rozdel_razitko_a_text(zaznam.get("text", ""))
        try:
            return datetime.strptime(razitko, "%d.%m.%Y %H:%M")
        except Exception:
            return datetime.min

    serazene = sorted(vektory, key=_sort_key)
    return [z.get("text", "") for z in serazene[-limit:] if z.get("text")]


def vytvor_point(vektor: list[float], text: str, user_id: str):
    uid = normalizuj_user_id(user_id)
    return models.PointStruct(
        id=str(uuid.uuid4()),
        vector=vektor,
        payload={
            "text": text,
            "user_id": uid,
            "memory_type": MEMORY_TYPE_VECTOR,
        },
    )


def vytvor_graph_point(
    vektor: list[float],
    user_id: str,
    subject: str,
    relation: str,
    object_: str,
):
    uid = normalizuj_user_id(user_id)
    text = f"[{subject}] -> ({relation}) -> [{object_}]"
    return models.PointStruct(
        id=str(uuid.uuid4()),
        vector=vektor if vektor else _placeholder_vector(),
        payload={
            "text": text,
            "user_id": uid,
            "memory_type": MEMORY_TYPE_GRAPH,
            "subject": subject,
            "relation": relation,
            "object": object_,
            "timestamp": "Graf",
        },
    )
