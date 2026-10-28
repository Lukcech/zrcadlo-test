# -*- coding: utf-8 -*-
"""Volání Gemini API: embeddingy, odpovědi, učení z paměti, retry 429/503."""

from __future__ import annotations

from dotenv import load_dotenv
import os
from pathlib import Path

load_dotenv(Path(__file__).resolve().parent.parent / ".env")
load_dotenv()

import json
import time
from typing import Callable, Optional

from google import genai
from google.genai import types

from . import database as db

# Primární lite (stabilní); fallback plný flash
DEFAULT_MODEL = "gemini-3.5-flash-lite"
FALLBACK_MODEL = "gemini-3.5-flash"
MODEL_NAME = os.getenv("GEMINI_MODEL", DEFAULT_MODEL).strip() or DEFAULT_MODEL
REZIM_KREATIVNI = "Kreativní parťák"
REZIM_TREZOR = "Striktní Trezor (NotebookLM)"
HLASKA_API_VYTIZENE = (
    "Google API je momentálně vytížené, zkus to prosím za pár sekund znovu."
)
HLASKA_API_NEDOSTUPNE = (
    "Omlouvám se, služba Gemini je teď dočasně nedostupná. "
    "Zkus prosím odeslat zprávu znovu za chvíli."
)
# Generování odpovědi: 3 pokusy, pauza 1.5 s při 503/429
GEN_MAX_POKUSU = 3
GEN_CEKANI_S = 1.5
# Embeddingy a ostatní volání
GEMINI_MAX_POKUSU = 3
GEMINI_CEKANI_S = 2

_genai_client: genai.Client | None = None


class GeminiVytizeneError(RuntimeError):
    """Gemini API zůstalo vytížené i po opakovaných pokusech."""


def nacti_api_klic() -> str:
    """Načte Gemini klíč z GEMINI_API_KEY nebo API_KEY."""
    klic = os.getenv("GEMINI_API_KEY") or os.getenv("API_KEY")
    if klic and klic not in {"tvuj_gemini_api_klic", ""}:
        return klic
    raise ValueError(
        "Gemini API klíč nebyl nalezen. Nastav GEMINI_API_KEY nebo API_KEY v .env."
    )


def get_genai_client() -> genai.Client:
    global _genai_client
    if _genai_client is None:
        _genai_client = genai.Client(api_key=nacti_api_klic())
    return _genai_client


def _normalizuj_nazev_modelu(nazev: str) -> str:
    """Odstraní prefix models/ — SDK očekává holý název modelu."""
    n = (nazev or "").strip()
    if not n:
        return DEFAULT_MODEL
    if n.startswith("models/"):
        n = n[len("models/") :]
    return n or DEFAULT_MODEL


def vyber_model(model: Optional[str] = None) -> str:
    """Výchozí gemini-3.5-flash-lite; frontend může přepsat."""
    return _normalizuj_nazev_modelu((model or "").strip() or MODEL_NAME)


def normalizuj_rezim(mode: str) -> str:
    """Mapuje aliasy režimu na kanonické názvy."""
    m = (mode or "").strip().lower()
    if m in {
        "trezor",
        "striktní trezor",
        "striktni trezor",
        "notebooklm",
        REZIM_TREZOR.lower(),
    }:
        return REZIM_TREZOR
    return REZIM_KREATIVNI


def teplota_pro_rezim(rezim: str) -> float:
    return 0.0 if rezim == REZIM_TREZOR else 0.6


def formatuj_historii_chatu(zpravy: list[dict], limit: int = 10) -> str:
    if not zpravy:
        return "Žádná předchozí konverzace v této relaci."
    casti = []
    for msg in zpravy[-limit:]:
        role = "Uživatel" if msg.get("role") == "user" else "Zrcadlo"
        casti.append(f"{role}: {msg.get('content', '')}")
    return "\n".join(casti)


def sestav_gemini_contents(
    historie_chatu: list[dict],
    aktualni_zprava: str,
    limit: int = 10,
) -> list:
    """
    Multi-turn contents pro Gemini API (role user/model).
    Historie = předchozí zprávy; aktuální zpráva je poslední user turn.
    """
    contents: list = []
    for msg in (historie_chatu or [])[-limit:]:
        role_raw = (msg.get("role") or "").strip().lower()
        text = (msg.get("content") or "").strip()
        if not text:
            continue
        gemini_role = "user" if role_raw == "user" else "model"
        contents.append(
            types.Content(role=gemini_role, parts=[types.Part.from_text(text=text)])
        )
    contents.append(
        types.Content(
            role="user",
            parts=[types.Part.from_text(text=(aktualni_zprava or "").strip())],
        )
    )
    return contents


def formatuj_chybu(exc: Exception) -> str:
    if isinstance(exc, GeminiVytizeneError):
        return str(exc) or HLASKA_API_VYTIZENE
    kod = getattr(exc, "code", None) or getattr(exc, "status_code", None)
    if kod is None:
        details = getattr(exc, "details", None)
        if isinstance(details, dict):
            kod = details.get("code") or details.get("status")
    cast = f"{type(exc).__name__}: {exc}"
    if kod is not None:
        return f"Chyba API [{kod}]: {cast}"
    return f"Chyba API: {cast}"


def je_rate_limit_429(exc: Exception) -> bool:
    """True jen u ResourceExhausted / HTTP 429 (ne plošně u všech Gemini chyb)."""
    if isinstance(exc, GeminiVytizeneError):
        return True
    kod = getattr(exc, "code", None) or getattr(exc, "status_code", None)
    text = str(exc).lower()
    nazev = type(exc).__name__.lower()
    return (
        kod == 429
        or "429" in text
        or "resourceexhausted" in nazev
        or "resource_exhausted" in text
        or "resource exhausted" in text
    )


def je_prechodna_gemini_chyba(exc: Exception) -> bool:
    """Přechodné chyby vhodné k retry (429, 503, overload)."""
    if je_rate_limit_429(exc):
        return True
    kod = getattr(exc, "code", None) or getattr(exc, "status_code", None)
    text = str(exc).lower()
    nazev = type(exc).__name__.lower()
    return (
        kod == 503
        or "503" in text
        or "servererror" in nazev
        or "server error" in text
        or "high demand" in text
        or "unavailable" in text
        or "overloaded" in text
        or "rate limit" in text
        or "ratelimit" in text
        or "quota" in text
    )


def loguj_gemini_chybu(exc: Exception) -> None:
    print(f"❌ DETEKTOVÁNA CHYBA GEMINI API: {type(exc).__name__} - {exc}")


def zprava_pro_uzivatele(exc: Exception) -> str:
    """Na frontend: friendly text jen u 429; jinak přesný popis chyby."""
    if je_rate_limit_429(exc):
        return HLASKA_API_VYTIZENE
    text = str(exc).strip()
    if text:
        return text
    return formatuj_chybu(exc)


def gemini_s_opakovanim(fn: Callable, pokusu: int = GEMINI_MAX_POKUSU):
    """Při 429 / ResourceExhausted / 503 zopakuje volání až 3× s pauzou 2 s."""
    posledni = None
    for pokus in range(1, pokusu + 1):
        try:
            return fn()
        except Exception as e:
            posledni = e
            loguj_gemini_chybu(e)
            if je_prechodna_gemini_chyba(e) and pokus < pokusu:
                print(
                    f"[Gemini] Přechodná chyba (pokus {pokus}/{pokusu}), "
                    f"čekám {GEMINI_CEKANI_S}s: {type(e).__name__}"
                )
                time.sleep(GEMINI_CEKANI_S)
                continue
            break

    if je_rate_limit_429(posledni):
        raise GeminiVytizeneError(HLASKA_API_VYTIZENE) from posledni
    raise RuntimeError(zprava_pro_uzivatele(posledni)) from posledni


def generuj_text_s_retry_a_fallback(
    contents,
    model_name: str,
    teplota: float,
    system_instruction: Optional[str] = None,
) -> str:
    """
    Generování odpovědi (multi-turn contents): až 3 pokusy při 503/429,
    pak záložní model. Při totálním selhání vrátí českou hlášku.
    """
    client = get_genai_client()
    primarni = _normalizuj_nazev_modelu(model_name or DEFAULT_MODEL)
    fallback = _normalizuj_nazev_modelu(FALLBACK_MODEL)
    kandidati = [primarni]
    if primarni != fallback:
        kandidati.append(fallback)

    config_kwargs: dict = {"temperature": teplota}
    if system_instruction:
        config_kwargs["system_instruction"] = system_instruction

    for idx, model in enumerate(kandidati):
        for pokus in range(1, GEN_MAX_POKUSU + 1):
            try:
                odpoved = client.models.generate_content(
                    model=model,
                    contents=contents,
                    config=types.GenerateContentConfig(**config_kwargs),
                )
                text = (odpoved.text or "").strip() if odpoved else ""
                if text:
                    if model != primarni:
                        print(f"[Gemini] Odpověď přes záložní model: {model}")
                    return text
                raise RuntimeError("Prázdná odpověď od Gemini API.")
            except Exception as e:
                loguj_gemini_chybu(e)
                je_posledni_pokus = pokus >= GEN_MAX_POKUSU
                je_posledni_model = idx >= len(kandidati) - 1

                if je_prechodna_gemini_chyba(e) and not je_posledni_pokus:
                    print(
                        f"[Gemini] {model}: 503/429 (pokus {pokus}/{GEN_MAX_POKUSU}), "
                        f"čekám {GEN_CEKANI_S}s…"
                    )
                    time.sleep(GEN_CEKANI_S)
                    continue

                if not je_posledni_model:
                    print(
                        f"[Gemini] Model {model} selhal po {pokus} pokusech — "
                        f"přepínám na {kandidati[idx + 1]}"
                    )
                    time.sleep(GEN_CEKANI_S)
                    break

                print(f"[Gemini] Všechny pokusy i fallback selhaly: {e}")
                return HLASKA_API_NEDOSTUPNE

    return HLASKA_API_NEDOSTUPNE


def ziskej_embedding(text: str) -> list[float]:
    """Vektorový otisk přes gemini-embedding-001 (dim 3072 dle test_step_by_step)."""
    client = get_genai_client()

    def _call():
        vysledek = client.models.embed_content(
            model=db.EMBED_MODEL,
            contents=text,
        )
        values = list(vysledek.embeddings[0].values)
        if not values:
            raise RuntimeError("Embedding vrátil prázdný vektor.")
        if len(values) != db.EMBED_DIM:
            raise RuntimeError(
                f"Embedding má dimenzi {len(values)}, očekáváno {db.EMBED_DIM}."
            )
        return values

    return gemini_s_opakovanim(_call)


def ziskej_kontext(dotaz: str, user_id: str) -> tuple[list[str], list[str]]:
    """Načte vektorové vzpomínky + graf. Při rate-limitu vrátí prázdný kontext."""
    vektory_text: list[str] = []
    try:
        vektor = ziskej_embedding(dotaz)
        vektory_text = db.hledej_vzpominky(vektor, user_id, limit=10)
    except Exception as e:
        loguj_gemini_chybu(e)
        if je_rate_limit_429(e):
            print(
                "[Paměť] Rate limit / vytížené API při vyhledávání — "
                "pokračuji bez vektorového kontextu."
            )
        else:
            print(f"[Paměť] Vyhledávání selhalo (soft): {e}")

    try:
        graf_res = db.nacti_graf(user_id)
    except Exception as e:
        print(f"[Paměť] Načtení grafu selhalo (soft): {e}")
        graf_res = []

    return vektory_text, graf_res


def uvitaci_zprava(user_id: str) -> str:
    jmeno = (user_id or "").strip().capitalize() or "příteli"
    return (
        f"Ahoj {jmeno}, vítám tě zpět! Procházím tvé uložené vzpomínky "
        "a navazuji tam, kde jsme skončili. Na co se chceš dnes zaměřit?"
    )


def uc_se_z_zpravy(zprava: str, user_id: str, model: Optional[str] = None) -> None:
    """Extrahuje fakta a ukládá je do Qdrantu + SQLite grafu (s podrobným logováním)."""
    uid = (user_id or "").strip().lower()
    model_name = vyber_model(model)
    print(f"[Paměť] Extrakce start user_id={uid!r} model={model_name}")

    prompt = f"""
Pokud zpráva obsahuje trvalé osobní fakta, preference, vztahy, plány cest nebo tělesné/psychické prožitky uživatele, vyextrahuj je.
Pokud zpráva neobsahuje žádná nová osobní fakta (jen pozdrav, obecný dotaz o světě apod.), vrať prázdná pole.

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

    client = get_genai_client()

    try:
        def _call():
            return client.models.generate_content(
                model=model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=schema,
                ),
            )

        odpoved = gemini_s_opakovanim(_call)
        data = json.loads(odpoved.text or "{}")
    except GeminiVytizeneError as e:
        print(f"[Paměť] Extrakce SELHALA (kvóta/vytížení) user_id={uid!r}: {e}")
        return
    except Exception as e:
        print(
            f"[Paměť] Extrakce SELHALA user_id={uid!r}: "
            f"{type(e).__name__}: {e}"
        )
        return

    fakta = [f for f in (data.get("facts") or []) if isinstance(f, str) and f.strip()]
    trojice = data.get("triples") or []
    print(f"[Paměť] Extrahovaná fakta ({len(fakta)}) pro user_id={uid!r}: {fakta}")
    print(f"[Paměť] Extrahované triples ({len(trojice)}) pro user_id={uid!r}: {trojice}")

    if not fakta and not trojice:
        print(f"[Paměť] Nic k uložení (prázdná extrakce) user_id={uid!r}")
        return

    if fakta:
        razitko = db.aktualni_razitko()
        body = []
        for f in fakta:
            text_s_casem = f"[{razitko}] {f}"
            try:
                print(f"[Paměť] Embedding fakta: {text_s_casem!r}")
                v = ziskej_embedding(text_s_casem)
                body.append(db.vytvor_point(v, text_s_casem, uid))
            except Exception as e:
                print(
                    f"[Paměť] Embedding SELHAL pro fakt {f!r} "
                    f"user_id={uid!r}: {type(e).__name__}: {e}"
                )
                continue
        if body:
            try:
                ok = db.uloz_vektory(body, user_id=uid)
                if ok:
                    print(
                        f"[Paměť] Qdrant UPSERT OK user_id={uid!r} "
                        f"bodů={len(body)}"
                    )
                else:
                    print(
                        f"[Paměť] Qdrant UPSERT NEÚSPĚŠNÝ user_id={uid!r} "
                        f"bodů={len(body)}"
                    )
            except Exception as e:
                print(
                    f"[Paměť] Qdrant UPSERT VÝJIMKA user_id={uid!r}: "
                    f"{type(e).__name__}: {e}"
                )
        else:
            print(f"[Paměť] Žádné body k upsertu (embeddingy selhaly) user_id={uid!r}")

    try:
        db.uloz_trojice(uid, trojice)
        print(f"[Paměť] Graf SQLite uložen user_id={uid!r} triples={len(trojice)}")
    except Exception as e:
        print(
            f"[Paměť] Graf SQLite SELHAL user_id={uid!r}: "
            f"{type(e).__name__}: {e}"
        )


def _spolecna_pravidla_chovani(user_id: str, je_prvni_zprava: bool) -> str:
    pozdrav = (
        "Toto je PRVNÍ zpráva relace — smíš krátce pozdravit."
        if je_prvni_zprava
        else (
            "Konverzace už běží — NEZDRAV (žádné „Ahoj …“), "
            "neopakuj jméno uživatele zbytečně, rovnou navazuj na téma."
        )
    )
    return f"""
Jsi Zrcadlo, AI průvodce uživatele '{user_id}'.

CHOVÁNÍ:
- {pozdrav}
- Piš přirozeně, stručně a věcně. Žádné submisivní omluvy typu
  „sypu si popel na hlavu“, „omlouvám se, že jsem to pokazil“.
  Při upřesnění jen krátce oprav a pokračuj v tématu.
- Drž se aktuálního tématu konverzace. Nenasazuj nesouvisející otázky
  (např. ubytování, když uživatel řeší faunu).
- Multi-turn: respektuj celou historii chatu (místa, jména, plány zmíněné dříve).

ZNALOSTI O SVĚTĚ vs. OSOBNÍ FAKTA:
- Encyklopedické znalosti (geografie, biologie, cestování, kultura…) POUŽÍVEJ VOLNĚ a přesně.
- NIKDY nevymýšlej geografické/biologické nesmysly (např. pumy žijící divoce v Indii —
  puma je americká; v Indii patří tygr, levhart atd.).
- Zákaz vymýšlení („grounding“) platí POUZE pro osobní fakta o uživateli
  (preference, vztahy, zdraví, majetek, plány). Ta ber JEN z paměti / historie chatu.
  Pokud osobní fakt v paměti není, řekni to otevřeně — nevymýšlej.
"""


def generuj_odpoved(
    dotaz: str,
    vektory: list[str],
    graf: list[str],
    user_id: str,
    historie_chatu: list[dict],
    rezim: str,
    uvitaci_kontext: Optional[list[str]] = None,
    model: Optional[str] = None,
    je_prvni_zprava: bool = False,
) -> str:
    rezim = normalizuj_rezim(rezim)
    teplota = teplota_pro_rezim(rezim)
    model_name = vyber_model(model)
    spolecna = _spolecna_pravidla_chovani(user_id, je_prvni_zprava)

    if rezim == REZIM_TREZOR:
        rezimova = f"""
Režim: Striktní Trezor (NotebookLM).
- Osobní fakta o uživateli: VÝHRADNĚ z vektorových vzpomínek a grafových vazeb.
  Pokud osobní informace v paměti není: „Tuto informaci v databázi nemám.“
- Světová/encyklopedická fakta smíš doplnit ze svých znalostí, pokud pomáhají
  kontextu — ale ne jako náhradu za chybějící osobní data.
- Historii chatu používej k pochopení otázky a kontinuity tématu.
"""
    else:
        rezimova = f"""
Režim: Kreativní parťák.
- Odpovídej přátelsky a s pochopením, plynule navazuj na historii.
- Osobní fakta o uživateli: jen z paměti / historie — nevymýšlej.
- Světová témata (příroda, země, cestování…): využij své znalosti naplno a přesně.
"""

    uvitaci = uvitaci_kontext or []
    vektory_jedinecne = []
    videne = set(uvitaci)
    for text in vektory:
        if text and text not in videne:
            vektory_jedinecne.append(text)
            videne.add(text)

    system_instruction = f"""
{spolecna}
{rezimova}

PAMĚŤOVÝ KONTEXT uživatele '{user_id}' (osobní fakta):
UVÍTACÍ PŘEHLED:
{chr(10).join(uvitaci) if uvitaci else "Žádný."}

VEKTOROVÉ VZPOMÍNKY (relevantní k dotazu):
{chr(10).join(vektory_jedinecne) if vektory_jedinecne else "Žádné."}

GRAFOVÉ VAZBY:
{chr(10).join(graf) if graf else "Žádné."}

Časová razítka ve vzpomínkách mají tvar [DD.MM.YYYY HH:MM] — ber je v potaz.
"""

    contents = sestav_gemini_contents(historie_chatu, dotaz, limit=10)
    print(
        f"[Gemini] multi-turn turns={len(contents)} "
        f"historie={len(historie_chatu)} prvni={je_prvni_zprava}"
    )
    return generuj_text_s_retry_a_fallback(
        contents,
        model_name,
        teplota,
        system_instruction=system_instruction.strip(),
    )


def zpracuj_zpravu(
    user_id: str,
    message: str,
    mode: str,
    historie_chatu: Optional[list[dict]] = None,
    uvitaci_kontext: Optional[list[str]] = None,
    model: Optional[str] = None,
    je_prvni_zprava: bool = False,
    **kwargs,
) -> str:
    """Chat pipeline: kontext → multi-turn odpověď.
    Učení probíhá přes BackgroundTasks v main.py.
    """
    if model is None and "model" in kwargs:
        model = kwargs.get("model")
    if "je_prvni_zprava" in kwargs and not je_prvni_zprava:
        je_prvni_zprava = bool(kwargs.get("je_prvni_zprava"))

    historie = list(historie_chatu or [])
    if uvitaci_kontext is None:
        uvitaci_kontext = db.posledni_vzpominky_texty(user_id, limit=10)

    print("👉 1/2: Vytvářím embedding dotazu a načítám kontext...")
    vektory, graf = ziskej_kontext(message, user_id)

    print("👉 2/2: Generuji multi-turn odpověď Zrcadla...")
    return generuj_odpoved(
        message,
        vektory,
        graf,
        user_id,
        historie,
        mode,
        uvitaci_kontext=uvitaci_kontext,
        model=model,
        je_prvni_zprava=je_prvni_zprava or len(historie) == 0,
    )


def uc_se_na_pozadi(
    zprava: str, user_id: str, model: Optional[str] = None
) -> None:
    """Wrapper pro BackgroundTasks — chyby nesmí shodit worker."""
    uid = (user_id or "").strip().lower()
    print(f"👉 [pozadí] Extrahuji a ukládám vzpomínky pro user_id={uid!r}...")
    try:
        uc_se_z_zpravy(zprava, uid, model=model)
        print(f"👉 [pozadí] Hotovo user_id={uid!r}")
    except Exception as e:
        print(f"[pozadí] Paměť SELHALA user_id={uid!r}: {type(e).__name__}: {e}")
