# -*- coding: utf-8 -*-
"""FastAPI backend pro Projekt Zrcadlo."""

from __future__ import annotations

from typing import Any, Optional
from uuid import UUID

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from . import ai_engine
from . import database as db

app = FastAPI(title="Projekt Zrcadlo API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Krátkodobá paměť chatu na straně serveru (per user_id)
_chat_histories: dict[str, list[dict]] = {}
_welcome_context: dict[str, list[str]] = {}


class ChatRequest(BaseModel):
    user_id: str = Field(..., min_length=1, description="Identifikátor uživatele")
    message: str = Field(..., min_length=1, description="Zpráva uživatele")
    mode: str = Field(
        default=ai_engine.REZIM_KREATIVNI,
        description="Kreativní parťák | Striktní Trezor (NotebookLM)",
    )
    model: Optional[str] = Field(
        default=None,
        description="Volitelný Gemini model (např. gemini-3.5-flash-lite)",
    )


class ChatResponse(BaseModel):
    user_id: str
    mode: str
    reply: str
    welcome: Optional[str] = None


class MemoryItem(BaseModel):
    id: str
    text: str
    timestamp: Optional[str] = None


class MemoriesResponse(BaseModel):
    user_id: str
    count: int
    memories: list[MemoryItem]


class DeleteMemoryResponse(BaseModel):
    ok: bool
    memory_id: str


def _parse_point_id(memory_id: str) -> Any:
    """Qdrant očekává UUID objekt nebo int; string UUID převedeme."""
    try:
        return UUID(memory_id)
    except ValueError:
        if memory_id.isdigit():
            return int(memory_id)
        return memory_id


@app.on_event("startup")
def startup() -> None:
    db.init_graph_db()


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.post("/api/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    user_id = req.user_id.strip().lower()
    mode = ai_engine.normalizuj_rezim(req.mode)
    message = req.message.strip()

    if not user_id:
        raise HTTPException(status_code=400, detail="user_id nesmí být prázdný.")
    if not message:
        raise HTTPException(status_code=400, detail="Zpráva nesmí být prázdná.")

    if user_id not in _chat_histories:
        _chat_histories[user_id] = []
    historie = _chat_histories[user_id]

    welcome_msg = None
    if not historie:
        _welcome_context[user_id] = db.posledni_vzpominky_texty(user_id, limit=10)
        welcome_msg = ai_engine.uvitaci_zprava(user_id)
        historie.append({"role": "assistant", "content": welcome_msg})

    historie.append({"role": "user", "content": message})

    try:
        reply = ai_engine.zpracuj_zpravu(
            user_id=user_id,
            message=message,
            mode=mode,
            historie_chatu=historie[:-1],
            uvitaci_kontext=_welcome_context.get(user_id, []),
            model=req.model,
        )
    except ai_engine.GeminiVytizeneError as e:
        print(f"❌ DETEKTOVÁNA CHYBA GEMINI API: {type(e).__name__} - {e}")
        # Uživatelsky přívětivá odpověď místo chybového JSON
        reply = str(e).strip() or ai_engine.HLASKA_API_NEDOSTUPNE
    except Exception as e:
        print(f"❌ DETEKTOVÁNA CHYBA GEMINI API: {type(e).__name__} - {e}")
        if ai_engine.je_prechodna_gemini_chyba(e) or ai_engine.je_rate_limit_429(e):
            reply = ai_engine.HLASKA_API_NEDOSTUPNE
        else:
            historie.pop()
            raise HTTPException(
                status_code=500,
                detail=ai_engine.zprava_pro_uzivatele(e),
            )

    historie.append({"role": "assistant", "content": reply})

    return ChatResponse(
        user_id=user_id,
        mode=mode,
        reply=reply,
        welcome=welcome_msg,
    )


@app.get("/api/memories", response_model=MemoriesResponse)
def list_memories(user_id: str = Query(..., min_length=1)):
    uid = user_id.strip().lower()
    if not uid:
        raise HTTPException(status_code=400, detail="user_id nesmí být prázdný.")
    print(f"[API] GET /api/memories user_id={uid!r}")
    try:
        raw = db.seznam_vzpominek_api(uid)
    except Exception as e:
        print(f"[API] GET /api/memories chyba: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    memories = [
        MemoryItem(
            id=m["id"],
            text=m["text"],
            timestamp=m.get("timestamp") or "Neznámé datum",
        )
        for m in raw
    ]
    print(f"[API] GET /api/memories → {len(memories)} položek pro frontend")
    return MemoriesResponse(user_id=uid, count=len(memories), memories=memories)


@app.delete("/api/memories/{memory_id}", response_model=DeleteMemoryResponse)
def delete_memory(
    memory_id: str,
    user_id: str = Query("", description="Volitelné user_id pro čištění grafu"),
):
    point_id = _parse_point_id(memory_id)
    print(f"[API] DELETE /api/memories/{memory_id} user_id={user_id!r}")
    try:
        db.smaz_vzpominku(point_id, user_id=user_id.strip().lower() if user_id else "")
    except Exception as e:
        print(f"[API] DELETE memory chyba: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Smazání vzpomínky selhalo: {e}",
        )
    return DeleteMemoryResponse(ok=True, memory_id=str(memory_id))


@app.get("/api/export")
def export_memories(user_id: str = Query(..., min_length=1)):
    uid = user_id.strip().lower()
    if not uid:
        raise HTTPException(status_code=400, detail="user_id nesmí být prázdný.")
    try:
        payload = db.export_vzpominek(uid)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    filename = f"{uid}_zrcadlo_memory.json"
    return JSONResponse(
        content=payload,
        headers={
            "Content-Disposition": f'attachment; filename="{filename}"',
        },
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=True)
