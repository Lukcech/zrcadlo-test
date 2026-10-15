import os
from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from backend.database import init_db, qdrant, COLLECTION_NAME
from backend.ai_engine import generate_response
from qdrant_client.http import models

app = FastAPI(title="Zrcadlo API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = Path(__file__).resolve().parent.parent
DIST_DIR = BASE_DIR / "frontend" / "dist"

@app.on_event("startup")
def startup_event():
    init_db()

class ChatRequest(BaseModel):
    user_id: str
    message: str
    persona: str = "Kreativní parťák"

@app.post("/api/chat")
def chat_endpoint(req: ChatRequest):
    reply = generate_response(
        user_id=req.user_id,
        prompt=req.message,
        user_persona=req.persona
    )
    return {"reply": reply}

@app.get("/api/memories/{user_id}")
def get_memories_endpoint(user_id: str):
    try:
        results = qdrant.scroll(
            collection_name=COLLECTION_NAME,
            scroll_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="user_id",
                        match=models.MatchValue(value=user_id)
                    )
                ]
            ),
            limit=100
        )[0]
        return [{"id": hit.id, "text": hit.payload["text"]} for hit in results]
    except Exception:
        return []

assets_dir = DIST_DIR / "assets"
if assets_dir.exists():
    app.mount("/assets", StaticFiles(directory=str(assets_dir)), name="assets")

@app.get("/{full_path:path}")
async def serve_react(full_path: str):
    if full_path.startswith("api/"):
        return {"detail": "Not Found"}
    
    file_path = DIST_DIR / full_path
    if file_path.exists() and file_path.is_file():
        return FileResponse(str(file_path))
    
    index_path = DIST_DIR / "index.html"
    if index_path.exists():
        return FileResponse(str(index_path))
    
    return {
        "status": "Chyba načtení frontendu",
        "detail": f"Složka s frontendem nebyla nalezena na adrese: {index_path}"
    }
