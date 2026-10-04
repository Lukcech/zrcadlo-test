import os
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
    except Exception as e:
        return []

# --- PROPOJENÍ S REACT FRONTENDEM ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DIST_DIR = os.path.join(BASE_DIR, "frontend", "dist")

if os.path.exists(DIST_DIR):
    assets_dir = os.path.join(DIST_DIR, "assets")
    if os.path.exists(assets_dir):
        app.mount("/assets", StaticFiles(directory=assets_dir), name="assets")

    @app.get("/{full_path:path}")
    async def serve_react(full_path: str):
        if full_path.startswith("api/"):
            return {"detail": "Not Found"}
        file_path = os.path.join(DIST_DIR, full_path)
        if os.path.exists(file_path) and os.path.isfile(file_path):
            return FileResponse(file_path)
        return FileResponse(os.path.join(DIST_DIR, "index.html"))
