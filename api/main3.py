from fastapi import FastAPI
from api.routes import huggingface_llm

app = FastAPI(title="Semantic Search API")

app.include_router(huggingface_llm.router, prefix="/api")

@app.get("/")
def read_root():
    return {"message": "Semantic Search API running."}
