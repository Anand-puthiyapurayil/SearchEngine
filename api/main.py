from fastapi import FastAPI
from api.routes import search

app = FastAPI(title="Semantic Search API")

app.include_router(search.router, prefix="/api")

@app.get("/")
def read_root():
    return {"message": "Semantic Search API running."}
