from fastapi import FastAPI
from api.routes import search2

app = FastAPI(title="Semantic Search API")

app.include_router(search2.router, prefix="/api")

@app.get("/")
def read_root():
    return {"message": "Semantic Search API running."}
