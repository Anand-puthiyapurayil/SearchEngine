# project/api/routes/search.py

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Optional
import yaml
import faiss
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from loguru import logger

logger.add("logs/search_endpoint.log", rotation="1 day", compression="zip")

router = APIRouter()

class SearchQuery(BaseModel):
    query: str
    top_k: int = 10

class SearchResult(BaseModel):
    product_id: Optional[str] = None
    product_name: Optional[str] = None
    service_id: Optional[str] = None
    service_name: Optional[str] = None
    description: str
    type: str
    price: Optional[float] = None       # New field for products
    rating: Optional[float] = None      # New field for products
    price_range: Optional[str] = None   # For services if available
    location: Optional[str] = None      # Added field for location
    similarity: float

class SearchResponse(BaseModel):
    results: List[SearchResult]

with open("config.yaml", 'r') as f:
    config = yaml.safe_load(f)

logger.info("Loading model and FAISS indices for search...")
prod_index = faiss.read_index(config['faiss']['products_index'])
serv_index = faiss.read_index(config['faiss']['services_index'])

df_products = pd.read_csv(config['faiss']['products_mapping'])
df_services = pd.read_csv(config['faiss']['services_mapping'])

model = SentenceTransformer(config['model']['fine_tuned'])

@router.post("/search")
def search_endpoint(query: SearchQuery):
    if not query.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")
    logger.info(f"Received query: {query.query}")

    # Determine intent
    intent = "service" if "service" in query.query.lower() else "product"

    if intent == "service":
        df = df_services
        index = serv_index
    else:
        df = df_products
        index = prod_index

    # Encode query
    query_emb = model.encode([query.query], convert_to_numpy=True).astype('float32')
    faiss.normalize_L2(query_emb)

    distances, indices = index.search(query_emb, query.top_k)
    results = []
    for dist, idx in zip(distances[0], indices[0]):
        if idx == -1:
            continue
        record = df.iloc[idx]

        def nan_to_none(value):
            return None if pd.isna(value) else value

        product_id = nan_to_none(record.get("product_id"))
        product_name = nan_to_none(record.get("product_name"))
        service_id = nan_to_none(record.get("service_id"))
        service_name = nan_to_none(record.get("service_name"))
        description = nan_to_none(record.get("description"))
        rtype = nan_to_none(record.get("type"))
        
        # Extract numerical fields
        price = nan_to_none(record.get("price"))
        rating = nan_to_none(record.get("rating"))
        price_range = nan_to_none(record.get("price_range"))
        
        # Extract location
        location = nan_to_none(record.get("supplier_location"))  # Ensure this matches your CSV column name

        # Adjust fields based on intent
        if intent == "product":
            # Show product_id, product_name, price, rating
            # Hide service fields
            service_id = None
            service_name = None
            price_range = None
        else:
            # Show service_id, service_name, price_range
            # Hide product fields, price, rating
            product_id = None
            product_name = None
            price = None
            rating = None

        res = SearchResult(
            product_id=product_id,
            product_name=product_name,
            service_id=service_id,
            service_name=service_name,
            description=description if description else "No description",
            type=rtype if rtype else "unknown",
            price=price,
            rating=rating,
            price_range=price_range,
            location=location,  # Include location in the result
            similarity=float(dist)
        )
        results.append(res)

    logger.info(f"Returning {len(results)} results for query: {query.query}")
    response_dict = {"results": [r.dict(exclude_none=True) for r in results]}
    return JSONResponse(content=response_dict)
