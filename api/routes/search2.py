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
    price: Optional[float] = None
    rating: Optional[float] = None
    price_range: Optional[str] = None
    location: Optional[str] = None
    similarity: float

class SearchResponse(BaseModel):
    results: List[SearchResult]

# 1) Load config
with open("config.yaml", 'r') as f:
    config = yaml.safe_load(f)

logger.add("logs/search_endpoint.log", rotation="1 day", compression="zip")
logger.info("Loading model and FAISS indices for search...")

# 2) Load FAISS indices
prod_index = faiss.read_index(config['faiss']['products_index'])
serv_index = faiss.read_index(config['faiss']['services_index'])

# 3) Load DataFrames
df_products = pd.read_csv(config['faiss']['products_mapping'])
df_services = pd.read_csv(config['faiss']['services_mapping'])

# 4) Load the fine-tuned model (SentenceTransformer)
model = SentenceTransformer(config['model']['fine_tuned'])

# 5) Price Percentiles from config
price_percentiles = config.get('price_percentiles', None)
if not price_percentiles:
    logger.warning("No price_percentiles found in config.yaml. Using fallback values.")
    price_percentiles = {'q1': 2000.0, 'q2': 5000.0, 'q3': 10000.0}

q1 = price_percentiles['q1']  # e.g. 3000
q2 = price_percentiles['q2']  # e.g. 7000
q3 = price_percentiles['q3']  # e.g. 14000

def parse_price_synonyms(query_text: str):
    """
    Translates synonyms into dynamic thresholds using q1, q2, q3.
    e.g. 'cheap' => price_lt = q1, 'affordable' => price_lt = q2, 'expensive' => price_gt = q3.
    Also handle 'under $xxxx' if needed.
    """
    text_lower = query_text.lower()

    price_lt = None
    price_gt = None

    # cheap synonyms => price < q1
    cheap_syns = ["cheap", "low cost", "budget friendly", "budget-friendly"]
    if any(syn in text_lower for syn in cheap_syns):
        price_lt = q1

    # 'affordable' => price < q2
    if "affordable" in text_lower:
        if (price_lt is None) or (q2 < price_lt):
            price_lt = q2

    # 'expensive', 'premium', etc => price > q3
    exp_syns = ["expensive", "premium", "high cost", "high-end"]
    if any(syn in text_lower for syn in exp_syns):
        price_gt = q3

    # optional: handle 'under $xxxx' => explicit numeric
    import re
    match = re.search(r"under\s*\$?(\d+)", text_lower)
    if match:
        val = float(match.group(1))
        if price_lt is None or val < price_lt:
            price_lt = val

    return price_lt, price_gt

@router.post("/search")
def search_endpoint(query: SearchQuery):
    if not query.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")
    logger.info(f"Received query: {query.query}")

    # 6) Decide product vs. service (your existing logic)
    if "service" in query.query.lower():
        df = df_services
        index = serv_index
        intent = "service"
    else:
        df = df_products
        index = prod_index
        intent = "product"

    # 7) Parse synonyms for cost
    price_lt, price_gt = parse_price_synonyms(query.query)

    # 8) Encode the query
    query_emb = model.encode([query.query], convert_to_numpy=True).astype('float32')
    faiss.normalize_L2(query_emb)

    # 9) Perform FAISS search
    distances, indices = index.search(query_emb, query.top_k)
    results = []
    for dist, idx in zip(distances[0], indices[0]):
        if idx == -1:
            continue
        record = df.iloc[idx]

        def nan_to_none(x):
            return None if pd.isna(x) else x

        product_id = nan_to_none(record.get("product_id"))
        product_name = nan_to_none(record.get("product_name"))
        service_id = nan_to_none(record.get("service_id"))
        service_name = nan_to_none(record.get("service_name"))
        description = nan_to_none(record.get("description"))
        rtype = nan_to_none(record.get("type"))
        price = nan_to_none(record.get("price"))
        rating = nan_to_none(record.get("rating"))
        price_range = nan_to_none(record.get("price_range"))
        location = nan_to_none(record.get("supplier_location"))

        # Adjust fields if product vs. service
        if intent == "product":
            service_id = None
            service_name = None
            price_range = None
        else:
            product_id = None
            product_name = None
            price = None
            rating = None

        # Build final object
        res = SearchResult(
            product_id=product_id,
            product_name=product_name,
            service_id=service_id,
            service_name=service_name,
            description=description if description else "No description",
            type=rtype if rtype else "unknown",
            price=price if price is not None else 0.0,
            rating=rating if rating is not None else 0.0,
            price_range=price_range,
            location=location,
            similarity=float(dist)
        )
        results.append(res)

    # 10) Filter by dynamic thresholds
    # e.g. keep only price < price_lt if price_lt is set
    # or keep only price > price_gt if price_gt is set
    filtered_results = []
    for r in results:
        if intent == "product":
            # we have numeric price
            if price_gt is not None and r.price <= price_gt:
                continue  # skip if not above q3
            if price_lt is not None and r.price >= price_lt:
                continue  # skip if not below q1 or q2
        else:
            # maybe no price filter for services
            pass
        filtered_results.append(r)

    # 11) (Optional) re-rank by ascending or descending price
    # if 'cheap' => ascending, if 'expensive' => descending
    text_lower = query.query.lower()
    if any(syn in text_lower for syn in ["cheap", "affordable", "low cost"]):
        filtered_results = sorted(filtered_results, key=lambda x: x.price)
    elif any(syn in text_lower for syn in ["expensive", "premium", "high cost", "high-end"]):
        filtered_results = sorted(filtered_results, key=lambda x: x.price, reverse=True)

    # 12) Return final top_k
    final = filtered_results[: query.top_k]

    logger.info(f"Returning {len(final)} results for query: {query.query}")
    return JSONResponse(content={"results": [r.dict(exclude_none=True) for r in final]})
