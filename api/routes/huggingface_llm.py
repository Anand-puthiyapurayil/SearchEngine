from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Optional
import yaml
import faiss
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import pipeline
from loguru import logger

router = APIRouter()

# Pydantic Models
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

# Load config
with open("config.yaml", 'r') as f:
    config = yaml.safe_load(f)

logger.add("logs/search_endpoint.log", rotation="1 day", compression="zip")
logger.info("Loading model and FAISS indices for search...")

# Load FAISS indices
prod_index = faiss.read_index(config['faiss']['products_index'])
serv_index = faiss.read_index(config['faiss']['services_index'])

# Load DataFrames
df_products = pd.read_csv(config['faiss']['products_mapping'])
df_services = pd.read_csv(config['faiss']['services_mapping'])

# Load the fine-tuned SentenceTransformer model
model = SentenceTransformer(config['model']['fine_tuned'])

# Initialize Hugging Face text generation pipeline
text_generator = pipeline("text-generation", model="gpt2")  # Replace with another model if needed

# Price Percentiles from config
price_percentiles = config.get('price_percentiles', {'q1': 2000.0, 'q2': 5000.0, 'q3': 10000.0})
q1, q2, q3 = price_percentiles['q1'], price_percentiles['q2'], price_percentiles['q3']

# Utility Functions
def nan_to_none(x):
    """
    Converts NaN values to None for JSON compatibility.
    """
    return None if pd.isna(x) else x

def parse_price_synonyms(query_text: str):
    """
    Translates synonyms into dynamic thresholds using q1, q2, q3.
    e.g. 'cheap' => price_lt = q1, 'affordable' => price_lt = q2, 'expensive' => price_gt = q3.
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
        if price_lt is None or q2 < price_lt:
            price_lt = q2

    # 'expensive', 'premium', etc => price > q3
    exp_syns = ["expensive", "premium", "high cost", "high-end"]
    if any(syn in text_lower for syn in exp_syns):
        price_gt = q3

    # Handle explicit price ranges (e.g., 'under $500')
    import re
    match = re.search(r"under\s*\$?(\d+)", text_lower)
    if match:
        val = float(match.group(1))
        if price_lt is None or val < price_lt:
            price_lt = val

    return price_lt, price_gt

def generate_llm_response(query: str, results: List[SearchResult]) -> str:
    """
    Generate a human-readable response using Hugging Face Transformers.
    Combines the query with the top retrieved results.
    """
    context = "\n".join(
        f"- {r.type.capitalize()} '{r.product_name or r.service_name}' "
        f"(Price: {r.price or 'N/A'}, Rating: {r.rating or 'N/A'}, Location: {r.location or 'Unknown'})"
        for r in results[:5]  # Limit context to top 5 results for brevity
    )

    prompt = f"""
    You are a helpful assistant for a product and service search engine. 
    Based on the following query and context, generate a concise response:
    
    Query: {query}
    Context:
    {context}
    
    Response:
    """
    response = text_generator(prompt, max_new_tokens=50, temperature=0.7)
    return response[0]["generated_text"].strip()

@router.post("/search")
def search_endpoint(query: SearchQuery):
    if not query.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")
    logger.info(f"Received query: {query.query}")

    # Determine intent: product or service
    if "service" in query.query.lower():
        df = df_services
        index = serv_index
        intent = "service"
    else:
        df = df_products
        index = prod_index
        intent = "product"

    # Parse price/rating thresholds
    price_lt, price_gt = parse_price_synonyms(query.query)

    # Encode the query
    query_emb = model.encode([query.query], convert_to_numpy=True).astype('float32')
    faiss.normalize_L2(query_emb)

    # Perform FAISS search
    distances, indices = index.search(query_emb, query.top_k)
    results = []
    for dist, idx in zip(distances[0], indices[0]):
        if idx == -1:
            continue
        record = df.iloc[idx]
        results.append(SearchResult(
            product_id=nan_to_none(record.get("product_id")),
            product_name=nan_to_none(record.get("product_name")),
            service_id=nan_to_none(record.get("service_id")),
            service_name=nan_to_none(record.get("service_name")),
            description=nan_to_none(record.get("description", "No description")),
            type=nan_to_none(record.get("type", "unknown")),
            price=nan_to_none(record.get("price", 0.0)),
            rating=nan_to_none(record.get("rating", 0.0)),
            price_range=nan_to_none(record.get("price_range")),
            location=nan_to_none(record.get("supplier_location")),
            similarity=float(dist)
        ))

    # Apply filters and re-ranking
    filtered_results = []
    for r in results:
        if intent == "product":
            if price_gt is not None and r.price <= price_gt:
                continue
            if price_lt is not None and r.price >= price_lt:
                continue
        filtered_results.append(r)

    # Optional: Re-rank based on price
    text_lower = query.query.lower()
    if any(syn in text_lower for syn in ["cheap", "affordable", "low cost"]):
        filtered_results = sorted(filtered_results, key=lambda x: x.price)
    elif any(syn in text_lower for syn in ["expensive", "premium", "high cost", "high-end"]):
        filtered_results = sorted(filtered_results, key=lambda x: x.price, reverse=True)

    # Generate LLM-based response
    llm_response = generate_llm_response(query.query, filtered_results[:query.top_k])

    # Return results and LLM response
    logger.info(f"Returning {len(filtered_results)} results for query: {query.query}")
    return JSONResponse(content={
        "llm_response": llm_response,
        "results": [r.dict(exclude_none=True) for r in filtered_results[:query.top_k]]
    })
