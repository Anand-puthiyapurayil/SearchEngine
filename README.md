# ***Semantic Search Engine for E-commerce Products and Services***

## ***Project Description***
This repository implements a semantic search engine designed to distinguish between e-commerce products and services, leveraging the following technologies:

- **Hugging Face Embeddings**: Utilizing a fine-tuned Sentence-BERT model for embedding generation.
- **FAISS**: For efficient vector storage and retrieval.
- **FastAPI**: For creating a RESTful API.
- **Loguru**: For efficient logging and monitoring.

The application demonstrates batch ingestion of products and services into FAISS, storing metadata in each document (e.g., whether it’s a “product” or “service”), and producing a ranked list of search results based on semantic similarity.

---

## ***Table of Contents***
1. [***Features***](#features)
2. [***Installation***](#installation)
3. [***Usage***](#usage)
4. [***Data Flow***](#data-flow)
5. [***Customization***](#customization)
6. [***License***](#license)

---

## ***Features***

- **Batch Data Ingestion**: Efficiently processes product and service datasets.
- **FAISS Vector Search**: Uses FAISS for fast similarity search across large datasets.
- **Semantic Search API**: Provides an API endpoint for performing semantic search queries.
- **Metadata Handling**: Each document includes metadata fields (e.g., `type`, `name`, `description`, `price`) for displaying relevant product/service information in the search results.

---

## ***Installation***

### ***Install Dependencies***

1. Clone the repository:
   ```bash
   git clone https://github.com/your-repo/semantic-search-engine.git
   cd semantic-search-engine
   ```

2. Create a virtual environment and activate it:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Linux/Mac
   venv\Scripts\activate    # On Windows
   ```

3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```


Ensure the following packages are included in your `requirements.txt`:

- `fastapi`
- `uvicorn`
- `faiss-cpu`
- `loguru`
- `pandas`
- `sentence-transformers`
- `pyyaml`

---
## **Create two folders  models & mlruns

## ***Set Up Environment Variables***

Create a `config.yaml` file in the project root containing:

```yaml
data:
  products_csv: data/raw/products.csv
  services_csv: data/raw/services.csv
  combined_dataset: data/processed/combined_dataset.csv
faiss:
  products_index: data/indices/products_index.faiss
  services_index: data/indices/services_index.faiss
embeddings:
  products_embeddings: data/embeddings/products_embeddings.npy
  services_embeddings: data/embeddings/services_embeddings.npy
model:
  base: sentence-transformers/all-MiniLM-L6-v2
  fine_tuned: models/fine-tuned-sbert
  batch_size: 8
  num_epochs: 3
```

---

## ***Usage***

### ***Prepare Your Data***
Place your raw product and service data CSV files in the `data/raw/` directory as specified in `config.yaml`.

### ***Run Data Preprocessing***
```bash
python scripts/preprocess_data.py
```

### ***Generate Embeddings***
```bash
python scripts/generate_embeddings.py
```

### ***Create FAISS Indices***
```bash
python scripts/create_faiss_index.py
```

### ***Run the API Server***
```bash
uvicorn main:app --reload
```

---

## ***Perform a Search***
Use the `/api/search` endpoint to perform a semantic search query.

### ***Search Request Example:***
```json
{
  "query": "I want to pump",
  "top_k": 10
}
```

### ***Search Response Example:***
```json
{
  "results": [
    {
      "product_id": "123",
      "product_name": "Water Pump",
      "description": "High-efficiency water pump.",
      "type": "product",
      "price": 100.0,
      "rating": 4.5,
      "location": "New York",
      "similarity": 0.95
    },
    {
      "service_id": "456",
      "service_name": "Pump Repair Service",
      "description": "Repair services for water pumps.",
      "type": "service",
      "price_range": "$50-$100",
      "location": "Los Angeles",
      "similarity": 0.92
    }
  ]
}
```

---

## ***Data Flow***

### **Preprocessing**
- Script: `preprocess_data.py`
- Function: `preprocess_data(df_products, df_services)`
  - Adds a `type` column to distinguish between products and services.
  - Fills missing values.
  - Combines metadata fields into a single `prefixed_text` column.

### **Embedding Generation**
- Script: `generate_embeddings.py`
- Function: Generates embeddings using a fine-tuned Sentence-BERT model and saves them as `.npy` files.

### **FAISS Index Creation**
- Script: `create_faiss_index.py`
- Function: Creates FAISS indices from the generated embeddings and saves them for future retrieval.

### **Search API**
- Script: `search.py`
- Endpoint: `/api/search`
  - Receives a search query.
  - Determines whether the query relates to products or services.
  - Performs a vector search using FAISS.
  - Returns the top-k results with metadata and similarity scores.

---

## ***Customization***

- **Model Configuration**: Adjust the model path, batch size, and number of epochs in `config.yaml`.
- **API Endpoints**: Modify `search.py` to add more endpoints or adjust query handling.
- **Data Preprocessing**: Customize `preprocess_data.py` to handle additional fields in your data.

---

## ***License***
This project is licensed under the MIT License. See the `LICENSE` file for more details.

