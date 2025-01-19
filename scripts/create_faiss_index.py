import yaml
import faiss
import numpy as np
from loguru import logger
import os

logger.add("logs/create_faiss_index.log", rotation="1 day", compression="zip")

if __name__ == "__main__":
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    logger.info("Creating FAISS indices...")

    products_embeddings_path = config['embeddings']['products_embeddings']
    services_embeddings_path = config['embeddings']['services_embeddings']

    if not os.path.isfile(products_embeddings_path) or not os.path.isfile(services_embeddings_path):
        raise FileNotFoundError("Embeddings not found. Run generate_embeddings.py first.")

    products_embeddings = np.load(products_embeddings_path)
    services_embeddings = np.load(services_embeddings_path)
    dimension = products_embeddings.shape[1]

    index_type = config['faiss'].get('index_type', 'IndexFlatIP')
    if index_type == "IndexFlatIP":
        prod_index = faiss.IndexFlatIP(dimension)
        serv_index = faiss.IndexFlatIP(dimension)
    else:
        prod_index = faiss.IndexFlatL2(dimension)
        serv_index = faiss.IndexFlatL2(dimension)

    prod_index.add(products_embeddings)
    serv_index.add(services_embeddings)

    faiss.write_index(prod_index, config['faiss']['products_index'])
    faiss.write_index(serv_index, config['faiss']['services_index'])
    logger.info("FAISS indices created.")
