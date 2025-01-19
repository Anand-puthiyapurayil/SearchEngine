import yaml
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
from loguru import logger
import os

logger.add("logs/generate_embeddings.log", rotation="1 day", compression="zip")

if __name__ == "__main__":
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    df = pd.read_csv(config['data']['combined_dataset'])
    df_products = df[df['type'] == 'product'].reset_index(drop=True)
    df_services = df[df['type'] == 'service'].reset_index(drop=True)

    logger.info("Loading fine-tuned model for embeddings.")
    model = SentenceTransformer(config['model']['fine_tuned'])

    logger.info("Generating product embeddings...")
    product_sentences = df_products['prefixed_text'].tolist()
    product_embeddings = model.encode(product_sentences, convert_to_numpy=True).astype('float32')
    faiss.normalize_L2(product_embeddings)
    np.save(config['embeddings']['products_embeddings'], product_embeddings)
    df_products.to_csv(config['faiss']['products_mapping'], index=False)
    logger.info("Product embeddings generated.")

    logger.info("Generating service embeddings...")
    service_sentences = df_services['prefixed_text'].tolist()
    service_embeddings = model.encode(service_sentences, convert_to_numpy=True).astype('float32')
    faiss.normalize_L2(service_embeddings)
    np.save(config['embeddings']['services_embeddings'], service_embeddings)
    df_services.to_csv(config['faiss']['services_mapping'], index=False)
    logger.info("Service embeddings generated.")
