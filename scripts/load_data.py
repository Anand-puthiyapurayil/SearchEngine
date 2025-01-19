import pandas as pd
import yaml
from loguru import logger

logger.add("logs/load_data.log", rotation="1 day", compression="zip")

def load_datasets(products_csv: str, services_csv: str, **kwargs):
    logger.info(f"Loading products from {products_csv}")
    df_products = pd.read_csv(products_csv, **kwargs)
    logger.info(f"Products loaded: {df_products.shape}")

    logger.info(f"Loading services from {services_csv}")
    df_services = pd.read_csv(services_csv, **kwargs)
    logger.info(f"Services loaded: {df_services.shape}")

    return df_products, df_services

if __name__ == "__main__":
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    df_products, df_services = load_datasets(config['data']['products_csv'], config['data']['services_csv'])
    logger.info("Data loading complete.")
