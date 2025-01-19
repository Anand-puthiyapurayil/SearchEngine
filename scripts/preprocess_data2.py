import numpy as np
import yaml
import pandas as pd
from loguru import logger

# Configure logger
logger.add("logs/preprocess_data.log", rotation="1 day", compression="zip")

def preprocess_data(df_products, df_services):
    logger.info("Preprocessing data...")

    # 1) Mark types
    df_products['type'] = 'product'
    df_services['type'] = 'service'

    # 2) Basic cleaning
    df_products['price'] = df_products['price'].fillna(0.0)
    df_products['rating'] = df_products['rating'].fillna(0.0)
    df_products['supplier_location'] = df_products['supplier_location'].fillna('unknown').str.lower()
    df_products['product_name'] = df_products['product_name'].fillna('unknownproduct').str.lower()
    df_products['description'] = df_products['description'].fillna('').str.lower()

    df_services['price_range'] = df_services['price_range'].fillna('unspecified').str.lower()
    df_services['supplier_location'] = df_services['supplier_location'].fillna('unknown').str.lower()
    df_services['supplier_name'] = df_services['supplier_name'].fillna('unknown').str.lower()
    df_services['service_name'] = df_services['service_name'].fillna('unknownservice').str.lower()
    df_services['description'] = df_services['description'].fillna('').str.lower()

    # 3) Build prefixed_text
    df_products['prefixed_text'] = (
        df_products['product_name'] + " " +
        df_products['description'] + " " +
        "price:" + df_products['price'].astype(str) + " " +
        "rating:" + df_products['rating'].astype(str) + " " +
        "location:" + df_products['supplier_location']
    ).str.lower()

    df_services['prefixed_text'] = (
        df_services['service_name'] + " " +
        df_services['description'] + " " +
        "pricerange:" + df_services['price_range'] + " " +
        "location:" + df_services['supplier_location'] + " " +
        "supplier:" + df_services['supplier_name']
    ).str.lower()

    # 4) Combine datasets
    df_combined = pd.concat([df_products, df_services], ignore_index=True)
    df_combined.dropna(subset=['prefixed_text'], inplace=True)
    logger.info(f"Combined dataset shape: {df_combined.shape}")

    # 5) Compute price percentiles (only for products)
    df_products_only = df_combined[df_combined['type'] == 'product'].copy()
    prices = df_products_only['price'].dropna().values
    if len(prices) > 0:
        q1 = np.percentile(prices, 25)
        q2 = np.percentile(prices, 50)  # median
        q3 = np.percentile(prices, 75)
        price_percentiles = {
            'q1': float(q1),
            'q2': float(q2),
            'q3': float(q3)
        }
        logger.info(f"Price distribution percentiles: {price_percentiles}")
    else:
        # Fallback if no prices available
        price_percentiles = {'q1': 1000.0, 'q2': 3000.0, 'q3': 7000.0}
        logger.warning("No product prices found, using fallback percentiles.")

    # 6) Return processed data and price percentiles
    return df_combined, price_percentiles

if __name__ == "__main__":
    # Load config file
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    # Load product and service CSV files
    df_products = pd.read_csv(config['data']['products_csv'])
    df_services = pd.read_csv(config['data']['services_csv'])

    # Preprocess data
    df_combined, price_percentiles = preprocess_data(df_products, df_services)

    # Save the combined dataset to CSV
    df_combined.to_csv(config['data']['combined_dataset'], index=False)
    logger.info("Preprocessing completed and combined dataset saved.")

    # Update price percentiles in the config file
    config['price_percentiles'] = price_percentiles
    with open('config.yaml', 'w') as f:
        yaml.safe_dump(config, f)
    logger.info("Updated config.yaml with price_percentiles.")
