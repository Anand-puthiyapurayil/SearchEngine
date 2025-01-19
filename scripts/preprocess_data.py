import pandas as pd
import yaml
from loguru import logger

logger.add("logs/preprocess_data.log", rotation="1 day", compression="zip")

def preprocess_data(df_products, df_services):
    logger.info("Preprocessing data...")

    df_products['type'] = 'product'
    df_services['type'] = 'service'

    # Ensure columns exist or fill them as needed. Adjust these lines based on your actual CSV schema.
    # For products: assume columns: product_name, description, price, rating, supplier_location
    df_products['price'] = df_products['price'].fillna(0.0)
    df_products['rating'] = df_products['rating'].fillna(0.0)
    df_products['supplier_location'] = df_products['supplier_location'].fillna('Unknown')
    df_products['product_name'] = df_products['product_name'].fillna('UnknownProduct')
    df_products['description'] = df_products['description'].fillna('')

    # For services: assume columns: service_name, description, price_range, supplier_location, supplier_name
    df_services['price_range'] = df_services['price_range'].fillna('Unspecified')
    df_services['supplier_location'] = df_services['supplier_location'].fillna('Unknown')
    df_services['supplier_name'] = df_services['supplier_name'].fillna('Unknown')
    df_services['service_name'] = df_services['service_name'].fillna('UnknownService')
    df_services['description'] = df_services['description'].fillna('')

    # Incorporate metadata into prefixed_text for products
    df_products['prefixed_text'] = (
        df_products['product_name'] + " " +
        df_products['description'] + " " +
        "Price:" + df_products['price'].astype(str) + " " +
        "Rating:" + df_products['rating'].astype(str) + " " +
        "Location:" + df_products['supplier_location']
    )

    # For services
    df_services['prefixed_text'] = (
        df_services['service_name'] + " " +
        df_services['description'] + " " +
        "PriceRange:" + df_services['price_range'] + " " +
        "Location:" + df_services['supplier_location'] + " " +
        "Supplier:" + df_services['supplier_name']
    )

    df_combined = pd.concat([df_products, df_services], ignore_index=True)
    df_combined.dropna(subset=['prefixed_text'], inplace=True)
    logger.info(f"Combined dataset shape: {df_combined.shape}")
    return df_combined

if __name__ == "__main__":
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    df_products = pd.read_csv(config['data']['products_csv'])
    df_services = pd.read_csv(config['data']['services_csv'])
    df_combined = preprocess_data(df_products, df_services)
    df_combined.to_csv(config['data']['combined_dataset'], index=False)
    logger.info("Preprocessing completed and combined dataset saved.")
