import yaml
import pandas as pd
import random
import torch
import mlflow
from loguru import logger
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader
from datetime import datetime
import os

logger.add("logs/fine_tune_triplet.log", rotation="1 day", compression="zip")

def set_seed(seed=42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def create_triplet_examples(df_products, df_services, max_examples=10):
    """
    Create triplet examples where each triplet consists of (anchor, positive, negative)
    all belonging to the same category (either all products or all services).

    Parameters:
    - df_products (pd.DataFrame): DataFrame containing product data.
    - df_services (pd.DataFrame): DataFrame containing service data.
    - max_examples (int): Maximum number of triplets to create per category.

    Returns:
    - List[InputExample]: List of triplet examples for fine-tuning.
    """
    examples = []
    
    # --- Create Triplets for Products ---
    if len(df_products) < 3:
        logger.warning("Not enough products to create triplet examples.")
    else:
        # Sample anchors from products
        products_sample = df_products.sample(n=min(len(df_products), max_examples), random_state=42)
        for _, anchor_prod in products_sample.iterrows():
            anchor_text = anchor_prod['prefixed_text']
            
            # Sample positive (another product)
            positive = df_products.sample(1, random_state=random.randint(0, 10000)).iloc[0]['prefixed_text']
            # Ensure positive is not the same as anchor
            while positive == anchor_text:
                positive = df_products.sample(1, random_state=random.randint(0, 10000)).iloc[0]['prefixed_text']
            
            # Sample negative (different product)
            negative = df_products.sample(1, random_state=random.randint(0, 10000)).iloc[0]['prefixed_text']
            # Ensure negative is distinct from anchor and positive
            while negative == anchor_text or negative == positive:
                negative = df_products.sample(1, random_state=random.randint(0, 10000)).iloc[0]['prefixed_text']
            
            # Append the triplet
            examples.append(InputExample(texts=[anchor_text, positive, negative]))
        
        logger.info(f"Created {len(products_sample)} triplet examples for category: 'product'")
    
    # --- Create Triplets for Services ---
    if len(df_services) < 3:
        logger.warning("Not enough services to create triplet examples.")
    else:
        # Sample anchors from services
        services_sample = df_services.sample(n=min(len(df_services), max_examples), random_state=42)
        for _, anchor_serv in services_sample.iterrows():
            anchor_text = anchor_serv['prefixed_text']
            
            # Sample positive (another service)
            positive = df_services.sample(1, random_state=random.randint(0, 10000)).iloc[0]['prefixed_text']
            # Ensure positive is not the same as anchor
            while positive == anchor_text:
                positive = df_services.sample(1, random_state=random.randint(0, 10000)).iloc[0]['prefixed_text']
            
            # Sample negative (different service)
            negative = df_services.sample(1, random_state=random.randint(0, 10000)).iloc[0]['prefixed_text']
            # Ensure negative is distinct from anchor and positive
            while negative == anchor_text or negative == positive:
                negative = df_services.sample(1, random_state=random.randint(0, 10000)).iloc[0]['prefixed_text']
            
            # Append the triplet
            examples.append(InputExample(texts=[anchor_text, positive, negative]))
        
        logger.info(f"Created {len(services_sample)} triplet examples for category: 'service'")
    
    logger.info(f"Total triplet examples created: {len(examples)}")
    return examples
if __name__ == "__main__":
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    fine_tuned_path = f"{config['model']['fine_tuned']}_{timestamp}"

    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment("ecommerce_semantic_search")

    df = pd.read_csv(config['data']['combined_dataset'])
    df_products = df[df['type'] == 'product'].reset_index(drop=True)
    df_services = df[df['type'] == 'service'].reset_index(drop=True)

    examples = create_triplet_examples(df_products, df_services)
    set_seed()

    model_name = config['model']['base']
    num_epochs = config['model']['num_epochs']
    batch_size = config['model']['batch_size']

    model = SentenceTransformer(model_name)
    if not examples:
        logger.info("No examples for fine-tuning; saving base model.")
        os.makedirs(fine_tuned_path, exist_ok=True)
        model.save(fine_tuned_path)
    else:
        with mlflow.start_run():
            mlflow.log_param("fine_tuning_type", "triplet")
            mlflow.log_param("model_name", model_name)
            mlflow.log_param("num_epochs", num_epochs)
            mlflow.log_param("batch_size", batch_size)

            train_dataloader = DataLoader(examples, shuffle=True, batch_size=batch_size)
            train_loss = losses.TripletLoss(model=model)

            logger.info("Starting fine-tuning...")
            model.fit(
                train_objectives=[(train_dataloader, train_loss)],
                epochs=num_epochs,
                show_progress_bar=True
            )
            logger.info("Triplet fine-tuning completed.")

            os.makedirs(fine_tuned_path, exist_ok=True)
            model.save(fine_tuned_path)
            logger.info(f"Model saved at {fine_tuned_path}")

            mlflow.pytorch.log_model(model, "model")
            logger.info("Model logged to MLflow.")
