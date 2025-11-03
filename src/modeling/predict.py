from pathlib import Path

from loguru import logger
import pandas as pd
import numpy as np
import typer

from src.config import PROCESSED_DATA_DIR
from src.recommender import recommandation_reviews

app = typer.Typer()


@app.command()
def main(
    dataset_path = PROCESSED_DATA_DIR / "content_dataset.csv",
    features_path: Path = PROCESSED_DATA_DIR / "features.csv",
    embeddings_path: Path = PROCESSED_DATA_DIR /"embeddings.npy",
    k: int = typer.Option(5, prompt="Number of recommendations"),
    index: int = typer.Option(0, prompt="Review index")
):
    logger.info("Performing inference for model...")

    data = pd.read_csv(dataset_path)
    features = pd.read_csv(features_path)
    embeddings = np.load(embeddings_path)

    data['review_embedding'] = embeddings.tolist()

    top_reviews, sim_scores = recommandation_reviews(index, data, k)

    logger.info(f"Review en cours de lecture :\n {data['review_content'][index]}")

    logger.info(f"{top_reviews[['id', 'review_content']]}")
    
    logger.success("Inference complete.")

if __name__ == "__main__":
    app()
