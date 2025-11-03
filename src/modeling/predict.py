from pathlib import Path

from loguru import logger
import pandas as pd
import numpy as np
import typer

from src.config import PROCESSED_DATA_DIR
from src.recommender import recommandation_reviews

app = typer.Typer()


@app.command("run")
def main(
    dataset_path = PROCESSED_DATA_DIR / "content_dataset.csv",
    embeddings_path: Path = PROCESSED_DATA_DIR /"embeddings.npy",
    k: int = typer.Option(5, prompt="Number of recommendations"),
    index: int = typer.Option(0, prompt="Review index")
):
    logger.info("Performing inference for model...")

    data = pd.read_csv(dataset_path)
    embeddings = np.load(embeddings_path)

    data['review_embedding'] = embeddings.tolist()

    review_id = data['id'][0]

    top_reviews, sim_scores = recommandation_reviews(review_id, data, k)

    logger.info(f"Review en cours de lecture :\n {data['review_content'][index]}")

    logger.info(f"{top_reviews[['id', 'review_content']]}")
    
    logger.success("Inference complete.")

if __name__ == "__main__":
    app()
