from pathlib import Path

from loguru import logger
from tqdm import tqdm
import typer
import pandas as pd
import gensim
import os

from src.config import PROCESSED_DATA_DIR

app = typer.Typer()


@app.command()
def main(
    input_path: Path = PROCESSED_DATA_DIR / "content_dataset.csv",
    output_path: Path = PROCESSED_DATA_DIR / "features.csv",
):
    
    data = pd.read_csv(input_path)
    logger.info("Generating features from dataset...")
    
    tqdm.pandas(desc='Création des tokens')
    tokens = data['review_content'].progress_apply(gensim.utils.simple_preprocess)

    features = pd.DataFrame({
        'id' : data['id'],
        'tokens' : tokens
    })

    logger.success("Features generation complete.")

    logger.info(f"Saving features to {output_path}.")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    features.to_csv(output_path)

    logger.success("Features saved.")

if __name__ == "__main__":
    app()
