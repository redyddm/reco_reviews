from pathlib import Path

from loguru import logger
from tqdm import tqdm
import pandas as pd
import numpy as np
import typer
import gensim
import ast
import pickle

from src.config import MODELS_DIR, PROCESSED_DATA_DIR
from src.utils import get_text_vector

app = typer.Typer()

@app.command("run")
def main(
    features_path: Path = PROCESSED_DATA_DIR / "features.csv",
    embeddings_path : Path = PROCESSED_DATA_DIR / "embeddings.npy",
    model_path: Path = MODELS_DIR / "model.pkl",
):
    features=pd.read_csv(features_path)
    model = gensim.models.Word2Vec(vector_size=300, window= 5 , min_count= 2)

    features['tokens']=features['tokens'].apply(ast.literal_eval)

    logger.info("Training some model...")
    model.build_vocab(features['tokens'])
    model.train(features['tokens'], total_examples=model.corpus_count, epochs=model.epochs)
    logger.success("Modeling training complete.")

    logger.info(f"Saving the model to {model_path}.")
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    logger.success("Model saved.")

    logger.info("Calculating the embeddings...")

    embeddings = np.vstack([
        get_text_vector(tokens, model) for tokens in tqdm(features['tokens'], desc="Calcul des embeddings")
    ])

    logger.info(f"Saving the embeddings to {embeddings_path}")
    np.save(embeddings_path, embeddings)
    logger.success("Embeddings saved.")

if __name__ == "__main__":
    app()
