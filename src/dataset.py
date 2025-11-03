from pathlib import Path

from loguru import logger
import typer
import os
import pandas as pd

from src.config import PROCESSED_DATA_DIR, RAW_DATA_DIR
from src.preprocessing import preprocess_data

app = typer.Typer()

@app.command()
def main(
    input_path: list[Path] = [RAW_DATA_DIR / "interstellar_critique.csv", RAW_DATA_DIR / "fightclub_critiques.csv"],
    output_path: Path = PROCESSED_DATA_DIR / "content_dataset.csv",
):
    logger.info("Processing dataset...")

    data = pd.DataFrame()

    for input in input_path:
        movie_title = str.split(input.name, '_')[0]

        df = pd.read_csv(input)
        df = preprocess_data(df)

        df['movie_title'] = movie_title

        data = pd.concat([data, df], axis=0)
        data.reset_index(drop=True, inplace=True)
    
    logger.success("Processing dataset complete.")

    logger.info(f"Saving processed dataset to {output_path}.")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    data.to_csv(output_path)

    logger.success("Processed dataset saved.")

if __name__ == "__main__":
    app()
