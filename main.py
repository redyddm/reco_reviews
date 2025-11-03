import typer
from src import dataset, features
from src.modeling import train, predict

app = typer.Typer()

app.add_typer(dataset.app, name="dataset")
app.add_typer(features.app, name="features")
app.add_typer(train.app, name="train")
app.add_typer(predict.app, name="predict")

if __name__ == "__main__":
    app()
