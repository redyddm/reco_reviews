# Recommandation de reviews

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

## Project Organization

```
├── LICENSE            <- Open-source license if one is chosen
├── README.md          <- The top-level README for developers using this project.
├── data
│   ├── external       <- Data from third party sources.
│   ├── interim        <- Intermediate data that has been transformed.
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
│
├── models             <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks          <- Jupyter notebooks. 
│
├── pyproject.toml     <- Project configuration file with package metadata for 
│                         src and configuration for tools like black
│
├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
│                         generated with `pip freeze > requirements.txt`
│
└── src                 <- Source code for use in this project.
    │
    ├── __init__.py             <- Makes reco_r a Python module
    │
    ├── config.py               <- Store useful variables and configuration
    │
    ├── dataset.py              <- Scripts to download or generate data
    │
    ├── features.py             <- Code to create features for modeling
    │
    ├── preprocessing.py        <- Store preprocessing functions
    │
    ├── recommender.py          <- Store recommendation function
    │
    ├── utils.py                <- Store usual functions (mean of embeddings for example)
    │
    ├── modeling                
        ├── __init__.py 
        ├── predict.py          <- Code to run model inference with trained models          
        └── train.py            <- Code to train models

```

--------

---

## Installation

Clone the repository and install the required packages:

```bash
git clone https://github.com/redyddm/projet_ter.git
cd projet_ter
pip install -r requirements.txt


