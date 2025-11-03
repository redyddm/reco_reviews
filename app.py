import streamlit as st
import pandas as pd
import numpy as np

from src.config import PROCESSED_DATA_DIR
from src.recommender import recommandation_reviews

@st.cache_data
def load_reviews():
    return pd.read_csv(PROCESSED_DATA_DIR / "content_dataset.csv")

@st.cache_data
def load_embeddings():
    return np.load(PROCESSED_DATA_DIR / "embeddings.npy")

reviews = load_reviews()
embeddings = load_embeddings()

st.title("Recommandation de reviews")

reviews['review_embedding'] = embeddings.tolist()

with st.form("reco"):
    id = st.selectbox("Review ID", options=reviews['id'])
    k = st.slider("Nombre de recommandations", 1, 20, 5)

    reco = st.form_submit_button("Recommandation")

if reco:

    review_initial = reviews.loc[reviews['id']==id, 'review_content'].iloc[0]
    preview = review_initial[:100] + '...' if len(review_initial) > 100 else review_initial

    st.write("## Review choisie")
    st.write(preview)
    with st.expander("review complète"):
        st.write(review_initial)

    top_reviews, sim_scores = recommandation_reviews(id, reviews, k)


    st.write("## Reviews recommandées")
    i=1
    for review in top_reviews['review_content']:
        preview = review[:100] + '...'
        st.write(preview)
        with st.expander("review complète"):
            st.write(review)