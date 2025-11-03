from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def recommandation_reviews(review_id, content_dataset, k=5):

    review_index = np.where(content_dataset['id'] == review_id)[0][0]
    
    movie_title = content_dataset['movie_title'][review_index]
    review_embedding = content_dataset['review_embedding'][review_index]

    sub_content = content_dataset[(content_dataset['movie_title']==movie_title) & (content_dataset.index != review_index)].copy()
    
    sim_scores = cosine_similarity([review_embedding], sub_content['review_embedding'].tolist())[0]
    top_k_index = np.argsort(sim_scores)[-k:][::-1]

    top_k_reviews = sub_content.iloc[top_k_index]

    return top_k_reviews, sim_scores[:k]