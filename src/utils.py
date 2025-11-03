import numpy as np

def get_text_vector(text, model):
    """ Fonction de création d'embeddings de textes.
    Args:
        text (str) : phrase à encoder
        model : modèle NLP (dans notre cas Word2Vec)
    """
    vectors = [model.wv[word] for word in text if word in model.wv]
    if vectors:
        return np.mean(vectors, axis=0)
    else:
        return np.zeros(model.vector_size)