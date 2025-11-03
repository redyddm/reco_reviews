import re
import html

def text_cleaning(s):
    """ Fonction de nettoyage de texte.
    Args:
        s (str) : texte à nettoyer
    """
    res = html.unescape(s)
    res = re.sub("<.*?>", "", res)
    res = re.sub("[\n\t\r]", "", res)
    
    return res

def preprocess_data(df):
    """ Fonction de prétraitement des données.
    Args:
        df (pd.DataFrame) : dataframe avec des reviews
    """
    df_preprocessed = df.copy()

    df_preprocessed.dropna(subset="review_content", inplace=True)
    df_preprocessed.reset_index(drop=True, inplace=True)

    df_preprocessed['review_content'] = df_preprocessed['review_content'].apply(text_cleaning)

    return df_preprocessed