import re
import html

def text_cleaning(s):
    res = html.unescape(s)
    res = re.sub("<.*?>", "", res)
    res = re.sub("[\n\t\r]", "", res)
    
    return res

def preprocess_data(df):
    df_preprocessed = df.copy()

    df_preprocessed.dropna(subset="review_content", inplace=True)
    df_preprocessed.reset_index(drop=True, inplace=True)

    df_preprocessed['review_content'] = df_preprocessed['review_content'].apply(text_cleaning)

    return df_preprocessed