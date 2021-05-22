import seaborn as sns
from sklearn.metrics import pairwise
from sentence_transformers import SentenceTransformer
import tensorflow as tf
import tensorflow_hub as hub
from text_cleaning import *
import pandas as pd
import numpy as np
from nltk.corpus import stopwords
import nltk
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import euclidean_distances


sbert_model = SentenceTransformer("bert-base-nli-mean-tokens") # Model being loaded...


def most_similar(doc_id, similarity_matrix, matrix):
    print(f"Similar Documents using {matrix}:")
    if matrix == "Cosine Similarity":
        similar_ix = np.argsort(similarity_matrix[doc_id])[::-1]
        print(similar_ix)
    elif matrix == "Euclidean Distance":
        similar_ix = np.argsort(similarity_matrix[doc_id])
        print(similar_ix)
    for ix in similar_ix:
        if ix == doc_id:
            continue
        print("\n")
        print(f"{matrix} : {similarity_matrix[doc_id][ix]}")


if __name__ == "__main__":

	# Text file which needs to be compared being read.
    f = open(
        "/home/ganesh/Desktop/Crawler/Information_Security1/1.txt",
        "r",
        encoding="utf-8",
    )
    text = f.read()
    text_compared_with = "information security"
    similariy_finder = [text, text_compared_with]
    documents_df = pd.DataFrame(similariy_finder, columns=["similariy_finder"])
    #removing stopwords and cleaning the text...
    stop_words_l = stopwords.words("english")
    documents_df["documents_cleaned"] = documents_df.similariy_finder.apply(
        lambda x: " ".join(
            re.sub(r"[^a-zA-Z]", " ", w).lower()
            for w in x.split()
            if re.sub(r"[^a-zA-Z]", " ", w).lower() not in stop_words_l
        )
    )
    #creating vector using the loaded model...
    document_embeddings = sbert_model.encode(documents_df["documents_cleaned"])
    pairwise_similarities = cosine_similarity(document_embeddings)
    pairwise_differences = euclidean_distances(document_embeddings)
    most_similar(0, pairwise_similarities, "Cosine Similarity")
    most_similar(0, pairwise_similarities, "Euclidean Distance")


# preprocess = hub.load("https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/1")
# bert = hub.load("https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/3")
