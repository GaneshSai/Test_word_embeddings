import pandas as pd
import numpy as np
from nltk.corpus import stopwords
import nltk
import re
from nltk.tokenize import word_tokenize
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import euclidean_distances
from gensim.models.doc2vec import Doc2Vec, TaggedDocument


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
    # removing stopwords and cleaning the text...
    stop_words_l = stopwords.words("english")
    documents_df["documents_cleaned"] = documents_df.similariy_finder.apply(
        lambda x: " ".join(
            re.sub(r"[^a-zA-Z]", " ", w).lower()
            for w in x.split()
            if re.sub(r"[^a-zA-Z]", " ", w).lower() not in stop_words_l
        )
    )
    tagged_data = [
        TaggedDocument(words=word_tokenize(doc), tags=[i])
        for i, doc in enumerate(documents_df.documents_cleaned)
    ]
    model_d2v = Doc2Vec(vector_size=100, alpha=0.025, min_count=1)
    model_d2v.build_vocab(tagged_data)
    for epoch in range(100):
        model_d2v.train(
            tagged_data, total_examples=model_d2v.corpus_count, epochs=model_d2v.epochs
        )
    document_embeddings = np.zeros((documents_df.shape[0], 100))

    for i in range(len(document_embeddings)):
        document_embeddings[i] = model_d2v.dv[i]

    pairwise_similarities=cosine_similarity(document_embeddings)
    pairwise_differences=euclidean_distances(document_embeddings)
    most_similar(0,pairwise_similarities,'Cosine Similarity')
    most_similar(0,pairwise_similarities,'Euclidean Distance')
