#!/usr/bin/env python3
# coding: utf-8

"""
This is example based on tutorial for Latent Dirichlet Anaysis
"""

# ======================================================================================
#                                     IMPORTS
# ======================================================================================
from __future__ import annotations

import argparse
import sys
from collections import Counter
from typing import Dict, List, Optional, Set

import gensim
import nltk
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# ======================================================================================
#                                     CONSTANS
# ======================================================================================
SEPARATOR_LINE: str = "".join(["="] * 88)

# ======================================================================================
#                                     HELPERS
# ======================================================================================


def read_documents(f_name: str) -> pd.DataFrame:
    data = pd.read_csv(f_name)
    print(SEPARATOR_LINE)
    print(f"Read CSV file at {sys.argv[1]}")
    data.info()

    documents = pd.DataFrame({"idx": data.index, "text": data["headline_text"]})
    print(SEPARATOR_LINE)
    print("After dataframe reformatting")
    documents.info()
    print(documents.head())
    return documents


# ======================================================================================
#                                   NLP CLASSES
# ======================================================================================
class MeaningfulWordExtractor:
    stopwords: Set[str]
    min_len: int
    lemmatizer: WordNetLemmatizer

    def __init__(self, extra_stopwords: List[str] = [], min_len: int = 3):
        self.stopwords = set(stopwords.words("english"))
        self.stopwords.union(extra_stopwords)
        self.min_len = min_len
        self.lemmatizer = WordNetLemmatizer()

    def extract(self, txt: str) -> List[str]:
        txt = txt.lower()
        tokens = word_tokenize(txt)
        meaningful_words: List[str] = []
        for token in tokens:
            if len(token) < self.min_len or token in self.stopwords:
                continue
            m_word = self.lemmatizer.lemmatize(token, pos="v")
            meaningful_words.append(m_word)
        return meaningful_words

    def __call__(self, txt: str) -> List[str]:
        return self.extract(txt)


# ======================================================================================
#                                  NLP FUNCTIONS
# ======================================================================================


def prepare_dictionary(tokens: pd.Series) -> gensim.corpora.Dictionary:
    dictionary = gensim.corpora.Dictionary(tokens)
    # Show words with counts from BoW
    count = 0
    for k, v in dictionary.iteritems():
        print(k, v)
        count += 1
        if count > 10:
            break
    # dictionary.filter_extremes(no_below=15, no_above=0.5, keep_n=100000)
    return dictionary


# ======================================================================================
#                                       MAIN
# ======================================================================================
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("data_path", help="Path to csv file")
    parser.add_argument(
        "-r", "--rows", type=int, default=0, help="Maximum number of rows"
    )
    args = parser.parse_args()
    documents = read_documents(args.data_path)
    if args.rows > 0:
        documents = documents[: args.rows]
    extractor = MeaningfulWordExtractor()
    documents["tokens"] = documents["text"].apply(extractor)
    print(documents.head())
    dictionary = prepare_dictionary(documents["tokens"])
    print(dictionary)
    bow_corpus = [dictionary.doc2bow(doc) for doc in documents["tokens"]]
    bow_doc_4 = bow_corpus[4]
    for i in range(len(bow_doc_4)):
        print(
            'Word {} ("{}") appears {} time.'.format(
                bow_doc_4[i][0], dictionary[bow_doc_4[i][0]], bow_doc_4[i][1]
            )
        )


if __name__ == "__main__":
    main()
