#!/usr/bin/env python3
# coding: utf-8

"""
This is example based on tutorial for Latent Dirichlet Anaysis
"""

# ======================================================================================
#                                     IMPORTS
# ======================================================================================
from __future__ import annotations

import sys
from collections import Counter
from typing import Dict, List, Optional, Set

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


class BagOfWords:
    bow: np.ndarray
    words: Dict[str, int]

    def __init__(self, bow: np.ndarray, words: Dict[str, int]) -> None:
        self.bow = bow
        self.words = words

    @staticmethod
    def build_from_dataset(data: pd.DataFrame, token_column: str) -> BagOfWords:
        words = BagOfWords._pepare_word_dict(data, token_column)
        bow = np.zeros((data.shape[0], len(words.keys())))
        for idx, row in data.iterrows():
            word_count = Counter(row[token_column])
            for word, count in word_count.most_common():
                bow[idx, words[word]] = count
        return BagOfWords(bow, words)

    @staticmethod
    def _pepare_word_dict(data: pd.DataFrame, token_column: str) -> Dict[str, int]:
        word_list: List[str] = []
        for _, row in data.iterrows():
            word_list.extend(row[token_column])
        word_set = set(word_list)
        words: Dict[str, int] = {word: idx for idx, word in enumerate(word_set)}
        return words


# ======================================================================================
#                                       MAIN
# ======================================================================================
def main() -> None:
    if len(sys.argv) != 2:
        print("Correct usage: lda.py PATH_TO_CSV")
        sys.exit()
    documents = read_documents(sys.argv[1])
    extractor = MeaningfulWordExtractor()
    documents = documents[:10]
    documents["tokens"] = documents["text"].apply(extractor)
    print(documents.head())
    BagOfWords.build_from_dataset(documents, "tokens")


if __name__ == "__main__":
    main()
