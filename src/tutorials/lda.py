#!/usr/bin/env python3
# coding: utf-8

"""
This is example based on tutorial for Latent Dirichlet Anaysis
"""

# ======================================================================================
#                                     IMPORTS
# ======================================================================================
import sys
from typing import List, Optional, Set

import nltk
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
    print(data.info())

    documents = pd.DataFrame({"idx": data.index, "text": data["headline_text"]})
    print(SEPARATOR_LINE)
    print("After dataframe reformatting")
    print(documents.info())
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


if __name__ == "__main__":
    main()
