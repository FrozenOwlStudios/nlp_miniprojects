"""
An attempt at implementing a named entity and relation extractor with nltk.

            +-----------------------+                     +-----------+
nl_text --->| SENTENCE SEGMENTATION |---> sentences[] --->| TOKENIZER |---> tokens[][]
            +-----------------------+                     +-----------+        |
                                                                               |
   +---------------------------------------------------------------------------+
   |
   |    +------------+                   +-----------------+
   +--->| POS TAGGER |---> words[][] --->| ENTITY DETECTOR |---> sent_repr[] ---+
        +------------+                   +-----------------+                    |
                                                                                |
   +----------------------------------------------------------------------------+
   |
   |    +---------------------+
   +--->| RELATION DETECTION  |---> sent_repr[]
        +---------------------+

"""

# ======================================================================================
#                                       IMPORT
# ======================================================================================
from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from typing import NoReturn

from nltk import pos_tag
from nltk.tokenize import sent_tokenize, word_tokenize


# ======================================================================================
#                                    HELPER FUNCTIONS
# ======================================================================================
def panic(msg: str) -> NoReturn:
    print(msg, file=sys.stderr, flush=True)
    sys.exit(1)


def read_text_file(f_path: str) -> str:
    if not os.path.exists(f_path):
        panic(f"File at {f_path} do not exist")
    txt = "ERROR"
    with open(f_path, "r") as txt_file:
        txt = txt_file.read()
    return txt


# ======================================================================================
#                                     NLP STACK
# ======================================================================================


def prepare_tokens(txt: str) -> list[list[str]]:
    return [word_tokenize(sent) for sent in sent_tokenize(txt)]


# ======================================================================================
#                                   CONFIGURATION
# ======================================================================================
@dataclass(frozen=True)
class Config:
    txt_file: str

    @staticmethod
    def from_args() -> Config:
        prsr = argparse.ArgumentParser(description=__doc__)

        prsr.add_argument(
            "txt_file", type=str, help="Path to file with text that will be processed"
        )

        args = prsr.parse_args()
        return Config(txt_file=args.txt_file)


# ======================================================================================
#                                     MAIN
# ======================================================================================
def main():
    cfg = Config.from_args()
    print(cfg)
    txt = read_text_file(cfg.txt_file)
    tokenized_sentences = prepare_tokens(txt)

    for sentence in tokenized_sentences:
        tagged_sent = pos_tag(sentence)
        print("=======================================")
        print(tagged_sent)


if __name__ == "__main__":
    main()
