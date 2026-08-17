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
#                                   CONFIGURATION
# ======================================================================================
@dataclass(frozen=True)
class Config:
    txt_file: str
    lines: int | None

    @staticmethod
    def from_args() -> Config:
        prsr = argparse.ArgumentParser(description=__doc__)

        prsr.add_argument(
            "txt_file", type=str, help="Path to file with text that will be processed"
        )

        prsr.add_argument(
            "-n",
            "--lines",
            type=int,
            help="Number of lines that will be processed.",
        )
        args = prsr.parse_args()
        return Config(
            txt_file=args.txt_file,
            lines=args.lines,
        )


POS_CODES_DESCRIPTION = {
    "CC": "coordinating conjunction",
    "CD": "cardinal digit",
    "DT": "determiner",
    "EX": "existential there (like: “there is” … think of it like “there exists”)",
    "FW": "foreign word",
    "IN": "preposition/subordinating conjunction",
    "JJ": "adjective ‘big’",
    "JJR": "adjective, comparative ‘bigger’",
    "JJS": "adjective, superlative ‘biggest’",
    "LS": "list marker 1)",
    "MD": "modal could, will",
    "NN": "noun, singular ‘desk’",
    "NNS": "noun plural ‘desks’",
    "NNP": "proper noun, singular ‘Harrison’",
    "NNPS": "proper noun, plural ‘Americans’",
    "PDT": "predeterminer ‘all the kids’",
    "POS": "possessive ending parent’s",
    "PRP": "personal pronoun I, he, she",
    "PRP$": "possessive pronoun my, his, hers",
    "RB": "adverb very, silently,",
    "RBR": "adverb, comparative better",
    "RBS": "adverb, superlative best",
    "RP": "particle give up",
    "TO,": "to go ‘to’ the store.",
    "UH": "interjection, errrrrrrrm",
    "VB": "verb, base form take",
    "VBD": "verb, past tense, took",
    "VBG": "verb, gerund/present participle taking",
    "VBN": "verb, past participle taken",
    "VBP": "verb, sing. present, non-3d take",
    "VBZ": "verb, 3rd person sing. present takes",
    "WDT": "wh-determiner which",
    "WP": "wh-pronoun who, what",
    "WP$": "possessive wh-pronoun whose",
    "WRB": "wh-adverb where, when",
}

# ======================================================================================
#                                   NLP FUNCTIONS
# ======================================================================================


def prepare_tagged_sentences(txt: str) -> list[list[tuple[str, str]]]:
    raw_sentences = sent_tokenize(txt)
    tokenized_sentences = [word_tokenize(sent) for sent in raw_sentences]
    tagged_sentences = [pos_tag(sent) for sent in tokenized_sentences]
    return tagged_sentences


# ======================================================================================
#                                     MAIN
# ======================================================================================
def main():
    cfg = Config.from_args()
    print(cfg)

    txt = read_text_file(cfg.txt_file)
    tagged_sentences = prepare_tagged_sentences(txt)
    for sentence in tagged_sentences:
        for word in sentence:
            try:
                tag = POS_CODES_DESCRIPTION[word[1]]
                print(f"{word[0]} - {word[1]} ({tag})")
            except KeyError:
                print(f"{word[0]} - {word[1]}")


if __name__ == "__main__":
    main()
