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

import spacy
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

SUBJECT_DEPENDENCIES = ("nsubj", "nsubjpass")
OBJECT_DEPENDENCIES = ("dobj", "obj", "attr")


@dataclass
class Entity:
    name: str
    label: str

    def __eq__(self, other) -> bool:
        if not isinstance(other, Entity):
            return False

        return self.name == other.name and self.label == other.label


@dataclass
class Relation:
    actor: str
    target: str
    action: str


def nlp_stack(txt: str):
    nlp = spacy.load("en_core_web_sm")
    document = nlp(txt)

    entity_list = [Entity(e.text, e.label_) for e in document.ents]
    entities = []
    for entity in entity_list:
        if entity not in entities:
            entities.append(entity)
    for entity in entities:
        print(entity)

    relations = []
    for word in document:
        if word.pos_ != "VERB":
            continue
        subjects = [t for t in word.children if t.dep_ in SUBJECT_DEPENDENCIES]
        objects = [t for t in word.children if t.dep_ in OBJECT_DEPENDENCIES]

        if len(subjects) == 1 and len(objects) == 1:
            relations.append(Relation(subjects[0].text, objects[0].text, word.text))

    for relation in relations:
        print(relation)


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
    nlp_stack(txt)


if __name__ == "__main__":
    main()
