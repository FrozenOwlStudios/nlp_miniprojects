# ======================================================================================
#                                        IMPORTS
# ======================================================================================
from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from typing import NoReturn

from cltk import NLP


# ======================================================================================
#                                    HELPER FUNCTIONS
# ======================================================================================
def panic(msg: str) -> NoReturn:
    print(msg, file=sys.stderr, flush=True)
    sys.exit(1)


def read_text_file(f_path: str, lines: int | None = None) -> str:
    if not os.path.exists(f_path):
        panic(f"File at {f_path} do not exist")
    content: list[str] = []
    lines_read = 0
    with open(f_path, "r") as txt_file:
        for line in txt_file:
            if lines and lines_read >= lines:
                break
            content.append(line)
            lines_read += 1
    txt = "".join(content)
    txt = txt.replace("\n", "").replace(".", ".\n")
    return txt


# ======================================================================================
#                                   CONFIGURATION
# ======================================================================================
@dataclass(frozen=True)
class Config:
    txt_file: str
    lines: int | None
    lang_code: str

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
        prsr.add_argument("-l", "--lang_code", type=str, default="lati1261")
        args = prsr.parse_args()
        return Config(
            txt_file=args.txt_file, lines=args.lines, lang_code=args.lang_code
        )


# ======================================================================================
#                                     MAIN
# ======================================================================================
def main():
    cfg = Config.from_args()
    print(cfg)
    txt = read_text_file(cfg.txt_file, cfg.lines)
    print(txt)
    nlp = NLP(
        cfg.lang_code,
        backend="stanza",
        suppress_banner=True,
    )

    return nlp.analyze(txt)


if __name__ == "__main__":
    main()
