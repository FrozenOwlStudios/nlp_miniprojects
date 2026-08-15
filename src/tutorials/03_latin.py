# ======================================================================================
#                                        IMPORTS
# ======================================================================================
from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from typing import NoReturn


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

    @staticmethod
    def from_args() -> Config:
        prsr = argparse.ArgumentParser(description=__doc__)

        prsr.add_argument(
            "txt_file", type=str, help="Path to file with text that will be processed"
        )

        prsr.add_argument(
            "-l",
            "--lines",
            type=int,
            help="Number of lines that will be processed.",
        )
        args = prsr.parse_args()
        return Config(txt_file=args.txt_file, lines=args.lines)


# ======================================================================================
#                                     MAIN
# ======================================================================================
def main():
    cfg = Config.from_args()
    print(cfg)
    txt = read_text_file(cfg.txt_file, cfg.lines)
    print(txt)


if __name__ == "__main__":
    main()
