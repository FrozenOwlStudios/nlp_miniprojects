# =====================================================================================
#                                       INCLUDES
# =====================================================================================
from __future__ import annotations

from argparse import ArgumentParser, ArgumentTypeError
import logging
from dataclasses import dataclass
from pathlib import Path

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama

# =====================================================================================
#                                     LOGGER
# =====================================================================================
logger = logging.getLogger("rag_chat")

LOGGING_LEVELS: dict[str, int] = {
    "debug": logging.DEBUG,
    "info": logging.INFO,
    "warning": logging.WARNING,
    "error": logging.ERROR,
    "critical": logging.CRITICAL,
}

LOG_FORMAT: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"


def initialize_logger(
    logfile_path: str, local_log_level: int, global_log_level
) -> None:

    fh = logging.FileHandler(logfile_path)
    fh.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)

    formatter = logging.Formatter(LOG_FORMAT)
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)

    root_logger = logging.getLogger()
    root_logger.setLevel(global_log_level)
    root_logger.addHandler(fh)
    root_logger.addHandler(ch)

    logger.setLevel(local_log_level)
    logger.addHandler(fh)
    logger.addHandler(ch)


# ======================================================================================
#                                   CONFIGURATION
# ======================================================================================


def arg_file_path(path_str: str) -> Path:
    path = Path(path_str)
    if not path.parent.exists() or not path.parent.is_dir():
        raise ArgumentTypeError(
            f"Directory {path.parent.absolute()} do not exist. Cannot place logfile inside"
        )
    return path


@dataclass(frozen=True)
class Config:
    log_file: Path | None

    @staticmethod
    def from_args(argv: list[str] | None = None) -> Config:
        prsr = ArgumentParser(description=__doc__)
        prsr.add_argument(
            "--log_file",
            type=arg_file_path,
            default=None,
            help="Path to file where logs will be written",
        )

        args = prsr.parse_args(argv)
        return Config(log_file=args.log_file)


# =====================================================================================
#                                        MAIN
# =====================================================================================
def main(argv: list[str] | None = None) -> None:
    cfg = Config.from_args(argv)
    print(cfg)


if __name__ == "__main__":
    main()
