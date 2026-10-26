# =====================================================================================
#                                       INCLUDES
# =====================================================================================
from __future__ import annotations

from argparse import ArgumentParser, ArgumentTypeError, Namespace
import logging
from dataclasses import dataclass
from pathlib import Path

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama

# =====================================================================================
#                                     LOGGER
# =====================================================================================
logger = logging.getLogger("rag_chat")


LOG_FORMAT: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"


def initialize_logger(
    logfile: Path | None, log_to_console: bool, local_log_level: int, global_log_level
) -> None:

    formatter = logging.Formatter(LOG_FORMAT)

    # Adding handlers to root logger so we can get logs from other modules
    root_logger = logging.getLogger()
    root_logger.setLevel(global_log_level)
    logger.setLevel(local_log_level)

    if logfile:
        fh = logging.FileHandler(logfile)
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        root_logger.addHandler(fh)

    if log_to_console:
        ch = logging.StreamHandler()
        ch.setLevel(logging.DEBUG)
        ch.setFormatter(formatter)
        root_logger.addHandler(ch)


# ======================================================================================
#                                   CONFIGURATION
# ======================================================================================


def arg_file_path(path_str: str) -> Path:
    path = Path(path_str)
    if not path.parent.exists() or not path.parent.is_dir():
        raise ArgumentTypeError(
            f"Directory {path.parent.absolute()} do not exist. Cannot place logfile inside"
        )
    if path.is_dir():
        raise ArgumentTypeError(
            f"Path {path.absolute()} is a directory. Provide path directly to logfile."
        )
    return path


def arg_log_level(level_name: str) -> int:
    LOGGING_LEVELS: dict[str, int] = {
        "debug": logging.DEBUG,
        "info": logging.INFO,
        "warning": logging.WARNING,
        "error": logging.ERROR,
        "critical": logging.CRITICAL,
    }

    if level_name not in LOGGING_LEVELS:
        raise ArgumentTypeError(
            f'Value "{level_name}" is incorrect log level,'
            f" available options {LOGGING_LEVELS.keys()}"
        )
    return LOGGING_LEVELS[level_name]


@dataclass(frozen=True)
class ConfigLogger:
    log_file: Path | None
    log_to_console: bool
    log_level_chat: int
    log_level_global: int

    @staticmethod
    def add_args_to_parser(prsr: ArgumentParser) -> None:
        prsr.add_argument(
            "--log_file",
            type=arg_file_path,
            default=None,
            help="Path to file where logs will be written",
        )
        prsr.add_argument(
            "--log_to_console",
            action="store_true",
            help="If set logs will be printed to console.",
        )
        prsr.add_argument(
            "--log_level_chat",
            type=arg_log_level,
            default=logging.INFO,
            help="Lowest level of severity that will be logged by chat module.",
        )
        prsr.add_argument(
            "--log_level_global",
            type=arg_log_level,
            default=logging.INFO,
            help="Lowest level of severity that will be logged by other modules.",
        )

    @staticmethod
    def from_arguments(args: Namespace) -> ConfigLogger:
        return ConfigLogger(
            log_file=args.log_file,
            log_to_console=args.log_to_console,
            log_level_chat=args.log_level_chat,
            log_level_global=args.log_level_global,
        )


@dataclass(frozen=True)
class Config:
    log: ConfigLogger

    @staticmethod
    def from_args(argv: list[str] | None = None) -> Config:
        prsr = ArgumentParser(description=__doc__)

        ConfigLogger.add_args_to_parser(prsr)

        args = prsr.parse_args(argv)
        log = ConfigLogger.from_arguments(args)

        return Config(log=log)


# =====================================================================================
#                                      CHAT CORE
# =====================================================================================


# =====================================================================================
#                                        MAIN
# =====================================================================================
def main(argv: list[str] | None = None) -> None:
    cfg = Config.from_args(argv)
    initialize_logger(
        cfg.log.log_file,
        cfg.log.log_to_console,
        cfg.log.log_level_chat,
        cfg.log.log_level_global,
    )
    logger.info("Booting chat with config : %s", cfg)


if __name__ == "__main__":
    main()
