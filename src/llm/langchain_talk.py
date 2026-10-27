# =====================================================================================
#                                       INCLUDES
# =====================================================================================
from __future__ import annotations

from argparse import ArgumentParser, ArgumentTypeError, Namespace
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, BaseMessage
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate

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


def arg_llm_temperature(txt: str) -> float:
    try:
        val = float(txt)
        if val < 0:
            raise ArgumentTypeError(
                f"LLM temperature must be no less than 0 ({val} given)."
            )
        return val
    except ValueError as e:
        raise ArgumentTypeError(
            f'String "{txt}" is not a valid floating point number.'
        ) from e


@dataclass(frozen=True)
class ConfigChatbot:
    llm_model: str
    llm_temp: float
    llm_sys_msg: str

    @staticmethod
    def add_args_to_parser(prsr: ArgumentParser) -> None:
        prsr.add_argument(
            "--llm_model",
            type=str,
            required=True,
            help="Name of LLM model that will be used (DUMMY for fake model placeholder)",
        )
        prsr.add_argument(
            "--llm_temp",
            type=arg_llm_temperature,
            default=0.7,
            help="LLM temperature (range from 0.7 to 0.8 for normal "
            "responses 0.0 for fully deterministic and above 0.8 for "
            "creative but unreliable ones).",
        )
        prsr.add_argument(
            "--llm_sys_msg",
            type=str,
            default="You are a person trying to maintain a conversation in private (1 on 1) chat channel",
            help="Description that informs llm what role it is playing.",
        )

    @staticmethod
    def from_arguments(args: Namespace) -> ConfigChatbot:
        return ConfigChatbot(
            llm_model=args.llm_model,
            llm_temp=args.llm_temp,
            llm_sys_msg=args.llm_sys_msg,
        )


@dataclass(frozen=True)
class Config:
    log: ConfigLogger
    chat: ConfigChatbot

    @staticmethod
    def from_args(argv: list[str] | None = None) -> Config:
        prsr = ArgumentParser(description=__doc__)

        ConfigLogger.add_args_to_parser(prsr)
        ConfigChatbot.add_args_to_parser(prsr)

        args = prsr.parse_args(argv)
        log = ConfigLogger.from_arguments(args)
        chat = ConfigChatbot.from_arguments(args)

        return Config(log=log, chat=chat)


# =====================================================================================
#                                      CHAT CORE
# =====================================================================================
class Chatbot:
    _model: ChatOllama | None
    _history: list[BaseMessage]

    def __init__(self, model: str, temperature: float, system_msg: str) -> None:
        self._model = None
        if model != "DUMMY":
            self._model = ChatOllama(model=model, temperature=temperature)
        self._history = [SystemMessage(system_msg)]

    def talk(self, user_prompt: str) -> str:
        self._history.append(HumanMessage(user_prompt))
        if not self._model:
            dummy_response = "Dummy model response"
            self._history.append(AIMessage(dummy_response))
            return dummy_response

        response: AIMessage = self._model.invoke(self._history)
        self._history.append(response)
        return str(response.content)

    def get_full_prompt(self) -> str:
        prompt = ChatPromptTemplate.from_messages(self._history)
        return prompt.pretty_repr()

    def __getitem__(self, key: int) -> str:
        if not isinstance(key, int):
            raise KeyError(
                f"Attempting to use {type()} instead of int as key for chat history"
            )
        if key < 0 or key >= (len(self._history) - 1):
            raise IndexError(
                f"Valid chat history indexes are from 0 to {len(self._history) - 1} "
                f"got {key} instead."
            )
        content: str | list[str | dict[Any, Any]] = self._history[key].content
        return str(content)


# =====================================================================================
#                                      CHAT INTERFACE
# =====================================================================================

def chat_cmd_exit(_cmd: str, _chat:Chatbot)->bool:
    return False

CHATBOT_ESCAPE_CHAR = "!"

chat_commands: dict[str, Callable[[str, Chatbot],bool]] = {
    "exit": chat_cmd_exit,
    "end": chat_cmd_exit,
    "stop": chat_cmd_exit
}

def run_conversation(chat: Chatbot)-> None:
    running: bool = True
    while running:
        user_msg: str = input("User > ")
        user_msg = user_msg.strip()
        # Empty user input
        if not user_msg:
            continue
        # Handle commands
        if user_msg[0] == CHATBOT_ESCAPE_CHAR:
            cmd = user_msg[1:].lower()
            try:
                running = chat_commands[cmd](cmd, chat)
            except KeyError:
                print(f"Unknown command : {cmd}")
            continue
        # Normal chat interaction
        chat_msg = chat.talk(user_msg)
        print(f"Chatbot > {chat_msg}")

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
    chat = Chatbot(cfg.chat.llm_model, cfg.chat.llm_temp, cfg.chat.llm_sys_msg)
    run_conversation(chat)


if __name__ == "__main__":
    main()
