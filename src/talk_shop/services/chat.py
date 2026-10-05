# ======================================================================================
#                                   IMPORTS
# ======================================================================================
import nltk
from flask import jsonify, request
from sqlalchemy import create_engine, select
from sqlalchemy.engine.base import Engine

# ======================================================================================
#                                   GLOBALS
# ======================================================================================

parser: nltk.ChartParser | None = None
database: Engine | None = None


def prepare_globals(database_path: str, grammar_path: str) -> None:
    global parser, database
    database = create_engine(f"sqlite:///{database_path}")
    grammar_str = ""
    with open(grammar_path, "r") as grammar_file:
        grammar_str = grammar_file.read()
    grammar = nltk.CFG.fromstring(grammar_str)
    parser = nltk.ChartParser(grammar)


def prepare_chat_response():
    data = request.get_json()

    user_message = data["message"]

    response = "Hello! This is a response from the Python backend."

    return jsonify({"response": response})
