# =====================================================================================
#                                     IMPORTS
# =====================================================================================
import ollama
import logging

# =====================================================================================
#                                     LOGGER
# =====================================================================================
logger = logging.getLogger("rag_chat")

def initialize_logger()-> None:
    global logger
    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler("local/chat.log")
    fh.setLevel(logging.DEBUG)
    # create console handler with a higher log level
    ch = logging.StreamHandler()
    ch.setLevel(logging.ERROR)
    # create formatter and add it to the handlers
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    # add the handlers to the logger
    # logger.addHandler(fh)
    # logger.addHandler(ch)
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    root_logger.addHandler(fh)
    root_logger.addHandler(ch)

# =====================================================================================
#                                  KNOWLEDGE BASE
# =====================================================================================
knowledge_base = {
    "python": "Python is a programming language created by Guido van Rossum.",
    "ollama": "Ollama is a tool for running large language models locally.",
    "rag": "RAG stands for Retrieval-Augmented Generation. "
    "It combines information retrieval with LLM generation.",
}


# =====================================================================================
#                                  CONTEXT BUILDER
# =====================================================================================
def prepare_context(question: str)-> str:
    logger.info(f"RUNNING RAG - > {question}")
    question = question.lower()

    for keyword, definition in knowledge_base.items():
        if keyword in question:
            logger.info(f"RAG RESPONSE -> {definition}")
            return definition

    logger.info(f"RAG RESPONSE -> No relevant information found.")
    return "No relevant information found."

# =====================================================================================
#                                      CHAT 
# =====================================================================================
def ask(question: str)-> str:
    context = prepare_context(question)

    prompt = f"""
    Answer the question using ONLY the context below.
    If the context doesn't contain the answer, say "I don't know".

    Context:
    {context}

    Question:
    {question}
    """

    response = ollama.chat(
        model="qwen2.5:0.5b",
        messages=[{"role": "user", "content": prompt}],
        options={
            "temperature": 0,
            "num_ctx": 2048,
        },
    )

    return response["message"]["content"]

def run_test_questions()-> None:
    questions = ["Who invented python ?", "What is RAG?", "Who invented Java?"]
    for question in questions:
        print(f"QUESTION -> {question}")
        print(f"RESPONSE -> {ask(question)}")

# =====================================================================================
#                                    MAIN 
# =====================================================================================
def main()->None:
    initialize_logger()
    run_test_questions()

if __name__ == "__main__":
    main()