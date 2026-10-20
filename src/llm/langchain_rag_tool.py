# =====================================================================================
#                                     IMPORTS
# =====================================================================================
import ollama

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
#                                  RAG AS TOOL
# =====================================================================================

def check_definition(word: str) -> str:
    """Verifies definition of a word

    Args:
        word: Name of the city.
    """
    return f"The term {word} means {knowledge_base[word]}"

# =====================================================================================
#                                  CHAT API
# =====================================================================================

def ask(question: str)-> str:
    response = ollama.chat(
        model="qwen2.5:0.5b",
        messages=[
            {
                "role": "user",
                "content": question
            }
        ],
        tools=[check_definition]
    )


    if response.message.tool_calls:

        for call in response.message.tool_calls:

            print("Tool:", call.function.name)
            print("Arguments:", call.function.arguments)

            if call.function.name == "check_definition":

                result = check_definition(
                    **call.function.arguments
                )

                print("Returning response from tool.")
                return result

    if response.message.content is not None:
        print("Returning response from model.")
        return response.message.content
    print("No response from tool nor from model")
    return "ERROR"


# =====================================================================================
#                                     MAIN
# =====================================================================================
def main()-> None:
    print(ask(input("Question>")))

if __name__ == "__main__":
    main()