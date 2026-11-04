from transformers import pipeline
from langchain_huggingface import HuggingFacePipeline, ChatHuggingFace
from langchain.messages import HumanMessage, SystemMessage, AIMessage
from transformers import BitsAndBytesConfig

# =====================================================================================
#                                     CONFIG
# =====================================================================================

#MODEL_NAME = "Qwen/Qwen2.5-0.5B"
MODEL_NAME = "textilelabs/Loom-Swift-Micro"
AGENT_INSTRUCTIONS = """
You are helpful assistant. Try to answer question relying on context and 
chat history. If you cannot do so just say that you do not know or are unsure.
"""

# =====================================================================================
#                                   KNOWLEDGE BASE
# =====================================================================================

knowledge_base: dict[str, str] = {
    "rag": "RAG stands for Retrieval-Augmented Generation. "
    "It combines information retrieval with text generation.",
    "llm": "LLM stands for Large Language Model. "
    "It is a neural network trained to understand and generate text.",
    "transformer": "A Transformer is a neural network architecture "
    "based on attention mechanisms.",
    "qwen": "Qwen is a family of language models developed by Alibaba.",
}


def retrieve_context(query: str) -> str:
    results = []

    for term, definition in knowledge_base.items():
        if term.lower() in query.lower():
            results.append(f"{term}: {definition}")

    return "\n".join(results)


def prepare_system_message(instructions: str, context: str) -> SystemMessage:
    return SystemMessage(
        content=f"""
    {instructions}
    Context:
    {context}
    """
    )


# =====================================================================================
#                                          NLP
# =====================================================================================
def prepare_chat():
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype="float16",
        bnb_4bit_use_double_quant=True,
    )

    llm = HuggingFacePipeline.from_model_id(
        model_id="Qwen/Qwen2.5-0.5B",
        task="text-generation",
        pipeline_kwargs=dict(
            max_new_tokens=512,
            do_sample=False,
            repetition_penalty=1.03,
            return_full_text=False,
        ),
        model_kwargs={"quantization_config": quantization_config},
    )
    chat = ChatHuggingFace(llm=llm)
    return chat


# =====================================================================================
#                                         MAIN
# =====================================================================================


def main():

    chat = prepare_chat()

    s_msg = prepare_system_message(AGENT_INSTRUCTIONS, "")
    chat_history: list[SystemMessage | HumanMessage | AIMessage] = [s_msg]

    print("Chatbot ready! Type 'exit' to quit.")

    while True:
        user_input = input("\nYou: ")

        if user_input.lower() == "exit":
            print("Goodbye!")
            break

        context = retrieve_context(user_input.lower())
        chat_history[0] = prepare_system_message(AGENT_INSTRUCTIONS, context)
        chat_history.append(HumanMessage(user_input))
        response = chat.invoke(chat_history)
        chat_history.append(response)
        print(f"Bot: {response.content}")


if __name__ == "__main__":
    main()
