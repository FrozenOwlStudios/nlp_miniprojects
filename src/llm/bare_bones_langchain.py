# ======================================================================================
#                                       INCLUDES
# ======================================================================================
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama


# ======================================================================================
#                                        MAIN
# ======================================================================================
def main():
    # Initilaizing model
    llm = ChatOllama(
        model="hf.co/textilelabs/Loom-Swift-Micro:F16",
        temperature=0.7,
    )

    # Simple memoryless chat
    print('Initializing chat, type "EXIT" to exit.')
    while True:
        user_prompt = input("User : ")
        if user_prompt == "EXIT":
            break

        # Chat history only contains single user prompt (so no memory)
        chat_history = [
            SystemMessage(content="You are helpfull asistant."),
            HumanMessage(content=user_prompt),
        ]

        response = llm.invoke(chat_history)
        print(f"Model : {response.content}")


if __name__ == "__main__":
    main()
