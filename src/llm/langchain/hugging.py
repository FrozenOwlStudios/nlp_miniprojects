from transformers import pipeline
from langchain_huggingface import HuggingFacePipeline
from langchain_core.prompts import PromptTemplate


generator = pipeline(
    task="text-generation",
    model="Qwen/Qwen2.5-0.5B",
    max_new_tokens=100
)


llm = HuggingFacePipeline(pipeline=generator)

prompt = PromptTemplate.from_template(
    "Answer the following question clearly and briefly:\n"
    "{question}\n"
    "Answer:"
)


chain = prompt | llm


history = []

print("Chatbot ready! Type 'exit' to quit.")

while True:
    user_input = input("\nYou: ")

    if user_input.lower() == "exit":
        print("Goodbye!")
        break

    # Generate a response
    response = chain.invoke({"question": user_input})

    # Save conversation
    history.append({
        "user": user_input,
        "assistant": response
    })

    print(f"Bot: {response}")