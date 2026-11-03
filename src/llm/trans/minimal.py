# =====================================================================================
#                                      IMPORTS
# =====================================================================================
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# =====================================================================================
#                                     CONFIG
# =====================================================================================

MODEL_NAME = "Qwen/Qwen2.5-0.5B"

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


# =====================================================================================
#                                    CHAT
# =====================================================================================


def main():

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype="auto", device_map="auto"
    )

    # Simple retrieval function

    messages = [{"role": "system", "content": "You are a helpful assistant."}]

    print("Chatbot ready! Type 'exit' to quit.\n")

    while True:
        user_input = input("You: ").strip()

        if user_input.lower() in ["exit", "quit"]:
            break

        if not user_input:
            continue

        # Retrieve relevant definitions
        context = retrieve_context(user_input)

        # Augment user message with retrieved knowledge
        if context:
            augmented_input = f"""
    Use the following information to answer the question:

    {context}

    Question: {user_input}
    """
        else:
            augmented_input = user_input

        # Add augmented message
        messages.append({"role": "user", "content": augmented_input})

        # Apply chat template
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        inputs = tokenizer(text, return_tensors="pt").to(model.device)

        # Generate response
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=tokenizer.eos_token_id,
            )

        response_tokens = outputs[0][inputs.input_ids.shape[-1] :]

        response = tokenizer.decode(response_tokens, skip_special_tokens=True)

        print(f"Bot: {response}\n")

        messages.append({"role": "assistant", "content": response})


if __name__ == "__main__":
    main()
