"""
A example of tokenizers.
"""

# =======================================================================================
#                                      IMPORTS
# =======================================================================================
import nltk
from nltk.tokenize import word_tokenize

# =======================================================================================
#                                     GLOBALS
# =======================================================================================
EXAMPLE_SENTENCE: str = (
    "A wizard is never late, Frodo Baggins."
    " Nor is he early. He arrives precisely when he means to."
)


# =======================================================================================
#                                  WORD TOKENIZE
# =======================================================================================
def word_tokenize_demo():
    print(f"Sentence : {EXAMPLE_SENTENCE}")
    tokens = word_tokenize(EXAMPLE_SENTENCE)
    print(f"Token count : {len(tokens)}")
    print(f"Tokens : {tokens}")


# =======================================================================================
#                                      MAIN
# =======================================================================================
demo_selector = [
    word_tokenize_demo,
]


def main():
    print("Select demo:")
    print("   1 - word tokenize")
    selection = input("SELECTION>")
    try:
        idx = int(selection) - 1
        demo_selector[idx]()
    except Exception:
        print(f"There is no possible selection {selection}.")
        print(f"Use number from 1 to {len(demo_selector)}.")


if __name__ == "__main__":
    main()
