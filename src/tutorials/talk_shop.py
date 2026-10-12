# ======================================================================================
#                                      INCLUDES
# ======================================================================================
from dataclasses import dataclass

import nltk
from lark import Lark, Token


# ======================================================================================
#                                     DATA MODEL
# ======================================================================================
@dataclass
class Product:
    name: str
    price: float
    stock: int


products: dict[str, Product] = {
    "monorail": Product(name="Monorail", price=1449.99, stock=30),
    "hewlett_packard": Product(name="Hewlett Packard", price=1849.99, stock=23),
    "compaq": Product(name="COMPAQ", price=2399.99, stock=8),
}

# ======================================================================================
#                                      NLP
# ======================================================================================
grammar: str = """
%import common.WS
%ignore WS

start : price_question 

price_question : HOW_MUCH BE DET ITEM

HOW_MUCH : "how much"
BE : "is"|"are"

DET : "the"|"a"|"any"

ITEM : /[a-z]+/

"""

parser = Lark(grammar, parser="earley")

tree = parser.parse("how much is the jacket")
print(tree.pretty())


def print_subtree(tree, depth=0):
    if isinstance(tree, Token):
        print("  " * depth + f"{tree.type} = {tree.value}")
        return
    print("  " * depth + tree.data)
    for subtree in tree.children:
        print_subtree(subtree, depth + 1)


print_subtree(tree)
