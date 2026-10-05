"""
A simple example of chatbot that allows user to ask for stock and prices.
"""

# ======================================================================================
#                                   IMPORTS
# ======================================================================================
import argparse

from core.shop import Product
from flask import Flask, jsonify, render_template, request
from models.shop import DatabaseModels, ProductModel
from services.chat import prepare_chat_response, prepare_globals
from sqlalchemy import create_engine, select
from sqlalchemy.engine.base import Engine
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column


# ======================================================================================
#                                    ENDPOINTS
# ======================================================================================
def main_site():
    return render_template("index.html")


# ======================================================================================
#                                 DATABASE HELPERS
# ======================================================================================
def prepare_db() -> Engine:
    engine = create_engine("sqlite:///local/demo.db")
    DatabaseModels.metadata.create_all(engine)
    return engine


def clear_db(engine: Engine) -> None:
    with Session(engine) as session:
        session.query(ProductModel).delete(synchronize_session="fetch")
        session.commit()


def populate_db(engine: Engine) -> None:
    products: list[Product] = []
    products.append(Product(name="jacket", price=200, stock=12))
    products.append(Product(name="glasses", price=30.50, stock=2))
    products.append(Product(name="basketball", price=12.99, stock=233))

    with Session(engine) as session:
        for product in products:
            session.add(ProductModel.from_object(product))
        session.commit()


def test_db(engine: Engine) -> None:
    with Session(engine) as session:
        products = session.query(ProductModel).all()
        for product in products:
            print(product.to_object())


# ======================================================================================
#                                     MAIN
# ======================================================================================
def main():
    prsr = argparse.ArgumentParser(description=__doc__)
    prsr.add_argument("-p", "--populate_database", action="store_true")
    args = prsr.parse_args()

    if args.populate_database:
        print("Populating database")
        engine = prepare_db()
        clear_db(engine)
        populate_db(engine)
        test_db(engine)
        return

    app = Flask(__name__)
    app.route("/", methods=["GET", "POST"])(main_site)
    app.route("/api/chat", methods=["POST"])(prepare_chat_response)
    app.run(debug=True)


if __name__ == "__main__":
    main()
