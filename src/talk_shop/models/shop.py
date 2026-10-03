# ======================================================================================
#                                       IMPORTS
# ======================================================================================
from __future__ import annotations

from core.shop import Product
from sqlalchemy import Float, Integer, String, create_engine, select
from sqlalchemy.engine.base import Engine
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column


# ======================================================================================
#                                       MODELS
# ======================================================================================
class DatabaseModels(DeclarativeBase):
    pass


class ProductModel(DatabaseModels):
    __tablename__ = "products"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(100))
    price: Mapped[float] = mapped_column(Float())
    stock: Mapped[int] = mapped_column(Integer())

    def to_object(self) -> Product:
        return Product(
            name=str(self.name), price=float(self.price), stock=int(self.stock)
        )

    @staticmethod
    def from_object(p: Product) -> ProductModel:
        return ProductModel(name=p.name, price=p.price, stock=p.stock)
