from sqlalchemy import Column, Text
from sqlalchemy.orm import declarative_base

Base = declarative_base()


class App(Base):
    __tablename__ = "apps"
    app_id = Column(Text, nullable=False, primary_key=True)
    app_json = Column(Text, nullable=False)
