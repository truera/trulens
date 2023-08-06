from sqlalchemy import Column, Text, VARCHAR
from sqlalchemy.orm import declarative_base
from trulens_eval import schema

Base = declarative_base()


class App(Base):
    __tablename__ = "apps"
    app_id = Column(VARCHAR(256), nullable=False, primary_key=True)
    app_json = Column(Text, nullable=False)

    @classmethod
    def parse(cls, obj: schema.AppDefinition) -> "App":
        return cls(app_id=obj.app_id, app_json=obj.json())
