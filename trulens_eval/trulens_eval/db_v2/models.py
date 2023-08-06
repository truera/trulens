from sqlalchemy import Column, Text, VARCHAR
from sqlalchemy.orm import declarative_base
from trulens_eval import schema

Base = declarative_base()


class AppDefinition(Base):
    __tablename__ = "apps"
    app_id = Column(VARCHAR(256), nullable=False, primary_key=True)
    app_json = Column(Text, nullable=False)

    @classmethod
    def parse(cls, obj: schema.AppDefinition) -> "AppDefinition":
        return cls(app_id=obj.app_id, app_json=obj.json())


class FeedbackDefinition(Base):
    __tablename__ = "feedback_defs"
    feedback_definition_id = Column(VARCHAR(256), nullable=False, primary_key=True)
    feedback_json = Column(Text, nullable=False)

    @classmethod
    def parse(cls, obj: schema.FeedbackDefinition) -> "FeedbackDefinition":
        return cls(app_id=obj.feedback_definition_id, app_json=obj.json())
