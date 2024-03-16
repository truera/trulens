
import factory
from faker import Faker

from trulens_eval.database import orm
from faker_enum import EnumProvider
from trulens_eval.schema import FeedbackResultStatus

import random


faker = Faker()
faker.add_provider(EnumProvider)




class AppDefinitionFactory(factory.Factory):
    class Meta:
        model = orm.AppDefinition

    app_id = random.randint(1000, 10000)
    app_json = faker.json()


class FeedbackDefinitionFactory(factory.Factory):
    class Meta:
        model = orm.FeedbackDefinition

    feedback_definition_id = random.randint(1000, 10000)
    feedback_json = faker.json()


class RecordFactory(factory.Factory):
    class Meta:
        model = orm.Record
    record_id = random.randint(10000, 10000)
    app =  factory.SubFactory(AppDefinitionFactory)
    input = faker.text()
    output = faker.text()
    record_json = faker.json()
    tags = faker.text()
    ts = int(faker.date_time().timestamp())
    cost_json = faker.json()
    perf_json = faker.json()

class FeedbackResultFactory(factory.Factory):
    class Meta:
        model = orm.FeedbackResult
    feedback_definition=factory.SubFactory(FeedbackDefinitionFactory)
    record=factory.SubFactory(RecordFactory)
    feedback_definition_id =random.randint(10000, 100000)
    last_ts = int(faker.date_time().timestamp())
    status = faker.enum(FeedbackResultStatus)
    error = faker.text()
    calls_json = faker.json()
    result = faker.pyfloat(left_digits=3, right_digits=2, positive=True)
    name =  faker.text()
    cost_json = faker.json()
    multi_result = faker.json()

