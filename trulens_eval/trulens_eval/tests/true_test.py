import unittest
from models.mock_model_factories import *
from trulens_eval.tru import Tru


class TrueTestCase(unittest.TestCase):
    def setUp(self):
        self.tru = Tru()
        self.app_defination = AppDefinitionFactory.create_batch(3)
        self.record = RecordFactory.create_batch(3)
        self.feedback_definiition = FeedbackDefinitionFactory.create_batch(3)
        self.feedback_result = FeedbackResultFactory.create_batch(3)

    # def test_list_record_when_invalid(self):
    #     record = self.tru.list_records(random.randint(100000, 1000000))
    #     self.assertEqual(record, [])

    def test_list_record_when_valid(self):

        print(self.tru.db.query(orm.Record).all())
        record = self.tru.list_records(self.record[0].app.app_id)
        print(self.record[0].app.__dict__)
        print(self.record[0].app.app_id)
        print(self.tru.list_records(self.record[0].app.app_id))

        self.assertEqual(record, [self.record[0].record_id])


if __name__ == '__main__':
    unittest.main()
