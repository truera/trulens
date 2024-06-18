from trulens_eval.react_components.record_viewer import record_viewer
from trulens_eval.schema.record import Record

from trulens_eval import Tru

import json

tru = Tru()

def trulens_trace(record: Record):
    app_json = tru.get_app(app_id=record.app_id)
    record_json = _get_record_json(record)
    record_viewer(record_json=record_json, app_json=app_json)

# a bit hacky, probably a better way to do this
def _get_record_json(record):
    records, feedback = tru.get_records_and_feedback()
    record_json = records.loc[records['record_id'] == record.record_id]['record_json'].values[0]
    return json.loads(record_json)
