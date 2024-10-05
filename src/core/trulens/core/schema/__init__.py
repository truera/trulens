"""
# Serializable Classes

Note: Only put classes which can be serialized in this module.

## Classes with non-serializable variants

Many of the classes defined here extending serial.SerialModel are meant to be
serialized into json. Most are extended with non-serialized fields in other files.

| Serializable       | Non-serializable        |
| ------------------ | ----------------------- |
| [AppDefinition][trulens.core.schema.app.AppDefinition] | [App][trulens.core.app.App], Tru{Chain, Llama, ...} |
| [FeedbackDefinition][trulens.core.schema.feedback.FeedbackDefinition] | [Feedback][trulens.core.Feedback] |

`AppDefinition.app` is the JSON-ized version of a wrapped app while `App.app` is the
actual wrapped app. We can thus inspect the contents of a wrapped app without
having to construct it. Additionally, JSONized objects like `AppDefinition.app`
feature information about the encoded object types in the dictionary under the
`core/utils/constantx.py:CLASS_INFO` key.

"""

from trulens.core.schema import app as app_schema
from trulens.core.schema import feedback as feedback_schema
from trulens.core.schema import record as record_schema
from trulens.core.schema import select as select_schema

AppDefinition = app_schema.AppDefinition
Select = select_schema.Select
FeedbackDefinition = feedback_schema.FeedbackDefinition
FeedbackResult = feedback_schema.FeedbackResult
FeedbackMode = feedback_schema.FeedbackMode
Record = record_schema.Record

__all__ = [
    "AppDefinition",
    "Select",
    "FeedbackDefinition",
    "FeedbackResult",
    "FeedbackMode",
    "Record",
]
