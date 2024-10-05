"""# Serializable Classes

Note: Only put classes which can be serialized in this module.

# Classes with non-serializable variants

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
