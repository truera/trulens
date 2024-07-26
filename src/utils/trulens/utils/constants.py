"""
This module contains common constants used throughout the trulens
"""

# Field/key name used to indicate a circular reference in dictified objects.
CIRCLE = "__tru_circular_reference"

# Field/key name used to indicate an exception in property retrieval (properties
# execute code in property.fget).
ERROR = "__tru_property_error"

# Key for indicating non-serialized objects in json dumps.
NOSERIO = "__tru_non_serialized_object"

# Key of structure where class information is stored.
CLASS_INFO = "tru_class_info"

ALL_SPECIAL_KEYS = set([CIRCLE, ERROR, CLASS_INFO, NOSERIO])
