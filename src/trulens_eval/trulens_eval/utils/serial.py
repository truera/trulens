# ruff: noqa: E402, F401
"""
!!! warning
    This module is deprecated and will be removed. Use `trulens.core.utils.serial`
    instead.
"""

from trulens.core.utils import deprecation as deprecation_utils

deprecation_utils.packages_dep_warn()

from trulens.core.utils import serial as serial_utilsPath
from trulens.core.utils import serial as serial_utils_BASES
from trulens.core.utils import serial as serial_utilsized
from trulens.core.utils.serial import Collect
from trulens.core.utils.serial import GetAttribute
from trulens.core.utils.serial import GetIndex
from trulens.core.utils.serial import GetIndices
from trulens.core.utils.serial import GetItem
from trulens.core.utils.serial import GetItemOrAttribute
from trulens.core.utils.serial import GetItems
from trulens.core.utils.serial import GetSlice
from trulens.core.utils.serial import Lens
from trulens.core.utils.serial import ParseException
from trulens.core.utils.serial import SerialBytes
from trulens.core.utils.serial import SerialModel
from trulens.core.utils.serial import Step
from trulens.core.utils.serial import StepItemOrAttribute
from trulens.core.utils.serial import all_objects
from trulens.core.utils.serial import all_queries
from trulens.core.utils.serial import leaf_queries
from trulens.core.utils.serial import leafs
from trulens.core.utils.serial import matching_objects
from trulens.core.utils.serial import matching_queries
from trulens.core.utils.serial import model_dump
