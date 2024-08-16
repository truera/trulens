"""Serializable dataset-related classes."""

import logging

from trulens.core.utils import pyschema
from trulens.core.utils import serial

logger = logging.getLogger(__name__)


class Dataset(pyschema.WithClassInfo, serial.SerialModel):
    pass
