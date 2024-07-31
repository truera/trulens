"""
!!! note "Additional Dependency Required"

    To use this module, you must have the `trulens-instrument-nemo` package installed.

    ```bash
    pip install trulens-instrument-nemo
    ```
"""

from trulens.instrument.nemo.tru_rails import RailsActionSelect
from trulens.instrument.nemo.tru_rails import RailsInstrument
from trulens.instrument.nemo.tru_rails import TruRails

__all__ = ["TruRails", "RailsInstrument", "RailsActionSelect"]
