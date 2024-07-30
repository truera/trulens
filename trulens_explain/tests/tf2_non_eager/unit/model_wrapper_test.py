import os

os.environ["TRULENS_BACKEND"] = "tensorflow"

from unittest import main

import tensorflow as tf

assert not tf.executing_eagerly()
if __name__ == "__main__":
    main()
