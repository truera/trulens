# Experimental Features

Each folder/model here corresponds to an experimental feature. These are
controlled/enabled by `TruSession.experimental_feature_enable` and related
methods with flags enumerated by `trulens.core.experimental.Feature`.

The contents of each folder is organized the same way as the main `trulens`
namespace indicating the locations where experimental features are to be
integrated once they graduate from experimental. For example
`trulens/experimental/otel_tracing.core/session.py` contains things which will be
integrated to `trulens/core/session.py`. Inside these duplicated paths,
duplicated class names are also used to indicate the classes where the
experimental code is to be integrated. For example `_TruSession` contains
experimental code to be integrated into `TruSession` after the experimental
phase.
