# 📦 Release policies

## Versioning

Releases are organized in `<major>.<minor>.<patch>` style. A release is made
about every week around tuesday-thursday. Releases increment the `minor` version
number. Occasionally bug-fix releases occur after a weekly release. Those
increment only the `patch` number. No releases have yet made a `major` version
increment. Those are expected to be major releases that introduce a large number
of breaking changes.

## Deprecation

Changes to the public API are governed by a deprecation process in three stages.
In the warning period of no less than _6 weeks_, the use of a deprecated
package, module, or value will produce a warning but otherwise operate as
expected. In the subsequent deprecated period of no less than _6 weeks_, the
use of that component will produce an error after the deprecation message. After
these two periods, the deprecated capability will be completely removed.

!!! Info "Deprecation process"

    - 6 weeks: deprecation warning

    - 6 weeks: deprecation message __and error__

    - removal

Changes that result in non-backwards compatible functionality are also reflected
in the version numbering. In such cases, the appropriate level version change
will occur at the introduction of the warning period.

## Currently deprecating features

- Starting 1.0, the `trulens_eval` package is being deprecated in favor of
  `trulens` and several associated required and optional packages. See
  [trulens_eval migration](/trulens/guides/trulens_eval_migration) for details.

    - Warning period: 2024-09-01 (`trulens-eval==1.0.1`) to 2024-10-14.
    Backwards compatibility during the warning period is provided by the new
    content of the `trulens_eval` package which provides aliases to the features
    in their new locations. See
    [trulens_eval](trulens/api/trulens_eval/index.md).

    - Deprecated period: 2024-10-14 to 2025-12-01. Usage of `trulens_eval` will
  	produce errors indicating deprecation.

    - Removed expected 2024-12-01 Installation of the latest version of
  	`trulens_eval` will be an error itself with a message that `trulens_eval` is
    no longer maintained.

## Experimental Features

Major new features are introduced to _TruLens_ first in the form of experimental
previews. Such features are indicated by the prefix `experimental_`. For
example, the OTEL exporter for `TruSession` is specified with the
`experimental_otel_exporter` parameter. Some features require additionally
setting a flag before they are enabled. This is controlled by the
`TruSession.experimental_{enable,disable}_feature` method:

```python
from trulens.core.session import TruSession
session = TruSession()
session.experimental_enable_feature("otel_tracing")

# or
from trulens.core.experimental import Feature
session.experimental_disable_feature(Feature.OTEL_TRACING)
```

If an experimental parameter like `experimental_otel_exporter` is used, some
experimental flags may be set. For the OTEL exporter, the `OTEL_EXPORTER` flag
is required and will be set.

Some features cannot be changed after some stages in the typical _TruLens_
use-cases. OTEL tracing, for example, cannot be disabled once an app has been
instrumented. An error will result in an attempt to change the feature after it
has been "locked" by irreversable steps like instrumentation.

### Experimental Features Pipeline

While in development, the experimental features may change in significant ways.
Eventually experimental features get adopted or removed.

For removal, experimental features do not have a deprecation period and will
produce "deprecated" errors instead of warnings.

For adoption, the feature will be integrated somewhere in the API without the
`experimental_` prefix and use of that prefix/flag will instead raise an error
indicating where in the stable API that feature relocated.
