# 📦 Release Policies

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

!!! Info "Deprecation Process"

    - 0-6 weeks: Deprecation warning

    - 6-12 weeks: Deprecation message __and error__

    - 12+ weeks: Removal

Changes that result in non-backwards compatible functionality are also reflected
in the version numbering. In such cases, the appropriate level version change
will occur at the introduction of the warning period.

## Currently Deprecated Features

### Legacy Instrumentation

The legacy stack-based instrumentation system is deprecated in favor of
OpenTelemetry (OTEL). This includes:

- `_RecordingContext` class → Use OTEL-based `OtelRecordingContext`
- `Instrument` class-based method wrapping → Use `@instrument` decorator from
  `trulens.core.otel.instrument`
- Custom thread/async context tracking (`TP`, `ThreadPoolExecutor`,
  `tru_new_event_loop`) → OTEL handles context propagation automatically

Some newer features like `App.run()`, `App.input()`, and
`App.instrumented_invoke_main_method()` only work with OTEL instrumentation.

### Core/Session

| Deprecated | Replacement |
|------------|-------------|
| `Tru()` | `TruSession()` |
| `TruCustomApp` | `TruApp` |
| `from trulens.apps.custom import instrument` | `from trulens.apps.app import instrument` |
| Custom `app_id` parameter | Use `app_name` and `app_version` |
| `TruSession.update_record()` | `connector.db.insert_record()` |

### TruSession App Factory Methods

Use `TruSession.App()` instead of these deprecated methods:

- `TruSession.Basic()`
- `TruSession.Custom()`
- `TruSession.Virtual()`
- `TruSession.Chain()`
- `TruSession.Llama()`
- `TruSession.Rails()`

### TruSession Dashboard Methods

| Deprecated | Replacement |
|------------|-------------|
| `TruSession.run_dashboard()` | `trulens.dashboard.run.run_dashboard()` |
| `TruSession.start_dashboard()` | `trulens.dashboard.run.run_dashboard()` |
| `TruSession.stop_dashboard()` | `trulens.dashboard.run.stop_dashboard()` |
| `TruSession.find_unused_port()` | `trulens.dashboard.run.find_unused_port()` |

### Feedback Functions

| Deprecated | Replacement |
|------------|-------------|
| `model_agreement()` | `GroundTruthAgreement(ground_truth, provider)` |
| `qs_relevance()` | `relevance()` |
| `qs_relevance_with_cot_reasons()` | `relevance_with_cot_reasons()` |
| `summarization_with_cot_reasons()` | `comprehensiveness_with_cot_reasons()` |
| Default provider in `GroundTruthAgreement` | Specify provider explicitly |
| `validate_rating()` | Use try/catch with `re_0_10_rating` |
| `Select.context()` | - |

### Endpoint

| Deprecated | Replacement |
|------------|-------------|
| `Endpoint.run_me()` | `Endpoint.run_in_pace()` |

### LangChain App

| Deprecated | Replacement |
|------------|-------------|
| `TruChain.call_with_record()` | - |
| `TruChain.acall_with_record()` | - |

## Previously Deprecated Features

- **`trulens_eval` (removed)**: As of TruLens 1.0, the `trulens_eval` package
  was deprecated in favor of `trulens` and its modular packages. The
  `trulens_eval` package is no longer maintained. See
  [trulens_eval migration](/component_guides/other/trulens_eval_migration) for
  migration details.

- **`QS_RELEVANCE_*` prompts (removed)**: Use `ANSWER_RELEVANCE` or
  `CONTEXT_RELEVANCE` instead.

## Experimental Features

Major new features may be introduced to _TruLens_ first in the form of
experimental previews. Such features are indicated by the prefix `experimental_`
on parameters or the `TruSession.experimental_{enable,disable}_feature` method:

```python
from trulens.core.session import TruSession
from trulens.core.experimental import Feature

session = TruSession()
session.experimental_enable_feature(Feature.SOME_FEATURE)
```

### Experimental Features Pipeline

While in development, experimental features may change in significant ways.
Eventually experimental features get adopted or removed.

For removal, experimental features do not have a deprecation period and will
produce "deprecated" errors instead of warnings.

For adoption, the feature will be integrated somewhere in the API without the
`experimental_` prefix and use of that prefix/flag will instead raise an error
indicating where in the stable API that feature relocated.
