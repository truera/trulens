# Release policies

## Versioning

Releases are organized in `<major>.<minor>.<patch>` style. A release is made
about every week around tuesday-thursday. Releases increment the `minor` version
number. Occasionally bug-fix releases occur after a weekly release. Those
increment only the `patch` number. No releases have yet made a `major` version
increment. Those are expected to be major releases that introduce large number
of breaking changes.

## Deprecation

Changes to the public API are governed by a deprecation process in three stages.
In the warning period of no less than _3 months_, the use of a deprecated
package, module, or value will produce a warning but otherwise operate as
expected. In the subsequent deprecated period of no less than _3 months_, the
use of that component will produce an error after the deprecation message. After
these two periods, the deprecated capability will be completely removed.

!!! Info "Deprecation process"

    - 3 months - warning

    - 3 months - warning + error

    - removal

Changes that result in non-backwards compatible functionality are also reflected
in the version numbering. In such cases, the appropriate level version change
will occur at the introduction of the warning period.

## Currently deprecating features

- Starting 1.0.0, the `trulens_eval` package is being deprecated in favor of
  `trulens` and several associated required and optional packages. See
  [trulens_eval migration](/trulens/guides/trulens_eval_migration) for details.

    - Warning period: 2024-09-01 with `trulens-eval` 1.0.0a0 -- 2024-12-01.
      Backwards compatibility during the warning period is provided by the new
      content of the `trulens_eval` package which provides aliases to the
      features in their new locations. See
      [trulens_eval](trulens/api/trulens_eval/index.md).

    - Deprecated period: 2024-09-01 -- 2025-02-01 . Usage of `trulens_eval` will
      produce warnings and errors.

    - Removed expected 2024-02-01 Installation of the latest version of
      `trulens_eval` will be an error itself.
