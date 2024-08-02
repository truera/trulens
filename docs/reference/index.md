# API Reference

TruLens features APIs at three levels which are each intended for different type
of users or work, and have different deprecation schedules. The high-level API
is meant for most _TruLens_ use cases and are maintained for the longest. The
low-level API is for rarer user tasks and are maintained for some amount of
time. The Dev API is the rest of _TruLens_ and is meant for developing the
public functionalities and should not be necessary for most cases except those
which aim to contribute to _TruLens_ itself.

## Top-level API

The top level public api is contained in the `trulens-api` package and exposes
the principal features of each of the _TruLens_ modules.

- [trulens](trulens/api/index.md)

The package only contains a module named `trulens_eval` for backwards
compatibility with the deprecated `trulens_eval` package.

- [trulens_eval](trulens/api/trulens_eval/index.md)

## Low-level API

- [core](trulens/core)
- (optional) [feedback](trulens/feedback)
- (optional) [dashboard](trulens/dashboard)
- (optional) [instrument](instrument/index.md)
- (optional) [providers](providers/index.md)

## Dev API

### Private names

Module members which begin with an underscore `_` are private are should not be
used by code outside of _TruLens_.

Module members which begin with double undescore `__` are class/module private
and should not be used outside of the defining module or class.